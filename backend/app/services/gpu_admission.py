"""
GPU admission — the "waiting for a GPU to be assigned" behaviour.

The scenario this exists for: a training run is going in project A, and the
user walks over to project B and starts another training (or an auto-annotate
batch). Before this module, nothing stopped the second job — the two would
fight for the card and one (or both) would OOM. The cloud-GPU answer is the
right mental model: the second job is ACCEPTED, holds in QUEUED, and starts
the moment the resources it needs actually exist.

DELIBERATELY RESOURCE-BASED, NOT A HARDCODED "one job" RULE
-----------------------------------------------------------
Admission asks the driver how much VRAM is actually free (nvidia-smi, which
sees every process — not just ours) and compares it against what the waiting
job is estimated to need. On a 24 GB card a nano training and a small
annotate batch genuinely coexist, and admission lets them; on a 4 GB card the
same pair doesn't fit, and the second waits. The waiting message carries the
LIVE numbers, so the UI never explains with a guess.

FIFO fairness on top: among waiting jobs, only the oldest may take newly
freed resources. Without it, whichever waiter happened to poll first would
win, and a big job could starve behind a stream of small ones.

CPU mode has no VRAM to account, so admission degrades to the only sane rule
there: one heavy job at a time.

The wait loop runs INSIDE the job's runner thread (BackgroundTasks), not in a
separate scheduler: the job row already exists, its status is already the
queue, and cancel-while-waiting falls out of the same `control` flag the
runners already honour. No new moving parts.
"""

from __future__ import annotations

import logging
import subprocess
import time
from typing import Literal

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import AnnotationJob, EvaluationJob, JobStatus, TrainingJob

logger = logging.getLogger(__name__)

#: How often a waiting job re-checks. Seconds are the right grain: jobs run
#: for minutes, and the cost of a check is one nvidia-smi + two SELECTs.
POLL_SECONDS = 3.0

#: Safety margin over the estimate. The estimates are rough (batch and image
#: size move real usage), and admitting into exactly-enough memory trades a
#: short wait for a probable OOM.
MARGIN_GB = 0.5

JobKind = Literal["training", "annotation"]


def _query_smi(field: str) -> float | None:
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={field}",
                "--format=csv,noheader,nounits",
                "-i",
                "0",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if out.returncode != 0:
            return None
        return float(out.stdout.strip().splitlines()[0]) / 1024.0
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return None


def free_vram_gb() -> float | None:
    """VRAM currently free on GPU 0, or None when there's no GPU to ask.

    nvidia-smi rather than torch: torch only accounts its OWN allocator, and
    the whole point is to see what everything on the machine — including the
    job we'd collide with — is really using.
    """
    return _query_smi("memory.free")


def total_vram_gb() -> float | None:
    """Total VRAM on GPU 0, for the reservation ledger below."""
    return _query_smi("memory.total")


def _running_jobs(db: Session) -> list[str]:
    """Human labels for every active GPU job, across ALL projects.

    Evaluations count even while QUEUED: they have no admission loop of their
    own — the moment their background task fires they take the card — so a
    queued evaluation is as good as running for anyone deciding to start.
    """
    labels: list[str] = []
    for j in db.scalars(
        select(TrainingJob).where(TrainingJob.status == JobStatus.RUNNING)
    ).all():
        labels.append(f"training v{j.version} ({j.trainer_key})")
    for j in db.scalars(
        select(AnnotationJob).where(AnnotationJob.status == JobStatus.RUNNING)
    ).all():
        labels.append(f"auto-annotate #{j.id}")
    for j in db.scalars(
        select(EvaluationJob).where(
            EvaluationJob.status.in_([JobStatus.QUEUED, JobStatus.RUNNING])
        )
    ).all():
        labels.append(f"evaluation #{j.id}")
    return labels


def _older_waiter_exists(db: Session, kind: JobKind, job_id: int, created_at) -> bool:
    """Is another QUEUED job ahead of this one in line? FIFO by created_at,
    with the id as the tiebreak so two jobs created in the same second still
    order deterministically."""
    for model, model_kind in ((TrainingJob, "training"), (AnnotationJob, "annotation")):
        for j in db.scalars(
            select(model).where(model.status == JobStatus.QUEUED)
        ).all():
            if model_kind == kind and j.id == job_id:
                continue
            if (j.created_at, model_kind, j.id) < (created_at, kind, job_id):
                return True
    return False


def check(
    db: Session, kind: JobKind, job_id: int, created_at, need_gb: float
) -> tuple[bool, str | None]:
    """May this queued job start now? Returns (admitted, reason-if-not).

    The reason is user-facing and built from the live numbers — it becomes the
    job's status_detail verbatim.

    A job with NOBODY else running is ALWAYS admitted, whatever the VRAM
    reading says. Admission arbitrates between our own jobs; whether a lone
    job fits the card at all is the OOM guard's question (which answers with
    settings advice), and the OS desktop already holds a few hundred MB that
    would otherwise make a tight-but-fine job wait forever for memory that is
    never coming back.
    """
    if _older_waiter_exists(db, kind, job_id, created_at):
        return False, "Waiting for GPU — another queued job is ahead in line."

    # ANNOTATION JOBS ARE MUTUALLY EXCLUSIVE, whatever the VRAM says: the
    # annotator registry holds exactly ONE resident model, so a second
    # concurrent batch would evict the first's weights mid-run — observed as
    # every predict() of the evicted job failing "Model not loaded". This is
    # an architecture fact, not a memory fact.
    if kind == "annotation":
        other = db.scalar(
            select(AnnotationJob.id).where(
                AnnotationJob.status == JobStatus.RUNNING
            )
        )
        if other is not None:
            return False, (
                f"Waiting for the model slot — auto-annotate #{other} is using "
                "it (one annotation batch at a time)."
            )

    running = _running_jobs(db)
    if not running:
        return True, None

    free = free_vram_gb()
    if free is None:
        # CPU (or no nvidia-smi): no memory ledger to consult, and two heavy
        # jobs on one CPU starve each other — one at a time is the only
        # honest policy.
        return False, f"Waiting for compute — {running[0]} is using it."

    # THE RESERVATION LEDGER. A just-admitted job takes seconds to actually
    # allocate its memory, and during that window the measured free VRAM still
    # shows the old world — which is how two jobs got admitted into space that
    # only existed once (observed live). So running jobs RESERVE their declared
    # need: effective free is the lower of what the driver measures and what
    # the ledger says is unclaimed. min(), not either alone — measurement
    # catches models that outgrow their declaration, the ledger catches models
    # that haven't claimed theirs yet.
    total = total_vram_gb()
    effective = free
    if total is not None:
        effective = min(free, total - _reserved_gb(db))

    if effective >= need_gb + MARGIN_GB:
        return True, None

    return False, (
        f"Waiting for GPU: {effective:.1f} GB free, needs ~{need_gb:.1f} GB. "
        f"{running[0]} is using the GPU."
    )


def _reserved_gb(db: Session) -> float:
    """Sum of the declared VRAM needs of every active GPU job."""
    reserved = 0.0
    for j in db.scalars(
        select(TrainingJob).where(TrainingJob.status == JobStatus.RUNNING)
    ).all():
        try:
            from app.ml.trainers import registry as trainer_registry

            reserved += trainer_registry.get_class(j.trainer_key).approx_vram_gb
        except KeyError:
            reserved += 3.0  # unknown trainer: assume something modest
    for j in db.scalars(
        select(AnnotationJob).where(AnnotationJob.status == JobStatus.RUNNING)
    ).all():
        try:
            from app.ml import registry as annotator_registry

            reserved += annotator_registry.get_class(j.model_key).approx_vram_gb
        except KeyError:
            reserved += 3.0
    for _ in db.scalars(
        select(EvaluationJob.id).where(
            EvaluationJob.status.in_([JobStatus.QUEUED, JobStatus.RUNNING])
        )
    ).all():
        reserved += 2.5  # a predictor over a trained checkpoint
    return reserved


def wait_for_gpu(db: Session, job, kind: JobKind, need_gb: float) -> bool:
    """Block until the job may start (True) or was cancelled while waiting
    (False). Writes the live reason into job.status_detail while it waits and
    clears it on admission.

    `job` is a TrainingJob or AnnotationJob row from `db`; both carry the
    `control` and `status_detail` columns this loop needs.
    """
    while True:
        # Re-read the row: cancel arrives from a different session, and
        # expire_on_commit alone doesn't help a loop that isn't committing.
        db.expire(job)
        if job.control == "cancel":
            return False

        admitted, reason = check(db, kind, job.id, job.created_at, need_gb)
        if admitted:
            if job.status_detail is not None:
                job.status_detail = None
                db.commit()
            return True

        if job.status_detail != reason:
            job.status_detail = reason
            db.commit()
            logger.info("%s job %s holding: %s", kind, job.id, reason)

        time.sleep(POLL_SECONDS)
