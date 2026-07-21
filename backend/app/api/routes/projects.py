"""Project CRUD endpoints."""

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import func, select, union_all
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import (
    Annotation,
    AnnotationJob,
    Category,
    DatasetVersion,
    Image,
    Project,
    TrainingJob,
)
from app.schemas import ProjectCreate, ProjectRead, ProjectUpdate
from app.services import storage
from app.services.naming import collides

router = APIRouter(tags=["projects"])


def get_project_or_404(project_id: int, db: Session) -> Project:
    """Fetch a project or raise 404.

    Shared by every route that takes a project_id. Centralising it means the
    error message and status code are identical everywhere, and no endpoint can
    forget the check and then blow up on `None.name` with a 500.
    """
    project = db.get(Project, project_id)
    if project is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project {project_id} not found",
        )
    return project


def _training_job_ids(db: Session, project_id: int) -> list[int]:
    """Run ids for a project — i.e. which storage/runs/<id>/ dirs it owns."""
    return list(
        db.scalars(
            select(TrainingJob.id).where(TrainingJob.project_id == project_id)
        ).all()
    )


def _reject_duplicate_project(
    db: Session, name: str, exclude_id: int | None = None
) -> None:
    """409 if another project already reads the same way.

    Projects had no uniqueness check at all, so two "Street Scenes" could sit in
    the list with nothing to tell them apart — and every screen that names the
    project you're in (the sidebar, a run's provenance) becomes ambiguous.

    Unlike classes there is no DB constraint to lean on, so this is the whole
    guard. See services/naming.py on the race that implies and why it's
    acceptable here.
    """
    query = select(Project.name)
    if exclude_id is not None:
        query = query.where(Project.id != exclude_id)
    if collides(name, list(db.scalars(query).all())):
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"A project named '{name.strip()}' already exists "
            "(names are compared without regard to case).",
        )


def _activity_selects(project_id: int | None = None) -> list:
    """`SELECT project_id, MAX(timestamp)` — one per thing that counts as activity.

    THE SINGLE DEFINITION OF "ACTIVITY", and the reason this is a function
    rather than inline SQL. There used to be TWO implementations: the list
    endpoint built its own subqueries while the single-project read used a
    separate helper, and they disagreed about what counts. The list — which is
    what the Projects page actually renders — was the narrower of the two, so
    the fuller answer was only ever visible on a screen nobody was looking at.
    Same shape as the `default_training_version` split: any question with two
    answers eventually gives the wrong one.

    WHAT COUNTS, AND WHY
      - Images, classes, dataset versions: adding to the dataset.
      - Training AND annotation jobs, including when they STARTED and FINISHED.
        A run that finishes an hour after it was queued is activity at the point
        it finished. Annotation jobs were missing from both old versions.
      - Annotations, by `created_at` AND `updated_at`. Labelling is the activity
        this application mostly exists for, and it was counted by neither. An
        afternoon of drawing boxes moved nothing, so a project you had just
        worked on read "3 days ago" and sorted to the bottom of a list ordered
        by activity. `updated_at` matters separately: a review pass that only
        corrects existing labels creates no rows at all.

    NOT COUNTED: deletions. There is no row left to carry a timestamp, and
    recording one would mean writing to the project on every child delete. The
    common deletions travel alongside other activity anyway.
    """
    selects = []

    def add(column, project_column, source=None, join=None):
        query = select(project_column.label("pid"), func.max(column).label("t"))
        if source is not None:
            query = query.select_from(source)
        if join is not None:
            query = query.join(*join)
        if project_id is not None:
            query = query.where(project_column == project_id)
        selects.append(query.group_by(project_column))

    # Directly project-scoped tables.
    for model in (Image, Category, DatasetVersion, TrainingJob, AnnotationJob):
        for name in ("created_at", "started_at", "finished_at"):
            column = getattr(model, name, None)
            if column is not None:
                add(column, model.project_id)

    # Annotations hang off images, not projects, so they need the join.
    for name in ("created_at", "updated_at"):
        add(
            getattr(Annotation, name),
            Image.project_id,
            source=Annotation,
            join=(Image, Image.id == Annotation.image_id),
        )

    return selects


def _last_activity_map(db: Session, project_id: int | None = None) -> dict[int, datetime]:
    """{project_id: when anything in it last changed}, in ONE query.

    Batched deliberately. Asking per project would be the N+1 problem the list
    endpoint's counts already avoid — with this many activity sources it would
    be dozens of round-trips per project.
    """
    combined = union_all(*_activity_selects(project_id)).subquery()
    rows = db.execute(
        select(combined.c.pid, func.max(combined.c.t)).group_by(combined.c.pid)
    ).all()
    return {pid: t for pid, t in rows if pid is not None and t is not None}


def _last_activity(db: Session, project: Project) -> datetime:
    """When anything in this project last changed.

    `Project.updated_at` only moves when the projects ROW is updated, so
    uploading images or training a model doesn't touch it. Sorting a list by it
    would order projects by when someone last renamed one — so the real answer
    is the latest timestamp across everything in `_activity_selects`, floored by
    the project's own stamps so a brand-new empty project still answers.
    """
    stamps = [project.updated_at, project.created_at]
    latest = _last_activity_map(db, project.id).get(project.id)
    if latest is not None:
        stamps.append(latest)
    return max(s for s in stamps if s is not None)


def _with_counts(db: Session, project: Project) -> ProjectRead:
    """Attach image/class counts to a project for the response."""
    image_count = db.scalar(
        select(func.count()).select_from(Image).where(Image.project_id == project.id)
    )
    class_count = db.scalar(
        select(func.count()).select_from(Category).where(Category.project_id == project.id)
    )
    data = ProjectRead.model_validate(project)
    data.image_count = image_count or 0
    data.class_count = class_count or 0
    data.last_activity_at = _last_activity(db, project)
    return data


@router.get("", response_model=list[ProjectRead])
def list_projects(db: Session = Depends(get_db)) -> list[ProjectRead]:
    """List all projects, newest first, with counts and last-activity time.

    Returns EVERY project in a stable order and leaves sorting and filtering to
    the client. That's the right split here: a local tool has tens of projects,
    not thousands, so re-sorting is instant in the browser and a search box that
    round-trips per keystroke would feel worse for no benefit. The server's job
    is to make the order deterministic and to supply `last_activity_at`, which
    only SQL can answer.

    The counts come from two GROUP BY subqueries joined onto the main select,
    rather than looping over projects and counting each one. That loop is the
    N+1 query problem: 50 projects would mean 101 round-trips. Here it's always
    exactly one query, whatever the row count.

    outerjoin (not join) is essential — an inner join would silently drop any
    project with zero images, so a newly created project would vanish from the
    list until you uploaded something.
    """
    image_counts = (
        select(Image.project_id, func.count(Image.id).label("n"))
        .group_by(Image.project_id)
        .subquery()
    )
    class_counts = (
        select(Category.project_id, func.count(Category.id).label("n"))
        .group_by(Category.project_id)
        .subquery()
    )

    rows = db.execute(
        select(
            Project,
            func.coalesce(image_counts.c.n, 0),
            func.coalesce(class_counts.c.n, 0),
        )
        .outerjoin(image_counts, image_counts.c.project_id == Project.id)
        .outerjoin(class_counts, class_counts.c.project_id == Project.id)
        # Deterministic by construction. created_at alone would tie for projects
        # made in the same second (SQLite stamps to the second), and a tie means
        # SQLite may return them in any order it likes — which is how a list
        # appears to reshuffle on its own between visits. The id tiebreaker
        # makes the order total, so it cannot.
        .order_by(Project.created_at.desc(), Project.id.desc())
    ).all()

    # Last activity comes from the SHARED definition, in one batched query —
    # this endpoint used to compute it from its own narrower set of subqueries,
    # so the list and the detail view could disagree about the same project.
    activity = _last_activity_map(db)

    results = []
    for project, n_images, n_classes in rows:
        data = ProjectRead.model_validate(project)
        data.image_count = n_images
        data.class_count = n_classes
        stamps = [
            s
            for s in (project.updated_at, project.created_at, activity.get(project.id))
            if s is not None
        ]
        data.last_activity_at = max(stamps) if stamps else None
        results.append(data)
    return results


@router.post("", response_model=ProjectRead, status_code=status.HTTP_201_CREATED)
def create_project(payload: ProjectCreate, db: Session = Depends(get_db)) -> ProjectRead:
    """Create a project.

    201 Created rather than 200 — it tells any client (including the auto-
    generated docs) that a new resource now exists, which is the whole point of
    using HTTP semantics instead of returning 200 for everything.
    """
    _reject_duplicate_project(db, payload.name)

    project = Project(
        name=payload.name,
        description=payload.description,
        task_type=payload.task_type.value,
    )
    db.add(project)
    db.commit()
    db.refresh(project)  # reload server-generated columns (id, created_at)
    return _with_counts(db, project)


@router.get("/{project_id}", response_model=ProjectRead)
def get_project(project_id: int, db: Session = Depends(get_db)) -> ProjectRead:
    project = get_project_or_404(project_id, db)
    return _with_counts(db, project)


@router.patch("/{project_id}", response_model=ProjectRead)
def update_project(
    project_id: int, payload: ProjectUpdate, db: Session = Depends(get_db)
) -> ProjectRead:
    """Partially update a project.

    `exclude_unset=True` is what makes this a real PATCH. It yields only the
    fields the client actually sent, so:
        {"name": "x"}                 -> leaves description alone
        {"description": null}         -> explicitly clears description
    Without it, every omitted field would arrive as None and PATCH would
    silently wipe data the caller never mentioned.
    """
    project = get_project_or_404(project_id, db)

    fields = payload.model_dump(exclude_unset=True)
    if fields.get("name") is not None:
        _reject_duplicate_project(db, fields["name"], exclude_id=project_id)

    for field, value in fields.items():
        setattr(project, field, value)

    db.commit()
    db.refresh(project)
    return _with_counts(db, project)


@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_project(project_id: int, db: Session = Depends(get_db)) -> None:
    """Delete a project, its DB rows, and its images on disk.

    Order matters. The DB commit happens FIRST, then the files are removed:

      - commit first, then unlink: a crash in between leaves orphaned files —
        wasted disk, but the app is consistent and nothing is broken.
      - unlink first, then commit: a crash in between leaves DB rows pointing at
        files that no longer exist — every image in the grid renders broken.

    Neither order is atomic (the filesystem and SQLite don't share a
    transaction). So we choose the failure mode that degrades gracefully.
    """
    project = get_project_or_404(project_id, db)
    # Collected BEFORE the delete: the cascade removes the training_jobs rows,
    # and with them the only record of which run directories belonged here.
    job_ids = _training_job_ids(db, project_id)

    db.delete(project)  # cascades to images + categories, via ORM and FK pragma
    db.commit()

    storage.delete_project_files(project_id, job_ids)


class BulkDelete(BaseModel):
    project_ids: list[int]


@router.post("/bulk-delete")
def bulk_delete(payload: BulkDelete, db: Session = Depends(get_db)) -> dict:
    """Delete several projects at once.

    POST, not DELETE: a request body on DELETE is legal but poorly supported —
    some proxies strip it, and fetch() in older browsers ignores it. The
    alternative, a comma-joined query string, breaks at a few hundred ids.

    Same commit-then-unlink ordering as the single delete, for the same reason:
    a crash mid-way leaves orphaned files (wasted disk) rather than DB rows
    pointing at files that no longer exist (a grid full of broken images).

    Skips ids that don't exist rather than 404-ing the whole batch. Deleting
    something already gone is not a failure — the caller wanted it gone, and it
    is. Failing the other nine because one id was stale would be hostile.
    """
    projects = list(
        db.scalars(select(Project).where(Project.id.in_(payload.project_ids))).all()
    )
    deleted_ids = [p.id for p in projects]
    # Same reason as the single delete: gather the run directories while the
    # rows naming them still exist.
    jobs_by_project = {pid: _training_job_ids(db, pid) for pid in deleted_ids}

    for project in projects:
        db.delete(project)
    db.commit()

    for project_id in deleted_ids:
        storage.delete_project_files(project_id, jobs_by_project[project_id])

    return {
        "deleted": len(deleted_ids),
        "not_found": sorted(set(payload.project_ids) - set(deleted_ids)),
    }
