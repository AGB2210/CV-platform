# Local CV Platform

A self-hosted computer vision platform that runs entirely on local hardware. It
mirrors the Roboflow workflow — **auto-annotate → review → train → evaluate →
deploy** — with no cloud service in the loop: model weights are downloaded
once, then annotation, training and inference all happen on your machine. Your
images, your GPU, your weights.

**Scope: object detection only** — models that draw a box around each object.
Segmentation (pixel-exact outlines) is a later phase; the architecture is built
to accommodate it without a rewrite.

> **A fresh clone starts empty.** `storage/`, the SQLite database and model
> weights are gitignored — this repository is code, not data. Run `start.bat`,
> create a project, and upload something.

---

## Status

| Phase | Scope                                             | State       |
| ----- | ------------------------------------------------- | ----------- |
| 0     | Scaffolding: FastAPI + SQLite + React/Vite shell   | ✅ Complete |
| 1     | Project & dataset management, class definitions    | ✅ Complete |
| 2     | Auto-annotation — Grounding DINO tiny/base, YOLO-World, OWLv2, Florence-2 | ✅ Complete |
| 3     | Annotation canvas (draw / edit / delete / accept-reject) | ✅ Complete |
| 4     | Training — YOLO12, YOLO26, RT-DETR, RF-DETR families + GPU job queue | ✅ Complete |
| 5     | Evaluation (test-set mAP) + inference playground + local API | ✅ Complete |

One idea underpins the whole annotation workflow: **proposals vs annotations**.
A model *proposes* boxes; nothing becomes a real annotation until a person
accepts it. Proposals live right next to accepted boxes, visibly marked — there
is no separate review queue to empty. Train/val/test splits are set on the
Dataset page.

---

## What it does

**Get data in.** Upload loose images, a `.zip`, or point it at a **folder**.
**COCO and YOLO** — the two standard annotation formats — are detected
automatically; there is no format dropdown to get wrong. `train/`,
`valid/` and `test/` subfolders keep their splits; anything else lands in
train. Segmentation polygons are read as their bounding boxes, so an instance-
segmentation export imports as a usable detection dataset.

**Nothing is stored twice.** Images are identified by the SHA-256 of their
bytes, so re-uploading a folder adds nothing and says so. Filenames are never
used for identity — `train/001.jpg` and `test/001.jpg` are different pictures
and both are kept.

**Label it.** Ten **open-vocabulary** annotators in four families: you type
what you want found ("helmet", "license plate") and they find it — zero-shot,
no training required. Picked as family + size: Grounding DINO (tiny/base),
YOLO-World (S–X) for fast bulk batches, OWLv2 (base/large) for rare classes,
Florence-2 (base/large) for descriptive prompts.
Accept or reject their proposals on an SVG canvas where one unit is one image
pixel, so boxes line up exactly with what you see. Re-uploading a corrected
annotation file also arrives as proposals rather than overwriting work.

**Save points.** "Save dataset" writes a metadata-only snapshot — which images,
their splits, their boxes, the classes. Restoring rewinds to it. Snapshots cost
kilobytes because image bytes are never copied.

**Train.** Four detector families — YOLO12 and YOLO26 (nano through xlarge),
RT-DETR (L/X) and RF-DETR (nano through large) — picked as family + size, with
each size's GPU-memory (VRAM) appetite shown up front. The default batch size
adapts to the detected GPU, and a live **mAP** curve (mean average precision,
the standard accuracy score for detectors) plus live logs show what the run is
doing. **Stop** ends a run cleanly at the next epoch boundary and keeps the
best checkpoint; **Cancel** aborts within seconds. Fine-tune from a previous
run's checkpoint. Runs train a *saved version* of the dataset, never the live
working copy, so "trained on v3" stays true no matter what you edit afterwards.
A validation split is required — without held-out data (images kept aside from
training) the reported mAP only measures what the model memorised. A run that
exceeds GPU memory stops with advice on what to lower, not a CUDA
out-of-memory stack trace.

**One GPU, many jobs.** Start a run while another project is still training
and it queues, cloud-style — holding with a live "waiting for GPU: X GB free,
needs ~Y GB" message computed from the actual card, and starting automatically
the moment the resources free up. First come, first served; cancel works while
waiting.

**Know your data.** A dataset **Health** page draws what raw counts hide —
class balance, box-size distributions — and turns the pathological cases into
named warnings: an unused class, a 20× imbalance, boxes too tiny to learn
from. Low mAP is usually the data's fault, and this is the page where it
shows.

**Evaluate.** Score any finished run on the held-out **test split** with COCO
metrics — overall mAP, mAP@50, per-class AP — plus **per-class PR curves**, a
**confusion matrix** (including missed and invented objects), and the
**worst-scoring test images** as thumbnails that open straight in the editor.
The number you quote comes from images the model never saw in training, and
the page shows *why* it is what it is.

**Deploy.** An inference playground (upload an image, see the boxes), plus every
trained model doubles as a **local REST endpoint** with ready-to-paste curl and
Python snippets — and the model itself downloads as a portable `.pt` or as
**ONNX**, runnable without PyTorch on edge devices and other stacks.

**One click, no setup.** Double-click `start.bat` and the app builds and serves
itself. The heavy ML libraries (torch and friends, several GB) are not installed
until a feature actually needs them — the first time you open Auto-annotate or
Train, the app installs them itself, picking the right PyTorch build for your
GPU, with progress shown on the page. No command line, ever.

---

## Architecture

One process serves everything: the compiled interface and the API, from the
same origin.

```
                  ┌───────────────────────────────────────┐
 browser  ──────► │  Backend (FastAPI, port 8000)         │
 http://          │                                       │
 localhost:8000   │  /         the built React UI         │
                  │  /api/*    the JSON API               │
                  │                                       │
                  │  ┌─────────────────────────────┐      │
                  │  │ api/       HTTP routes      │      │
                  │  │ services/  the actual logic │      │
                  │  │ ml/        model adapters   │      │
                  │  │ models/    database tables  │      │
                  │  └─────────────────────────────┘      │
                  └──────────┬───────────────┬────────────┘
                             ▼               ▼
                      ┌────────────┐  ┌────────────┐
                      │  SQLite    │  │  storage/  │
                      │  metadata  │  │  bytes     │
                      └────────────┘  └────────────┘
```

**One origin, no CORS, no dev/prod switch.** The frontend calls relative paths
(`/api/health`), and FastAPI serves both the built UI at `/` and the API under
`/api` (see `app/main.py`). Because the page and the API come from the same
origin, the browser never raises a cross-origin (CORS) complaint, and the same
code runs whether you built it yourself or unpacked a release — no hardcoded
hostnames.

**The interface is built, then served as static files** — it does not run its
own server. `start.bat` builds it (when working from source and it is stale) and
hands the compiled output to FastAPI. A React dev server with hot-reload is still
available for UI work if you want it (see *Running it*), but it is optional, not
part of the normal path.

**Where data lives.** Metadata (projects, images, classes, jobs) goes in SQLite.
Bytes (images, weights, checkpoints) go on the filesystem under `storage/`, with
only their path recorded in the DB. Databases are bad at storing large binary
files; filesystems are good at it, and a file on disk is something the browser
can fetch directly via `<img src>`.

---

## Layout

```
cv app/
├── start.bat                Double-click to run everything
├── scripts/app.py           The launcher: install, build, serve, health-check
│
├── backend/
│   ├── app/
│   │   ├── main.py          FastAPI entrypoint: CORS, routers, lifespan
│   │   ├── config.py        Settings (paths, DB URL) from env/.env
│   │   ├── database.py      SQLAlchemy engine, session, Base, get_db
│   │   ├── api/             HTTP layer — thin route handlers only
│   │   ├── models/          SQLAlchemy ORM models (how data is stored)
│   │   ├── schemas/         Pydantic schemas (what the API accepts/returns)
│   │   ├── services/        Business logic — the actual work
│   │   └── ml/              Model integrations (annotators, trainers)
│   ├── requirements.txt     Core deps — installs in seconds
│   └── requirements-ml.txt  torch/transformers/etc — Phase 2+, multi-GB
│
├── frontend/
│   └── src/
│       ├── components/      Reusable UI (layout shell, status badge)
│       ├── pages/           One component per route
│       ├── lib/api.ts       Typed API client — the only place fetch() is called
│       └── index.css        Design tokens (@theme) + shared primitives
│
└── storage/                 gitignored — images, weights, runs
```

Two boundaries are worth internalising, because they're what keep this
maintainable as the ML code grows:

1. **`models/` vs `schemas/`** — storage shape vs API contract. They change for
   different reasons. Collapsing them makes every DB column public API by
   accident.
2. **`api/` vs `services/` vs `ml/`** — `ml/` never imports FastAPI and doesn't
   know what a request is. That's what lets a training loop be called from an
   endpoint, a background job, or a test script without modification.

---

## Running it

**Double-click `start.bat`.** That is the whole story — for a downloaded release
and for a source checkout alike.

Requires **Python 3.10 – 3.13** — 3.14 is newer than parts of the ML stack,
and if it's your machine's default, the launcher finds an installed 3.10–3.13
and switches to it by itself. A source checkout also needs **Node 18+** the first
time, to build the interface; a downloaded release ships the interface already
built and needs no Node at all.

On first run it creates the virtual environment and installs the core
dependencies (about a minute), builds the interface if it needs building, then
serves everything and opens the browser. After that it starts in seconds.

```
  CV Platform

✓ Core dependencies up to date
✓ Frontend up to date
✓ Ready  http://127.0.0.1:8000
```

Everything runs as **one process on one port** — the compiled UI at `/` and the
API under `/api`. **Ctrl+C stops it.** `http://localhost:8000/docs` has the
interactive API reference.

The heavy ML dependencies (torch, transformers, ultralytics — several GB) are
**not** installed up front. The first time you open **Auto-annotate** or
**Train**, the app installs them itself, with progress shown on the page, and
then continues. Nothing about that touches a command line. Everything else — 
uploading, annotating, splits, versions, import/export — works without them.

Options:

```bash
start.bat --no-browser    # don't open a browser
start.bat --setup-only    # install and build, then exit
start.bat --port 9000     # serve on a different port
```

> `start.bat` is a thin shim; the logic is in
> [`scripts/app.py`](scripts/app.py), because batch is poor at environment
> setup, health polling and clean shutdown, and Python is already a
> prerequisite. In `cmd.exe` type `start.bat` in full — bare `start` is a cmd
> builtin.

> **It binds to `127.0.0.1` deliberately.** The app has no authentication of any
> kind. `--host 0.0.0.0` publishes an unauthenticated upload-and-delete API to
> your whole network; the launcher warns before doing it.

### Live UI development (optional)

`start.bat` builds the interface and serves it static, which is the real user
experience but means a UI edit needs a rebuild to show. When iterating on the
frontend, the Vite dev server with hot module reloading is faster. Run the two
sides in separate terminals:

```bash
# Terminal 1 — backend
cd backend
venv\Scripts\activate          # Windows  (source venv/bin/activate on macOS/Linux)
uvicorn app.main:app --reload --port 8000

# Terminal 2 — frontend dev server (proxies /api to :8000)
cd frontend
npm run dev                     # http://localhost:5173
```

This is a convenience for UI work, not the supported way to run the app. The dev
server runs at <http://localhost:5173> and proxies `/api` to the backend on
<http://localhost:8000> (interactive docs at `/docs`).

The ML dependencies in `requirements-ml.txt` are installed automatically the
first time a feature needs them — you should not have to install torch by hand.
If you ever do (a headless setup, say), the comments at the top of that file
explain how to pick the right PyTorch build for your GPU.

### Troubleshooting

| Symptom                          | Cause / fix                                                                                      |
| -------------------------------- | ------------------------------------------------------------------------------------------------ |
| `Port 8000 is already in use`    | **Just run `start.bat` again** — it clears a stale server of its own first. Or pass `--port` to use another. |
| A change didn't take effect      | Re-run `start.bat`. From source it rebuilds the interface if you edited it; for backend changes it restarts the one process. |
| Auto-annotate / Train says it's installing | Expected on first use — it's fetching the multi-GB ML stack once. Let it finish; later runs skip it. |
| `npm not found on PATH` (source) | Install Node 18+, then open a **new** terminal (PATH changes don't reach existing ones). A downloaded release needs no Node. |
| Upload fails with "Failed to fetch" | A connection-level failure, so there's no server message. Usually the backend stopped — check the `start.bat` window. |
| "Window too narrow" overlay      | The layout needs ≥1024px. It's an overlay over the still-running app, so widening the window restores your place. |
| Want a clean slate               | Delete `backend/venv`, `frontend/node_modules`, `frontend/dist`, `backend/cvplatform.db` and `storage/`, then run `start.bat`. |

### Upgrading

Unpack a newer version **over your existing folder** — or pull — and run
`start.bat`. Nothing else to do: the app migrates its own database at startup
(Alembic, `backend/migrations/`), and your projects, images and annotations
carry over. If a future major version ever needs more than that, its release
notes will say so explicitly.

Your data is part of the folder, not the download: each install keeps its
images in `storage/` and its database at `backend/cvplatform.db`, both outside
the zip. A zip unpacked somewhere **new** is therefore a separate, empty
install — to move data between installs, copy those two items across while
neither copy is running.

---

## Design conventions

The UI targets "professional internal tool" — dense, neutral, restrained.
Concretely:

- **One accent colour** (`accent-*`, a muted slate-blue), used only for primary
  actions, active nav, and focus rings. Everything else is Tailwind's neutral
  gray scale.
- **Status by meaning, not colour** — `status-idle/busy/good/bad` tokens, so job
  states look identical everywhere.
- Real icons (lucide), never emoji. No gradients, no glassmorphism, no glow.
- Left sidebar navigation, data tables over cards, left-aligned empty states.

Tokens live in `frontend/src/index.css` under `@theme`. Pick from them rather
than adding new colours.

---

## Tests

```bash
cd backend  && venv/Scripts/python.exe -m pytest    # ~240 tests, no GPU needed
cd frontend && npm run typecheck && npm run lint
```

The suite runs without torch, ultralytics or transformers: the tests insert
annotation proposals directly and register fake trainers into the real
registry, so every workflow is exercised without a single model download. That
keeps CI to a couple of minutes and keeps honest the claim that
the core app works with no GPU. What it deliberately does *not* cover is real
inference and real training — those are verified by driving the running app.

CI (`.github/workflows/ci.yml`) runs exactly these three commands on push and
pull request. Nothing more: CI that checks something different from what people
run locally is how "works on my machine" starts.

## Versioning

Standard semantic versioning — `MAJOR.MINOR.PATCH`. Each number answers one
question for someone upgrading: what does this cost me?

| number | means |
| ------ | ----- |
| **MAJOR** | Breaking. Something that worked before now works differently. That version's release notes state what changed and what — if anything — you need to do. |
| **MINOR** | New capability, with nothing that worked before stopping working. Safe to take. |
| **PATCH** | Fixes and internal work. Nothing new, nothing to migrate. |

The number is authored once, in `VERSION` at the repo root; the backend reads
it and reports it at `/api/health`, and CI fails if any copy — or a release
tag — disagrees with it. The version on a download always matches the code
inside it.

## Releases

Each [GitHub Release](https://github.com/AGB2210/CV-platform/releases) is a
zip you download, unpack anywhere, and run with `start.bat` — **delivery, not
deployment**, deliberately: this is a local, GPU-bound, single-user tool with
no authentication, so there is no server it belongs on, and a public instance
would be an open upload endpoint.

Every release is smoke-tested before it is published: the workflow unpacks
its own zip into a clean directory, installs from it, starts the server and
asks it real questions — does the app shell load, does a deep link resolve,
does a mistyped API path return a clean 404. If any of that fails, nothing is
published. A release nobody has run is worse than no release: it looks
finished and breaks on someone else's machine.

---

## License

None. This is a personal portfolio project, published to be read rather than
reused — without a license, the default applies and nobody else has permission
to copy, modify or distribute it.
