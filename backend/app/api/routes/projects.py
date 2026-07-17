"""Project CRUD endpoints."""

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Category, Image, Project
from app.schemas import ProjectCreate, ProjectRead, ProjectUpdate
from app.services import storage

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
    return data


@router.get("", response_model=list[ProjectRead])
def list_projects(db: Session = Depends(get_db)) -> list[ProjectRead]:
    """List all projects, newest first, each with its image and class counts.

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
        .order_by(Project.created_at.desc(), Project.id.desc())
    ).all()

    results = []
    for project, n_images, n_classes in rows:
        data = ProjectRead.model_validate(project)
        data.image_count = n_images
        data.class_count = n_classes
        results.append(data)
    return results


@router.post("", response_model=ProjectRead, status_code=status.HTTP_201_CREATED)
def create_project(payload: ProjectCreate, db: Session = Depends(get_db)) -> ProjectRead:
    """Create a project.

    201 Created rather than 200 — it tells any client (including the auto-
    generated docs) that a new resource now exists, which is the whole point of
    using HTTP semantics instead of returning 200 for everything.
    """
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

    for field, value in payload.model_dump(exclude_unset=True).items():
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
    db.delete(project)  # cascades to images + categories, via ORM and FK pragma
    db.commit()

    storage.delete_project_dir(project_id)


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

    for project in projects:
        db.delete(project)
    db.commit()

    for project_id in deleted_ids:
        storage.delete_project_dir(project_id)

    return {
        "deleted": len(deleted_ids),
        "not_found": sorted(set(payload.project_ids) - set(deleted_ids)),
    }
