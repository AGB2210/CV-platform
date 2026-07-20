"""Class (category) endpoints, nested under a project."""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.database import get_db
from app.enums import CLASS_COLORS
from app.models import Annotation, Category
from app.schemas import CategoryCreate, CategoryRead, CategoryUpdate
from app.services.naming import collides
from app.api.routes.projects import get_project_or_404

router = APIRouter(tags=["classes"])


def _reject_duplicate_class(
    db: Session, project_id: int, name: str, exclude_id: int | None = None
) -> None:
    """409 if this project already has a class reading the same way."""
    query = select(Category.name).where(Category.project_id == project_id)
    if exclude_id is not None:
        query = query.where(Category.id != exclude_id)
    if collides(name, list(db.scalars(query).all())):
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"A class named '{name.strip()}' already exists in this project "
            "(class names are compared without regard to case).",
        )


def _next_color(db: Session, project_id: int) -> str:
    """Pick the next palette colour for a new class.

    Cycles by class count, so the first few classes in a project get visually
    distinct colours. Modulo means it wraps rather than crashing on project 13 —
    duplicate colours are a cosmetic issue, an IndexError is a broken endpoint.
    """
    count = (
        db.scalar(
            select(func.count()).select_from(Category).where(Category.project_id == project_id)
        )
        or 0
    )
    return CLASS_COLORS[count % len(CLASS_COLORS)]


@router.get("/projects/{project_id}/classes", response_model=list[CategoryRead])
def list_classes(project_id: int, db: Session = Depends(get_db)) -> list[CategoryRead]:
    """A project's classes, each with how many boxes use it.

    The count comes from one GROUP BY joined on, not a COUNT per class — the
    N+1 problem. It's here rather than in a separate endpoint because the UI
    needs it at exactly the moment it renders the list: the delete control sits
    on each row, and it has to be able to say what deleting destroys.
    """
    get_project_or_404(project_id, db)

    counts = dict(
        db.execute(
            select(Annotation.category_id, func.count(Annotation.id))
            .join(Category, Category.id == Annotation.category_id)
            .where(Category.project_id == project_id)
            .group_by(Annotation.category_id)
        ).all()
    )

    rows = db.scalars(
        select(Category).where(Category.project_id == project_id).order_by(Category.id)
    ).all()

    results = []
    for row in rows:
        data = CategoryRead.model_validate(row)
        data.annotation_count = counts.get(row.id, 0)
        results.append(data)
    return results


@router.post(
    "/projects/{project_id}/classes",
    response_model=CategoryRead,
    status_code=status.HTTP_201_CREATED,
)
def create_class(
    project_id: int, payload: CategoryCreate, db: Session = Depends(get_db)
) -> Category:
    """Add a class to a project."""
    get_project_or_404(project_id, db)

    # Case-insensitive check BEFORE the insert. The unique constraint below
    # catches exact duplicates, but SQLite compares strings case-sensitively, so
    # "car" and "Car" both satisfy it — and then export as two classes, teaching
    # the model to split one concept in half. See services/naming.py for why
    # this half lives in the application rather than the schema.
    _reject_duplicate_class(db, project_id, payload.name)

    category = Category(
        project_id=project_id,
        name=payload.name,
        color=payload.color or _next_color(db, project_id),
    )
    db.add(category)

    try:
        db.commit()
    except IntegrityError:
        # Raised by the uq_category_project_name constraint. We let the DATABASE
        # detect the duplicate rather than doing a SELECT-then-INSERT check,
        # because that check has a race: two simultaneous requests can both see
        # "no duplicate" before either inserts. The unique constraint is the only
        # thing that can actually guarantee it.
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"A class named '{payload.name}' already exists in this project",
        ) from None

    db.refresh(category)
    return category


@router.patch("/classes/{class_id}", response_model=CategoryRead)
def update_class(
    class_id: int, payload: CategoryUpdate, db: Session = Depends(get_db)
) -> Category:
    category = db.get(Category, class_id)
    if category is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Class {class_id} not found")

    fields = payload.model_dump(exclude_unset=True)
    if "name" in fields and fields["name"] is not None:
        # Excluding itself: renaming "car" to "Car" is a legitimate correction
        # of its own capitalisation, not a collision with itself.
        _reject_duplicate_class(db, category.project_id, fields["name"], exclude_id=class_id)

    for field, value in fields.items():
        setattr(category, field, value)

    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"A class named '{payload.name}' already exists in this project",
        ) from None

    db.refresh(category)
    return category


@router.delete("/classes/{class_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_class(class_id: int, db: Session = Depends(get_db)) -> None:
    category = db.get(Category, class_id)
    if category is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Class {class_id} not found")
    db.delete(category)
    db.commit()
