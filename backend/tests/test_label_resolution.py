"""
Mapping Grounding DINO's returned phrase back to one of OUR class names.

Pure logic — no model, no GPU. It earns its own file because getting it wrong
does not raise: it writes a confidently mislabelled box into the dataset, which
survives review (the box looks right, and only the class is wrong) and ends up
as training data.

THE BUG THIS SUITE EXISTS FOR: the resolver used to fall back to
`class_names[0]` whenever it couldn't decide. Grounding DINO returns an EMPTY
label for its lower-confidence detections, so on a "car, person" project every
undecided box silently became a car. Confirmed against the real model on the
shapes demo: every person-shaped box below ~0.3 confidence was stored as "car".
"""

from __future__ import annotations

from app.ml.annotators.grounding_dino import GroundingDinoAnnotator

resolve = GroundingDinoAnnotator._resolve_label

# What _build_prompt produces for classes ("car", "person") with no custom
# prompts: the phrase the model is given -> the class it means.
BACK_MAP = {"car": "car", "person": "person"}


def test_exact_phrase_resolves():
    assert resolve("car", BACK_MAP) == "car"
    assert resolve("person", BACK_MAP) == "person"


def test_trailing_punctuation_and_case_are_ignored():
    """The model echoes prompt formatting back at us."""
    assert resolve("Car .", BACK_MAP) == "car"
    assert resolve(" PERSON. ", BACK_MAP) == "person"


def test_empty_label_is_dropped_not_guessed():
    """THE regression. An empty label used to become class_names[0]."""
    assert resolve("", BACK_MAP) is None
    assert resolve("   ", BACK_MAP) is None
    assert resolve(" . ", BACK_MAP) is None


def test_merged_phrase_naming_two_classes_is_dropped():
    """Grounding DINO merges adjacent prompt phrases: "car . person ." comes
    back as the single span "car person".

    That span contains both class names, so the old containment pass returned
    whichever it iterated first — making the label depend on class ORDER rather
    than on the image. A span naming two classes identifies neither.
    """
    assert resolve("car person", BACK_MAP) is None
    # ...and the reversed class order must give the same answer, which is the
    # whole point: order must not decide.
    assert resolve("car person", {"person": "person", "car": "car"}) is None


def test_sub_span_of_a_custom_prompt_resolves():
    """A custom prompt ("a parked car" for the class "car") is echoed as a
    sub-span, and must still map home."""
    back = {"a parked car": "car", "person": "person"}
    assert resolve("car", back) == "car"
    assert resolve("parked", back) == "car"


def test_unrelated_phrase_is_dropped():
    assert resolve("bicycle", BACK_MAP) is None


def test_shared_word_resolves_only_when_unambiguous():
    """The loosest pass still has to identify exactly one class."""
    back = {"delivery truck": "truck", "fire truck": "engine"}
    # "truck" is a word in both prompts — genuinely ambiguous.
    assert resolve("truck", back) is None
    # "delivery" belongs to one.
    assert resolve("delivery", back) == "truck"
