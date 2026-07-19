import pytest

from satira import labels


def test_canonical_names_are_the_four_class_taxonomy() -> None:
    assert labels.CANONICAL_NAMES == (
        "authentic",
        "satire",
        "misleading_context",
        "fabricated",
    )
    assert [labels.Label.AUTHENTIC, labels.Label.SATIRE] == [0, 1]


def test_str_to_int_round_trips_canonical_names() -> None:
    for index, name in enumerate(labels.CANONICAL_NAMES):
        assert labels.str_to_int(name) == index
        assert labels.int_to_str(index) == name


def test_str_to_int_is_case_and_whitespace_insensitive() -> None:
    assert labels.str_to_int("  Satire ") == 1
    assert labels.str_to_int("AUTHENTIC") == 0


def test_parody_is_an_alias_for_satire() -> None:
    assert labels.str_to_int("parody") == labels.Label.SATIRE
    # ...but only via the alias path; parody is not a canonical class.
    assert "parody" not in labels.CANONICAL_NAMES
    with pytest.raises(ValueError):
        labels.str_to_int("parody", allow_aliases=False)


def test_str_to_int_raises_on_unknown_label() -> None:
    with pytest.raises(ValueError):
        labels.str_to_int("totally_bogus")


def test_str_to_int_rejects_non_string() -> None:
    with pytest.raises(TypeError):
        labels.str_to_int(1)  # type: ignore[arg-type]


def test_int_to_str_raises_out_of_range() -> None:
    with pytest.raises(ValueError):
        labels.int_to_str(99)


def test_class_names_subset() -> None:
    assert labels.class_names(2) == ["authentic", "satire"]
    assert labels.class_names() == list(labels.CANONICAL_NAMES)
    with pytest.raises(ValueError):
        labels.class_names(0)
    with pytest.raises(ValueError):
        labels.class_names(len(labels.CANONICAL_NAMES) + 1)


def test_binary_map_is_the_first_two_classes() -> None:
    assert labels.BINARY_STR_TO_INT == {"authentic": 0, "satire": 1}
    # .get() returns None for out-of-subset labels so callers can drop them.
    assert labels.BINARY_STR_TO_INT.get("fabricated") is None
