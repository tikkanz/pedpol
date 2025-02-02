import polars as pl
import pytest

from pedpol.core import parents
from pedpol.validation import (
    add_missing_records,
    get_animals_are_own_parent,
    get_animals_born_before_parents,
    get_animals_with_multiple_records,
    get_parent_sex_mismatches,
    get_parents_both_sires_and_dams,
    get_parents_without_own_record,
    null_parents_without_own_record,
    recode_pedigree,
    validate_pedigree,
)

pl.Config.set_tbl_rows(15)


@pytest.fixture
def anims_are_own_parent(ped_errors):
    return get_animals_are_own_parent(*ped_errors)


def test_no_anims_are_own_parent(ped_lit):
    assert get_animals_are_own_parent(*ped_lit).height == 0


def test_number_of_animals_that_are_their_own_parent(anims_are_own_parent):
    assert anims_are_own_parent.height == 1


def test_record_of_animal_that_is_their_own_parent(anims_are_own_parent):
    assert anims_are_own_parent.to_dicts() == [{"anim": 5, "sire": 9, "dam": 5}]


@pytest.fixture
def ped_errors_sex(ped_errors):
    ped, lbls = ped_errors
    return ped.with_columns((pl.col(lbls[0]) % 2 + 1).alias("sex")), lbls


def test_find_parent_sex_mismatches(ped_errors_sex):
    ped, lbls = ped_errors_sex
    assert get_parent_sex_mismatches(ped, lbls[:2]).height == 2  # just non-male sires
    assert (
        get_parent_sex_mismatches(ped, [lbls[i] for i in [0, 2]], sex_codes=(2,)).height
        == 1
    )  # just non-female dams
    assert get_parent_sex_mismatches(*ped_errors_sex).height == 3  # all mismatches


def test_no_anims_born_before_parents(ped_jv_classified):
    assert (
        get_animals_born_before_parents(
            *ped_jv_classified, age_label="generation"
        ).height
        == 0
    )


def test_find_animals_born_before_their_parents(ped_circular_classified):
    assert (
        get_animals_born_before_parents(
            *ped_circular_classified, age_label="generation"
        ).height
        > 0
    )


def test_classify_and_find_animals_born_before_their_parents(ped_circular):
    assert get_animals_born_before_parents(*ped_circular).collect().height > 0


def test_number_of_multiple_records_found(ped_errors):
    assert get_animals_with_multiple_records(*ped_errors).height == 2


def test_no_multiple_records_found(ped_lit):
    assert get_animals_with_multiple_records(*ped_lit).height == 0


def test_find_parents_that_are_both_sire_and_dam(ped_errors):
    assert get_parents_both_sires_and_dams(ped_errors[0]).height > 0


def test_no_parents_are_both_sire_and_dam(ped_jv):
    assert get_parents_both_sires_and_dams(ped_jv[0]).height == 0


def test_all_parents_have_own_record(ped_jv):
    assert get_parents_without_own_record(*ped_jv).height == 0


def test_find_parents_without_own_record(ped_lit):
    assert get_parents_without_own_record(*ped_lit)["parent"].sort().to_list() == [
        "Daisey",
        "Fatma",
        "George",
        "Gloria",
        "Jessica",
        "Jim",
        "Michelle",
        "Susan",
        "Tom",
    ]


def test_validate_invalid_pedigree_no_age(ped_lit):
    ped, lbls = ped_lit
    valid, errors = validate_pedigree(ped, lbls)
    assert not valid
    assert errors.height > 0


def test_validate_valid_pedigree_no_age(ped_lit):
    ped, lbls = ped_lit
    valid, errors = validate_pedigree(add_missing_records(ped, lbls), lbls)
    assert valid
    assert errors.height == 0


def test_validate_valid_pedigree(ped_jv_classified):
    valid, errors = validate_pedigree(*ped_jv_classified, age_label="generation")
    assert valid
    assert errors.height == 0


def test_validate_invalid_pedigree(ped_errors):
    valid, errors = validate_pedigree(*ped_errors)
    assert not valid
    assert errors.height > 0


def test_add_record_for_parents_without_their_own(ped_errors):
    assert add_missing_records(*ped_errors).height == (ped_errors[0].height + 1)


def test_null_parents_without_own_record(ped_lit):
    assert (
        null_parents_without_own_record(*ped_lit)
        .select(parents(("Father", "Mother")))
        .height
        == 5
    )


def test_recode_ids(ped_lit):
    ped, lbls = ped_lit
    tmp = add_missing_records(ped, lbls)
    recoded_ped, id_map = recode_pedigree(tmp, lbls)
    assert recoded_ped["Child"].to_list() == [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
    ]
