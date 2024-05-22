import polars as pl
import pytest
from pedigree.checks import (
    get_animals_are_own_parent,
    get_animals_with_multiple_records,
    get_parents_both_sires_and_dams,
    get_parents_without_own_record,
)

pl.Config.set_tbl_rows(15)


@pytest.fixture
def anims_are_own_parent(ped_errors):
    return get_animals_are_own_parent(ped_errors, ("anim", "sire", "dam"))


def test_no_anims_are_own_parent(ped_lit):
    assert (
        get_animals_are_own_parent(ped_lit, ("Child", "Father", "Mother")).height == 0
    )


def test_number_of_animals_that_are_their_own_parent(anims_are_own_parent):
    assert anims_are_own_parent.height == 1


def test_record_of_animal_that_is_their_own_parent(anims_are_own_parent):
    assert anims_are_own_parent.to_dicts() == [{"anim": 5, "sire": 9, "dam": 5}]


def test_number_of_multiple_records_found(ped_errors):
    assert (
        get_animals_with_multiple_records(ped_errors, ("anim", "sire", "dam")).height
        == 2
    )


def test_no_multiple_records_found(ped_lit):
    assert (
        get_animals_with_multiple_records(ped_lit, ("Child", "Father", "Mother")).height
        == 0
    )


def test_find_parents_that_are_both_sire_and_dam(ped_errors):
    assert get_parents_both_sires_and_dams(ped_errors).height > 0


def test_no_parents_are_both_sire_and_dam(ped_jv):
    assert get_parents_both_sires_and_dams(ped_jv).height == 0


def test_all_parents_have_own_record(ped_jv):
    assert (
        get_parents_without_own_record(ped_jv, ("progeny", "sire", "dam")).height == 0
    )


def test_find_parents_without_own_record(ped_lit):
    assert get_parents_without_own_record(ped_lit, ("Child", "Father", "Mother"))[
        "parents"
    ].sort().to_list() == [
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
