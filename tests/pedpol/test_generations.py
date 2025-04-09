from pedpol.generations import (
    classify_generations,
    get_ancestors_of,
    get_descendants_of,
    get_parents_of,
    get_progeny_of,
)


def test_generation_classification_of_valid_pedigree(ped_jv_classified):
    assert ped_jv_classified[0].get_column("generation").value_counts().sort(
        by="generation"
    ).get_column("count").to_list() == [2, 6, 4, 2, 1]


def test_generation_classification_of_valid_pedigree_direct(ped_jv):
    ped, lbls = ped_jv
    assert classify_generations(ped, lbls).get_column("generation").value_counts().sort(
        by="generation"
    ).get_column("count").to_list() == [2, 6, 4, 2, 1]


def test_generation_classification_of_invalid_pedigree(ped_circular_classified):
    assert ped_circular_classified[0].get_column("generation").value_counts().sort(
        by="generation"
    ).get_column("count").to_list() == [9, 1]


def test_get_progeny_of_single_id(ped_jv):
    ped, lbls = ped_jv
    ids = [3]
    assert get_progeny_of(ped, ids, pedigree_labels=lbls).collect().height == 3


def test_lazy_get_progeny_of_single_id(ped_jv):
    ped, lbls = ped_jv
    ids = [3]
    assert get_progeny_of(ped.lazy(), ids, pedigree_labels=lbls).collect().height == 3


def test_get_parents_of_founder(ped_jv):
    ped, lbls = ped_jv
    ids = [3]
    assert get_parents_of(ped, ids, pedigree_labels=lbls).collect().height == 0


def test_get_parents_of_multiple_ids(ped_jv):
    ped, lbls = ped_jv
    ids = [11, 15]
    assert get_parents_of(ped, ids, pedigree_labels=lbls).collect().height == 2


def test_get_ancestors_of_single_id(ped_jv):
    ped, lbls = ped_jv
    ids = [6]
    assert get_ancestors_of(ped, ids, pedigree_labels=lbls).height == 9


def test_lazy_get_ancestors_of_single_id(ped_jv):
    ped, lbls = ped_jv
    ids = [6]
    assert get_ancestors_of(ped.lazy(), ids, pedigree_labels=lbls).collect().height == 9


def test_get_ancestors_of_multiple_literal_ids(ped_lit_valid):
    ped, lbls = ped_lit_valid
    ids = ["Barry", "Emily"]
    assert get_ancestors_of(ped, ids, pedigree_labels=lbls).height == 13


def test_get_1gen_ancestors_same_as_get_parents(ped_lit_valid):
    ped, lbls = ped_lit_valid
    ids = ["Barry", "Emily"]
    assert (
        get_ancestors_of(
            ped, ids, pedigree_labels=lbls, generations=1, include_ids=False
        ).height
        == get_parents_of(ped, ids, pedigree_labels=lbls).collect().height
    )


def test_get_descendents_of_single_id(ped_jv):
    ped, lbls = ped_jv
    assert get_descendants_of(ped, [3], pedigree_labels=lbls).height == 11


def test_get_descendents_of_single_literal_id(ped_lit_valid):
    ped, lbls = ped_lit_valid
    assert get_descendants_of(ped, ["Harry"], pedigree_labels=lbls).height == 8


def test_get_descendents_of_multiple_ids(ped_jv):
    ped, lbls = ped_jv
    ids = [11, 15]
    assert get_descendants_of(ped, ids, pedigree_labels=lbls).height == 8


def test_get_descendents_of_multiple_ids_when_include_ids_false(ped_jv):
    ped, lbls = ped_jv
    ids = [11, 15]
    assert get_descendants_of(
        ped, ids, include_ids=False, pedigree_labels=lbls
    ).height == (8 - len(ids))
