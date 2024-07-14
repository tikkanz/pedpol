def test_generation_classification_of_valid_pedigree(ped_jv_classified):
    assert ped_jv_classified[0].get_column("generation").value_counts().sort(
        by="generation"
    ).get_column("count").to_list() == [2, 6, 4, 2, 1]


def test_generation_classification_of_invalid_pedigree(ped_circular_classified):
    assert ped_circular_classified[0].get_column("generation").value_counts().sort(
        by="generation"
    ).get_column("count").to_list() == [9, 1]


# test_get_progeny_of

# test_get_parents_of

# test_get_ancestors_of

# test_get_descendents_of
