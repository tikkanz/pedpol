import polars as pl

from pedigree.core import ParentLabels, PedigreeLabels, parents


def get_parents_both_sires_and_dams(
    df: pl.LazyFrame | pl.DataFrame,
    pedigree_labels: tuple[str, str] = ParentLabels,
) -> pl.LazyFrame | pl.DataFrame:
    """Returns any animals that are both sires and dams"""
    sire, dam = pedigree_labels
    return df.select(parents((sire,))).join(
        df.select(dam), left_on="parents", right_on=dam, how="inner"
    )


# to get records for all animals that are occur as both sire & dam
# pederr.filter(pl.any_horizontal(pl.col("sire", "dam").is_in(pederr.pipe(pc.get_parents_both_sires_and_dams, ("sire", "dam")))))


def get_animals_with_multiple_records(
    df: pl.LazyFrame | pl.DataFrame,
    pedigree_labels: tuple[str, str, str] = PedigreeLabels,
) -> pl.LazyFrame | pl.DataFrame:
    """Returns any animals that have multiple pedigree records"""
    animal, sire, dam = pedigree_labels
    return df.filter(df.select(animal).is_duplicated())


def get_parents_without_own_record(
    df: pl.LazyFrame | pl.DataFrame,
    pedigree_labels: tuple[str, str, str] = PedigreeLabels,
) -> pl.LazyFrame | pl.DataFrame:
    """Returns the parents that don't have their own record in the pedigree"""
    animal = pedigree_labels[0]
    return df.select(parents(pedigree_labels[1:])).join(
        df, left_on="parents", right_on=animal, how="anti"
    )


def get_animals_are_own_parent(
    df: pl.LazyFrame | pl.DataFrame,
    pedigree_labels: tuple[str, str, str] = PedigreeLabels,
) -> pl.LazyFrame | pl.DataFrame:
    """Returns any animals that are their own parent"""
    animal, sire, dam = pedigree_labels
    return df.filter(
        pl.any_horizontal([pl.col(animal).is_in(pl.concat_list([sire, dam]))])
    )
