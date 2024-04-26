import polars as pl

from pedigree.core import ParentLabels, PedigreeLabels, parents


def get_parents_both_sires_and_dams(
    pedigree: pl.LazyFrame | pl.DataFrame,
    parent_labels: tuple[str, str] = ParentLabels,
) -> pl.LazyFrame | pl.DataFrame:
    """Returns any animals that are both sires and dams"""
    sire, dam = parent_labels
    return (
        pedigree.select(parents((sire,)))
        .join(pedigree.select(dam), left_on="parents", right_on=dam, how="inner")
        .unique()
    )


# to get records for all animals that are occur as both sire & dam
# pederr.filter(pl.any_horizontal(pl.col("sire", "dam").is_in(pederr.pipe(pc.get_parents_both_sires_and_dams, ("sire", "dam")))))


def get_animals_with_multiple_records(
    pedigree: pl.LazyFrame | pl.DataFrame,
    pedigree_labels: tuple[str, str, str] = PedigreeLabels,
) -> pl.LazyFrame | pl.DataFrame:
    """Returns any animals that have multiple pedigree records"""
    animal = pedigree_labels[0]
    return pedigree.filter(pedigree.select(animal).is_duplicated())


def get_parents_without_own_record(
    pedigree: pl.LazyFrame | pl.DataFrame,
    pedigree_labels: tuple[str, str, str] = PedigreeLabels,
) -> pl.LazyFrame | pl.DataFrame:
    """Returns the parents that don't have their own record in the pedigree"""
    animal = pedigree_labels[0]
    return pedigree.select(parents(pedigree_labels[1:])).join(
        pedigree, left_on="parents", right_on=animal, how="anti"
    )


def get_animals_are_own_parent(
    pedigree: pl.LazyFrame | pl.DataFrame,
    pedigree_labels: tuple[str, str, str] = PedigreeLabels,
) -> pl.LazyFrame | pl.DataFrame:
    """Returns any animals that are their own parent"""
    animal, sire, dam = pedigree_labels
    return pedigree.filter(
        pl.any_horizontal([pl.col(animal).is_in(pl.concat_list([sire, dam]))])
    )


def add_missing_records(
    pedigree: pl.LazyFrame | pl.DataFrame,
    pedigree_labels: tuple[str, str, str] = PedigreeLabels,
) -> pl.LazyFrame | pl.DataFrame:
    """Returns pedigree with new records added for parents without their own"""
    animal = pedigree_labels[0]

    new_records = get_parents_without_own_record(
        pedigree, pedigree_labels=pedigree_labels
    ).select(
        pl.col("parents").alias(animal),
        *[
            pl.lit(None).cast(dtype).alias(col)
            for col, dtype in pedigree.schema.items()
            if col != animal
        ],
    )

    return pl.concat([new_records, pedigree])
