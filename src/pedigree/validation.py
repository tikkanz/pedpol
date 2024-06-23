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


def get_animals_born_before_parents(
    pedigree: pl.LazyFrame | pl.DataFrame,
    pedigree_labels: tuple[str, str, str] = PedigreeLabels,
    age_column: str = "generation",
) -> pl.LazyFrame | pl.DataFrame:
    """Returns any animals that are used as a sire before they were born

    `age_column` is column in `pedigree` to use for age comparisons. For example
    'birth_year' or 'generation'"""
    animal, sire, dam = pedigree_labels
    pedigree = pedigree.lazy()
    return pl.concat(
        [
            pedigree.join(
                pedigree.select(animal, age_column),
                left_on=sire,
                right_on=animal,
                suffix="_parent",
            )
            .filter(pl.col(age_column) <= pl.col(f"{age_column}_parent"))
            .with_columns(pl.lit(f"born before {sire}").alias("error")),
            pedigree.join(
                pedigree.select(animal, age_column),
                left_on=dam,
                right_on=animal,
                suffix="_parent",
            )
            .filter(pl.col(age_column) <= pl.col(f"{age_column}_parent"))
            .with_columns(pl.lit(f"born before {dam}").alias("error")),
        ]
    ).collect()


def get_animals_are_sires_before_birth(
    pedigree: pl.LazyFrame | pl.DataFrame,
    pedigree_labels: tuple[str, str, str] = PedigreeLabels,
    age_column: str = "generation",
) -> pl.LazyFrame | pl.DataFrame:
    """Returns any animals that are used as a sire before they were born

    `age_column` is column in `pedigree` to use for age comparisons. For example
    'birth_year' or 'generation'"""
    animal, sire, dam = pedigree_labels

    return (
        pedigree.lazy()
        .join(
            pedigree.lazy().group_by(sire).agg(pl.col(age_column).min()),
            left_on=animal,
            right_on=sire,
            suffix="_as_parent",
        )
        .filter(pl.col(age_column) >= pl.col(f"{age_column}_as_parent"))
        .collect()
    )


def get_animals_are_dams_before_birth(
    pedigree: pl.LazyFrame | pl.DataFrame,
    pedigree_labels: tuple[str, str, str] = PedigreeLabels,
    age_column: str = "generation",
) -> pl.LazyFrame | pl.DataFrame:
    """Returns any animals that are used as a dam before they were born

    `age_column` is column in `pedigree` to use for age comparisons. For example
    'birth_year' or 'generation'"""
    animal, sire, dam = pedigree_labels

    return pedigree.join(
        pedigree.group_by(dam).agg(pl.col(age_column).min()),
        left_on=animal,
        right_on=dam,
        suffix="_as_parent",
    ).filter(pl.col(age_column) >= pl.col(f"{age_column}_as_parent"))


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


def null_parents_without_own_record(
    pedigree: pl.LazyFrame | pl.DataFrame,
    pedigree_labels: tuple[str, str, str] = PedigreeLabels,
) -> pl.LazyFrame | pl.DataFrame:
    """Returns pedigree where parents without their own record are marked as `null`"""
    parent_cols = pedigree_labels[1:]
    null_parents = get_parents_without_own_record(
        pedigree, pedigree_labels=pedigree_labels
    )
    return pedigree.with_columns(
        *[
            pl.when(pl.col(col).is_in(null_parents))
            .then(pl.lit(None))
            .otherwise(pl.col(col))
            .alias(col)
            for col in parent_cols
        ]
    )


def recode_pedigree(
    pedigree: pl.LazyFrame | pl.DataFrame,
    pedigree_labels: tuple[str, str, str] = PedigreeLabels,
) -> tuple[pl.LazyFrame | pl.DataFrame]:
    """Recodes a pedigree to use integer ids from 1 to the number of animals in the pedigree

    Returns recoded pedigree & map of old id to recoded id."""
    # If not all parents have their own record then raise ValueError
    animal, sire, dam = pedigree_labels
    no_own_record = get_parents_without_own_record(pedigree, pedigree_labels)
    if (err_count := no_own_record.height) != 0:
        raise ValueError(
            f"{err_count} parents did not have their own record in the pedigree: \n {no_own_record}"
        )

    id_map = pedigree.select(animal).with_row_index(name="recoded", offset=1)

    new_pedigree = (
        pedigree.join(id_map, on=animal, how="left")
        .join(id_map, left_on=sire, right_on=animal, how="left", suffix="_sire")
        .join(id_map, left_on=dam, right_on=animal, how="left", suffix="_dam")
    )
    return new_pedigree.select(
        pl.col("recoded").alias(animal),
        pl.col("recoded_sire").alias(sire),
        pl.col("recoded_dam").alias(dam),
        pl.all().exclude(animal, sire, dam, "recoded", "recoded_sire", "recoded_dam"),
    ), id_map
