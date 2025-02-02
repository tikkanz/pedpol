import polars as pl

from pedpol.core import PedigreeLabels, SexIds, SexLabel, parents
from pedpol.generations import classify_generations


def get_parents_both_sires_and_dams(
    pedigree: pl.LazyFrame | pl.DataFrame,
    parent_labels: tuple[str, str] = PedigreeLabels[1:],
) -> pl.LazyFrame | pl.DataFrame:
    """Returns any animals that are both sires and dams"""
    sire, dam = parent_labels
    return (
        pedigree.select(parents((sire,)))
        .join(pedigree.select(dam), left_on="parent", right_on=dam, how="inner")
        .unique()
    )


# to get records for all animals that are occur as both sire & dam
# pederr.filter(pl.any_horizontal(pl.col("sire", "dam").is_in(pederr.pipe(pc.get_parents_both_sires_and_dams, ("sire", "dam")))))


def get_animals_with_multiple_records(
    pedigree: pl.LazyFrame | pl.DataFrame,
    pedigree_labels: tuple[str, str, str] = PedigreeLabels,
) -> pl.DataFrame:
    """Returns any animals that have multiple pedigree records"""
    animal = pedigree_labels[0]
    return pedigree.filter(pedigree.lazy().select(animal).collect().is_duplicated())


def get_parents_without_own_record(
    pedigree: pl.LazyFrame | pl.DataFrame,
    pedigree_labels: tuple[str, str, str] = PedigreeLabels,
) -> pl.LazyFrame | pl.DataFrame:
    """Returns the parents that don't have their own record in the pedigree"""
    animal = pedigree_labels[0]
    return pedigree.select(parents(pedigree_labels[1:])).join(
        pedigree, left_on="parent", right_on=animal, how="anti"
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
    age_label: str | None = None,
) -> pl.LazyFrame | pl.DataFrame:
    """Returns any animals that are used as a sire before they were born

    `age_label` is column in `pedigree` to use for age comparisons. For example
    'birth_year' or 'generation'"""
    if age_label is None:
        pedigree = classify_generations(
            pedigree, pedigree_labels=pedigree_labels
        ).lazy()
        age_label = "generation"

    animal, sire, dam = pedigree_labels
    return pl.concat(
        [
            pedigree.join(
                pedigree.select(animal, age_label),
                left_on=sire,
                right_on=animal,
                suffix="_parent",
            )
            .filter(pl.col(age_label) <= pl.col(f"{age_label}_parent"))
            .with_columns(pl.lit(f"was born before {sire}").alias("error")),
            pedigree.join(
                pedigree.select(animal, age_label),
                left_on=dam,
                right_on=animal,
                suffix="_parent",
            )
            .filter(pl.col(age_label) <= pl.col(f"{age_label}_parent"))
            .with_columns(pl.lit(f"was born before {dam}").alias("error")),
        ]
    )


def get_animals_are_parents_before_birth(
    pedigree: pl.LazyFrame | pl.DataFrame,
    pedigree_labels: tuple[str, str, str] = PedigreeLabels,
    age_label: str = "generation",
) -> pl.LazyFrame | pl.DataFrame:
    """Returns any animals that are used as a parent before they were born

    `age_label` is column in `pedigree` to use for age comparisons. For example
    'birth_year' or 'generation'"""
    animal = pedigree_labels[0]

    mated_before_birth = []
    for parent_type in pedigree_labels[1:]:
        not_born_yet = pedigree.join(
            pedigree.group_by(parent_type).agg(pl.col(age_label).min()),
            left_on=animal,
            right_on=parent_type,
            suffix="_as_parent",
        ).filter(pl.col(age_label) >= pl.col(f"{age_label}_as_parent"))

        mated_before_birth.append(not_born_yet)

    return pl.concat(mated_before_birth)


def get_parent_sex_mismatches(
    pedigree: pl.LazyFrame | pl.DataFrame,
    pedigree_labels: tuple[str, str, str] = PedigreeLabels,
    sex_label: str = SexLabel,
    sex_codes: tuple[any, any] = SexIds,
) -> pl.LazyFrame | pl.DataFrame:
    """Returns records where parent's sex doesn't match parent type (e.g. female sires)

    Relies on order of sex_codes matching order of parents in pedigree_labels,
    i.e. both sire label & male id come first."""
    animal = pedigree_labels[0]

    mismatches = []
    for parent_type, sex in zip(pedigree_labels[1:], sex_codes[:2]):
        mismatch = (
            pedigree.filter(pl.col(sex_label) != sex)
            .join(pedigree.select(parent_type), left_on=animal, right_on=parent_type)
            .unique()
        )
        mismatches.append(mismatch)
    return pl.concat(mismatches)


def validate_pedigree(
    pedigree: pl.LazyFrame | pl.DataFrame,
    pedigree_labels: tuple[str, str, str] = PedigreeLabels,
    age_label: str | None = None,
    sex_label: str | None = None,
    sex_codes: tuple[any, any] | None = None,
) -> tuple[bool, pl.DataFrame]:
    """Validates a pedigree

    Checks for:
     * parents without their own individual record
     * animals with multiple individual records
     * animals that occur as both sires & dams
     * animals that are their own parent
     * animals born before their parents (if `age_label` is None then
       generations will be classified)
     * animals with mis-matched sex and parent type (requires `sex_label`)
    """
    # Raise error if pedigree doesn't have 3 columns (animal, sire, dam)
    if missing_lbls := [
        lbl
        for lbl in pedigree_labels
        if lbl not in pedigree.lazy().collect_schema().names()
    ]:
        raise ValueError(f"Required column(s) {missing_lbls} not found in pedigree.")

    pedigree = pedigree.lazy().collect().lazy()
    errors = []
    errors.append(
        get_animals_born_before_parents(
            pedigree, pedigree_labels=pedigree_labels, age_label=age_label
        ).select(*pedigree.collect_schema().names(), "error")
    )
    errors.append(
        get_missing_records(pedigree, pedigree_labels=pedigree_labels).with_columns(
            pl.lit("has no own record").alias("error")
        )
    )  # records for parents with no own record
    errors.append(
        get_animals_are_own_parent(
            pedigree, pedigree_labels=pedigree_labels
        ).with_columns(pl.lit("is own parent").alias("error"))
    )
    errors.append(
        get_animals_with_multiple_records(
            pedigree, pedigree_labels=pedigree_labels
        ).with_columns(pl.lit("has multiple own records").alias("error"))
    )
    errors.append(
        pedigree.join(
            get_parents_both_sires_and_dams(
                pedigree, parent_labels=pedigree_labels[1:3]
            ),
            left_on=pedigree_labels[0],
            right_on="parent",
        ).with_columns(pl.lit("is both sire and dam").alias("error"))
    )
    if sex_label:
        errors.append(
            get_parent_sex_mismatches(
                pedigree,
                pedigree_labels=pedigree_labels,
                sex_label=sex_label,
                sex_codes=sex_codes,
            ).with_columns(pl.lit("wrong sex for parental role").alias("error"))
        )

    errors = pl.concat(pl.collect_all(errors))
    is_valid_pedigree = errors.height == 0

    return is_valid_pedigree, errors


def get_missing_records(
    pedigree: pl.LazyFrame | pl.DataFrame,
    pedigree_labels: tuple[str, str, str] = PedigreeLabels,
) -> pl.LazyFrame | pl.DataFrame:
    """Returns new records for parents without their own"""
    animal = pedigree_labels[0]

    return get_parents_without_own_record(
        pedigree, pedigree_labels=pedigree_labels
    ).select(
        pl.col("parent").alias(animal),
        *[
            pl.lit(None).cast(dtype).alias(col)
            for col, dtype in pedigree.collect_schema().items()
            if col != animal
        ],
    )


def add_missing_records(
    pedigree: pl.LazyFrame | pl.DataFrame,
    pedigree_labels: tuple[str, str, str] = PedigreeLabels,
) -> pl.LazyFrame | pl.DataFrame:
    """Returns pedigree with new records added for parents without their own"""
    new_records = get_missing_records(pedigree, pedigree_labels=pedigree_labels)

    return pl.concat([new_records, pedigree])


def null_parents_without_own_record(
    pedigree: pl.LazyFrame | pl.DataFrame,
    pedigree_labels: tuple[str, str, str] = PedigreeLabels,
) -> pl.LazyFrame | pl.DataFrame:
    """Returns pedigree where parents without their own record are marked as `null`"""
    parent_cols = pedigree_labels[1:]
    null_parents = (
        get_parents_without_own_record(pedigree.lazy(), pedigree_labels=pedigree_labels)
        .collect()
        .to_series()
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
        pedigree.join(id_map, on=animal, how="left", coalesce=False)
        .join(
            id_map,
            left_on=sire,
            right_on=animal,
            how="left",
            coalesce=False,
            suffix="_sire",
        )
        .join(
            id_map,
            left_on=dam,
            right_on=animal,
            how="left",
            coalesce=False,
            suffix="_dam",
        )
    )
    return new_pedigree.select(
        pl.col("recoded").alias(animal),
        pl.col("recoded_sire").alias(sire),
        pl.col("recoded_dam").alias(dam),
        pl.all().exclude(animal, sire, dam, "recoded", "recoded_sire", "recoded_dam"),
    ), id_map
