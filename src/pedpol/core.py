import polars as pl

PedigreeLabels = ("animal", "sire", "dam")
"""Default labels for individual (1st) male parent (2nd) & female parent (3rd)"""

SexLabel = "sex"
"""Default label for column describing the sex of an animal"""

SexIds = (1, 2)
"""Default Ids for male (1st) & female (2nd) animals"""

SexCodes = {
    1: 1,
    2: 2,
    "M": 1,
    "F": 2,
    "male": 1,
    "female": 2,
}
"""Mapping of common sex codes to SexId"""


def is_integer(df: pl.LazyFrame | pl.DataFrame, column) -> bool:
    """Determines if the specified column in the DataFrame has an integer type"""
    return isinstance(df.schema[column], (pl.Int64, pl.Int32, pl.Int16))


def get_unknown_parent_value(
    pedigree: pl.LazyFrame | pl.DataFrame, parent_label: str
) -> int | str:
    """Determines the value used to represent unknown sires/dams

    Assumes `0` if parent Ids are integers, or `'.'` if the parent Ids are literals.
    ### Example use:
    ```python
    get_unknown_parent_value(ped_df, "Father") # -> '.'
    ```
    """
    map_null = {True: 0, False: "."}
    return map_null[is_integer(pedigree, parent_label)]


def null_unknown_parents(
    pedigree: pl.LazyFrame | pl.DataFrame,
    parent_labels: tuple[str, str] = PedigreeLabels[1:],
    unknown_parent_value=None,
) -> pl.LazyFrame | pl.DataFrame:
    """Replaces the parent Id used to represent an 'unknown' parent with null

    ### Example use:
    ```python
    null_unknown_parents(ped_df, ("Father", "Mother"))
    ```"""
    if unknown_parent_value is None:
        unknown_parent_value = get_unknown_parent_value(pedigree, parent_labels[0])
    return pedigree.with_columns(
        [
            pl.when(pl.col(label) == unknown_parent_value)
            .then(pl.lit(None))
            .otherwise(pl.col(label))
            .alias(label)
            for label in parent_labels
        ]
    )


def pedigree_ids(pedigree_labels: tuple[str, str, str] = PedigreeLabels) -> pl.Expr:
    """Returns an expression describing all the Ids in a pedigree

    ### Example use:
    ```python
    allIds_df = ped_df.select(pedigree_ids(("indiv","sire","dam")))
    ```"""
    animal, sire, dam = pedigree_labels
    return (
        pl.col(animal)
        .append(pl.col(sire))
        .append(pl.col(dam))
        .drop_nulls()
        .unique()
        .alias("animal")
    )


def known_unique(column_expr: pl.Expr) -> pl.Expr:
    return column_expr.drop_nulls().unique()


def sires(sire_label: str = PedigreeLabels[1]) -> pl.Expr:
    """Returns an expression describing the unique sires in a pedigree"""
    return known_unique(sire_label).alias("Parent"), pl.lit(sire_label).alias(
        "Parent_type"
    )


def dams(dam_label: str = PedigreeLabels[2]) -> pl.Expr:
    """Returns an expression describing the unique dams in a pedigree"""
    return known_unique(dam_label).alias("Parent"), pl.lit(dam_label).alias(
        "Parent_type"
    )


def parents(parent_labels: tuple[str] = PedigreeLabels[1:]) -> pl.Expr:
    """Returns an expression describing the parents in a pedigree

    ### Example use:
    ```python
    parents_df = ped_df.select(parents(("Father","Mother")))
    sires_df = ped_df.select(parents(("Father",)))
    ```"""
    return pl.concat([known_unique(pl.col(pnt)) for pnt in parent_labels]).alias(
        "parent"
    )
