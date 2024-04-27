import polars as pl

PedigreeLabels = ("animal", "sire", "dam")
"""Default labels for individual (1st) male parent (2nd) & female parent (3rd)"""

ParentLabels = ("sire", "dam")
"""Default labels for male (1st) & female (2nd) parents"""


def is_integer(df: pl.LazyFrame | pl.DataFrame, column) -> bool:
    """Determines if the specified column in the DataFrame has an integer type"""
    return isinstance(df.schema[column], (pl.Int64, pl.Int32, pl.Int16))


def get_unknown_parent_value(
    pedigree: pl.LazyFrame | pl.DataFrame, parent_label: str
) -> int | str:
    """Determines the value used to represent unknown sires/dams

    Assumes `0` if parent Ids are integers, or `'.'` if the parent Ids are literals.
    ```python
    get_unknown_parent_value(ped_df, "Father")
    ```
    """
    map_null = {True: 0, False: "."}
    return map_null[is_integer(pedigree, parent_label)]


def null_unknown_parents(
    pedigree: pl.LazyFrame | pl.DataFrame,
    parent_labels: tuple[str, str] = ParentLabels,
    unknown_parent_value=None,
) -> pl.LazyFrame | pl.DataFrame:
    """Replaces the parent Id used to represent an 'unknown' parent with null

    ```python
    null_unknown_parents(ped_df, ("Father", "Mother"))
    ```
    """
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

    ```python
    ped_df.select(pedigree_ids("indiv","sire","dam"))
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


def parents(parent_labels: tuple[str, str] = ParentLabels) -> pl.Expr:
    """Returns an expression describing the parents in a pedigree

    ```python
    ped_df.select(parents("Father","Mother"))
    ```"""
    sire, dam = parent_labels[0], parent_labels[-1]
    return pl.col(sire).append(pl.col(dam)).drop_nulls().unique().alias("parents")


def get_parents(
    pedigree: pl.LazyFrame | pl.DataFrame, parent_labels: tuple[str, str] = ParentLabels
) -> pl.LazyFrame | pl.DataFrame:
    """Returns a Dataframe containing the parents in a pedigree

    ```python
    ped_df.pipe(get_parents, ("Father","Mother"))
    ```"""
    return pedigree.select(parents(parent_labels=parent_labels))
