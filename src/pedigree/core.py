import polars as pl

PedigreeLabels = ("animal", "sire", "dam")
ParentLabels = ("sire", "dam")


def is_integer(df, column):
    """Determines if the specified column in the DataFrame has an integer type"""
    return isinstance(df.schema[column], (pl.Int64, pl.Int32, pl.Int16))


def get_unknown_parent_value(df, parent_label):
    """Determines the value used to represent unknown sires/dams

    Assumes `0` if parent Ids are integers, or `'.'` if the parent Ids are literals."""
    map_null = {True: 0, False: "."}
    return map_null[is_integer(df, parent_label)]


# get_unknown_parent_value(ped, "Father")


def null_unknown_parents(
    df: pl.LazyFrame | pl.DataFrame,
    parent_labels: tuple[str, str] = ParentLabels,
    unknown_parent_value=None,
) -> pl.LazyFrame | pl.DataFrame:
    """Replaces the parent Id used to represent 'unknown' with null"""
    if unknown_parent_value is None:
        unknown_parent_value = get_unknown_parent_value(df, parent_labels[0])
    return df.with_columns(
        [
            pl.when(pl.col(label) == unknown_parent_value)
            .then(pl.lit(None))
            .otherwise(pl.col(label))
            .alias(label)
            for label in parent_labels
        ]
    )


def pedigree_ids(pedigree_labels: tuple[str, str, str] = PedigreeLabels) -> pl.Expr:
    """Returns an expression describing all the Ids in a pedigree"""
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
    """Returns an expression describing the parents in a pedigree"""
    sire, dam = parent_labels[0], parent_labels[-1]
    return pl.col(sire).append(pl.col(dam)).drop_nulls().unique().alias("parents")


def get_parents(
    df: pl.LazyFrame | pl.DataFrame, parent_labels: tuple[str, str] = ParentLabels
) -> pl.LazyFrame | pl.DataFrame:
    return df.select(parents(parent_labels=parent_labels))
