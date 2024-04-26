import polars as pl


def is_integer(df, column):
    return isinstance(df.schema[column], (pl.Int64, pl.Int32, pl.Int16))


def get_unknown_parent_value(df, parent_column):
    """Determines if"""
    map_null = {True: 0, False: "."}
    return map_null[is_integer(df, parent_column)]


# get_unknown_parent_value(ped, "Father")


def null_unknown_parents(
    df: pl.LazyFrame | pl.DataFrame,
    parent_labels: tuple[str, str] = ("sire", "dam"),
    unknown_parent_value=None,
) -> pl.LazyFrame | pl.DataFrame:
    """Replaces the parent ID used to represent 'unknown' with null"""
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


def parents(parent_labels: tuple[str, str] = ("sire", "dam"), null_value=0) -> pl.Expr:
    """Returns an expression describing the parents in a pedigree"""
    sire, dam = parent_labels[0], parent_labels[-1]
    return (
        pl.col(sire)
        .filter(pl.col(sire) != null_value)
        .append(pl.col(dam).filter(pl.col(dam) != null_value))
        .unique()
        .alias("parents")
    )


def get_parents(
    df: pl.LazyFrame | pl.DataFrame,
    parent_labels: tuple[str, str] = ("sire", "dam"),
    null_value=None,
) -> pl.LazyFrame | pl.DataFrame:
    if null_value is None:
        null_value = get_unknown_parent_value(df, parent_labels[0])
    return df.select(parents(parent_labels=parent_labels, null_value=null_value))


def classify_generations(
    df: pl.DataFrame | pl.LazyFrame,
    pedigree_labels: tuple[str, str, str] = ("animal", "sire", "dam"),
) -> pl.DataFrame:
    """Add column classifying the animals into generations within the pedigree"""
    animal, sire, dam = pedigree_labels
    g = 0
    df_parents = df = df.with_columns(pl.lit(g).alias("generation")).lazy().collect()
    while df_parents.height > 0:
        df = df.with_columns(
            pl.when(pl.col(animal).is_in(df_parents.select(parents((sire, dam)))))
            .then(pl.col("generation") + 1)
            .otherwise("generation")
        )
        df_parents = df.filter(pl.col("generation") > g)
        g += 1

    return df.with_columns(
        (pl.col("generation").max() - pl.col("generation")).alias("generation")
    )  # reverse generation order (0 is earliest/oldest)
