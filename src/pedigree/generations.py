import polars as pl


def is_integer(df, column):
    """Determines if the specified column in the DataFrame has an integer type"""
    return isinstance(df.schema[column], (pl.Int64, pl.Int32, pl.Int16))


def get_unknown_parent_value(df, parent_column):
    """Determines the value used to represent unknown sires/dams

    Assumes `0` if parent Ids are integers, or `'.'` if the parent Ids are literals."""
    map_null = {True: 0, False: "."}
    return map_null[is_integer(df, parent_column)]


# get_unknown_parent_value(ped, "Father")


def null_unknown_parents(
    df: pl.LazyFrame | pl.DataFrame,
    parent_labels: tuple[str, str] = ("sire", "dam"),
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


def parents(parent_labels: tuple[str, str] = ("sire", "dam")) -> pl.Expr:
    """Returns an expression describing the parents in a pedigree"""
    sire, dam = parent_labels[0], parent_labels[-1]
    return pl.col(sire).append(pl.col(dam)).drop_nulls().unique().alias("parents")


def get_parents(
    df: pl.LazyFrame | pl.DataFrame, parent_labels: tuple[str, str] = ("sire", "dam")
) -> pl.LazyFrame | pl.DataFrame:
    return df.select(parents(parent_labels=parent_labels))


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
