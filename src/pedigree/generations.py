import polars as pl


def parents(parent_labels: tuple[str, str] = ("sire", "dam")) -> pl.Expr:
    """Returns an expression describing the list of the parents in a pedigree"""
    sire, dam = parent_labels
    return pl.col(sire).append(pl.col(dam)).unique()


def classify_generations(
    df: pl.DataFrame | pl.LazyFrame,
    pedigree_labels: tuple[str, str, str] = ("animal", "sire", "dam"),
) -> pl.DataFrame:
    """Add column classifying the animals into generations within the pedigree"""
    animal, sire, dam = pedigree_labels
    g = 0
    df = df.with_columns(pl.lit(g).alias("generation")).lazy().collect()
    df_parents = df.filter(pl.col("generation") == g)
    while df_parents.height > 0:
        df = df.with_columns(
            pl.when(
                pl.col(animal).is_in(
                    df_parents.select(pl.col(sire).append(pl.col(dam)))
                )
            )
            .then(pl.col("generation") + 1)
            .otherwise("generation")
        )
        df_parents = df.filter(pl.col("generation") > g)
        g += 1

    return df
