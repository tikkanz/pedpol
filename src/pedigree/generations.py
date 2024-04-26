import polars as pl

from pedigree.core import PedigreeLabels, parents


def classify_generations(
    df: pl.DataFrame | pl.LazyFrame,
    pedigree_labels: tuple[str, str, str] = PedigreeLabels,
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
