from typing import Collection

import polars as pl

from pedigree.core import PedigreeLabels, parents


def get_progeny_of(
    df: pl.DataFrame | pl.LazyFrame,
    ids: pl.Expr | Collection[any] | pl.Series,
    pedigree_labels: tuple[str, str, str] = PedigreeLabels,
) -> pl.DataFrame | pl.LazyFrame:
    """Return the progeny of the animals specified"""
    return df.filter(pl.any_horizontal(pl.col(pedigree_labels[1:]).is_in(ids)))


def get_parents_of(
    df: pl.DataFrame | pl.LazyFrame,
    ids: pl.Expr | Collection[any] | pl.Series,
    pedigree_labels: tuple[str, str, str] = PedigreeLabels,
) -> pl.DataFrame | pl.LazyFrame:
    """Return the parents of the animals specified"""
    animal, sire, dam = pedigree_labels
    anims = df.filter(pl.col(animal).is_in(ids))
    return df.filter(pl.col(animal).is_in(anims.select(parents((sire, dam)))))

def get_descendents_of(
    df: pl.DataFrame | pl.LazyFrame,
    ids: pl.Expr | Collection[any] | pl.Series,
    generations: int | None = None,
    include_ids: bool = True,
    pedigree_labels: tuple[str, str, str] = PedigreeLabels,
) -> pl.DataFrame | pl.LazyFrame:
    """Return the descendents of the animals specified"""
    animal = pedigree_labels[0]
    ids = pl.DataFrame({animal: ids}, schema=df.select(animal).schema)
    ids_descendants = pl.DataFrame(schema=ids.schema)
    g = 0
    ids_g = ids
    while generations > g and ids_g.height != 0:
        ids_g = get_progeny_of(df, ids_g, pedigree_labels).select(animal)
        ids_descendants = pl.concat([ids_descendants, ids_g])
        g += 1

    if include_ids:
        ids_descendants = pl.concat([ids_descendants, ids])
    return df.filter(pl.col(animal).is_in(ids_descendants))


def get_ancestors_of(
    df: pl.DataFrame | pl.LazyFrame,
    ids: pl.Expr | Collection[any] | pl.Series,
    generations: int | None = None,
    include_ids: bool = True,
    pedigree_labels: tuple[str, str, str] = PedigreeLabels,
) -> pl.DataFrame | pl.LazyFrame:
    """Return the ancestors of the animals specified"""
    animal = pedigree_labels[0]
    ids = pl.DataFrame({animal: ids}, schema=df.select(animal).schema)
    ids_ancestors = pl.DataFrame(schema=ids.schema)
    g = 0
    ids_g = ids
    while generations > g and ids_g.height != 0:
        ids_g = get_parents_of(df, ids_g, pedigree_labels).select(animal)
        ids_ancestors = pl.concat([ids_ancestors, ids_g])
        g += 1

    if include_ids:
        ids_ancestors = pl.concat([ids_ancestors, ids])
    return df.filter(pl.col(animal).is_in(ids_ancestors))

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
