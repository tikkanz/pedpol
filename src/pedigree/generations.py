from typing import Collection

import polars as pl

from pedigree.core import PedigreeLabels, parents


def get_progeny_of(
    pedigree: pl.DataFrame | pl.LazyFrame,
    ids: pl.Expr | Collection[any] | pl.Series,
    pedigree_labels: tuple[str, str, str] = PedigreeLabels,
) -> pl.DataFrame | pl.LazyFrame:
    """Return the progeny of the animals specified"""
    return pedigree.filter(pl.any_horizontal(pl.col(pedigree_labels[1:]).is_in(ids)))


def get_parents_of(
    pedigree: pl.DataFrame | pl.LazyFrame,
    ids: pl.Expr | Collection[any] | pl.Series,
    pedigree_labels: tuple[str, str, str] = PedigreeLabels,
) -> pl.DataFrame | pl.LazyFrame:
    """Return the parents of the animals specified"""
    animal, sire, dam = pedigree_labels
    anims = pedigree.filter(pl.col(animal).is_in(ids))
    return pedigree.filter(pl.col(animal).is_in(anims.select(parents((sire, dam)))))


def _get_relatives_of(
    pedigree: pl.DataFrame | pl.LazyFrame,
    ids: pl.Expr | Collection[any] | pl.Series,
    relatives_function: callable,
    generations: int = 100,
    include_ids: bool = True,
    pedigree_labels: tuple[str, str, str] = PedigreeLabels,
):
    """General utility for iterating through pedigree to find relatives"""
    animal = pedigree_labels[0]
    ids = pl.DataFrame({animal: ids}, schema=pedigree.select(animal).schema)
    ids_relatives = pl.DataFrame(schema=ids.schema)
    g = 0
    ids_g = ids
    while generations > g and ids_g.height != 0:
        ids_g = relatives_function(pedigree, ids_g, pedigree_labels).select(animal)
        ids_relatives = pl.concat([ids_relatives, ids_g])
        g += 1

    if include_ids:
        ids_relatives = pl.concat([ids_relatives, ids])
    return pedigree.filter(pl.col(animal).is_in(ids_relatives))


def get_descendants_of(
    pedigree: pl.DataFrame | pl.LazyFrame,
    ids: pl.Expr | Collection[any] | pl.Series,
    generations: int = 100,
    include_ids: bool = True,
    pedigree_labels: tuple[str, str, str] = PedigreeLabels,
) -> pl.DataFrame | pl.LazyFrame:
    """Return the descendants of the animals specified"""
    return _get_relatives_of(
        pedigree,
        ids,
        get_progeny_of,
        generations=generations,
        include_ids=include_ids,
        pedigree_labels=pedigree_labels,
    )


def get_ancestors_of(
    pedigree: pl.DataFrame | pl.LazyFrame,
    ids: pl.Expr | Collection[any] | pl.Series,
    generations: int = 100,
    include_ids: bool = True,
    pedigree_labels: tuple[str, str, str] = PedigreeLabels,
) -> pl.DataFrame | pl.LazyFrame:
    """Return the descendants of the animals specified"""
    return _get_relatives_of(
        pedigree,
        ids,
        get_parents_of,
        generations=generations,
        include_ids=include_ids,
        pedigree_labels=pedigree_labels,
    )


def classify_generations(
    pedigree: pl.DataFrame | pl.LazyFrame,
    pedigree_labels: tuple[str, str, str] = PedigreeLabels,
) -> pl.DataFrame:
    """Add column classifying the animals into generations within the pedigree

    Journal of Animal and Veterinary Advances
    Year: 2009 | Volume: 8 | Issue: 1 | Page No.: 177-182
    An Algorithm to Sort Complex Pedigrees Chronologically without Birthdates
    Zhiwu Zhang , Changxi Li , Rory J. Todhunter , George Lust , Laksiri Goonewardene and Zhiquan Wang"""
    animal, sire, dam = pedigree_labels
    g = 0
    parents_df = pedigree = (
        pedigree.with_columns(pl.lit(g).alias("generation")).lazy().collect()
    )
    while parents_df.height > 0:
        pedigree = pedigree.with_columns(
            pl.when(pl.col(animal).is_in(parents_df.select(parents((sire, dam)))))
            .then(pl.col("generation") + 1)
            .otherwise("generation")
        )
        parents_df = pedigree.filter(pl.col("generation") > g)
        g += 1

    return pedigree.with_columns(
        (pl.col("generation").max() - pl.col("generation")).alias("generation")
    )  # reverse generation order (0 is earliest/oldest)
