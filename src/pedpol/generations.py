from typing import Collection

import polars as pl

from pedpol.core import PedigreeLabels, parents


def get_progeny_of(
    pedigree: pl.DataFrame | pl.LazyFrame,
    ids: pl.Expr | Collection[any] | pl.Series,
    pedigree_labels: tuple[str, str, str] = PedigreeLabels,
) -> pl.LazyFrame:
    """Return records for the progeny of the animals specified"""
    ids = pl.Series(values=ids, dtype=pedigree.collect_schema()[pedigree_labels[0]])
    return pedigree.lazy().filter(
        pl.any_horizontal(pl.col(pedigree_labels[1:]).is_in(ids))
    )


def get_parents_of(
    pedigree: pl.DataFrame | pl.LazyFrame,
    ids: pl.Expr | Collection[any] | pl.Series,
    pedigree_labels: tuple[str, str, str] = PedigreeLabels,
) -> pl.LazyFrame:
    """Return records for the parents of the animals specified"""
    animal, sire, dam = pedigree_labels
    if isinstance(ids, list):
        ids = pl.LazyFrame(
            {animal: ids}, schema=pedigree.select(animal).collect_schema()
        )
    anims = pedigree.lazy().join(ids.lazy(), on=animal)
    prnts = anims.select(parents((sire, dam)))
    return pedigree.lazy().join(prnts, left_on=animal, right_on="parent")


def _get_relatives_of(
    pedigree: pl.DataFrame | pl.LazyFrame,
    ids: pl.Expr | Collection[any] | pl.Series,
    relatives_function: callable,
    generations: int = 100,
    include_ids: bool = True,
    pedigree_labels: tuple[str, str, str] = PedigreeLabels,
) -> pl.DataFrame | pl.LazyFrame:
    """General utility for iterating through pedigree to find relatives"""
    animal = pedigree_labels[0]
    ids = pl.DataFrame({animal: ids}, schema=pedigree.select(animal).collect_schema())
    g = 0
    ids_g = ids
    relatives = []
    while generations > g and ids_g.height != 0:
        ids_g = (
            relatives_function(pedigree, ids_g, pedigree_labels)
            .select(animal)
            .collect()
        )
        relatives.append(ids_g)
        g += 1

    if include_ids:
        relatives.append(ids)
    ids_relatives = pl.concat(relatives).unique()
    if isinstance(pedigree, pl.LazyFrame):
        ids_relatives = ids_relatives.lazy()
    return pedigree.join(ids_relatives, on=animal)


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
    parent_count = parents_df.height + 1
    while parents_df.height < parent_count:
        parent_count = parents_df.height
        pedigree = pedigree.with_columns(
            pl.when(
                pl.col(animal).is_in(
                    parents_df.select(parents((sire, dam))).to_series()
                )
            )
            .then(pl.col("generation") + 1)
            .otherwise("generation")
        )
        parents_df = pedigree.filter(pl.col("generation") > g)
        g += 1

    return pedigree.with_columns(
        (pl.col("generation").max() - pl.col("generation")).alias("generation")
    )  # reverse generation order (0 is earliest/oldest)
