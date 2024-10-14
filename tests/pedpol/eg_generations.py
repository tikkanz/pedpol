from pathlib import Path

import polars as pl

from pedpol.core import null_unknown_parents
from pedpol.generations import (
    classify_generations,
    get_ancestors_of,
    get_descendants_of,
)

pl.Config.set_tbl_rows(15)
data_dir = Path("/home/rishe0/dev/pedigree/tests/resources")

ped_jv = pl.read_csv(
    data_dir / "ped_jv.csv", schema_overrides=3 * [pl.Int32], comment_prefix="#"
).pipe(
    null_unknown_parents,
)

ped_lit = pl.read_csv(
    data_dir / "ped_literal.csv", schema_overrides=3 * [pl.Utf8], comment_prefix="#"
).pipe(null_unknown_parents, parent_labels=("Father", "Mother"))

ids = [3, 4]
descendants = get_descendants_of(
    ped_jv,
    ids,
    generations=6,
    include_ids=False,
    pedigree_labels=("progeny", "sire", "dam"),
)
print("Descendants of 3 & 4 (not including ids)", "\n", descendants)

ids = ["Kristi"]
ancestors = get_ancestors_of(
    ped_lit.lazy(),
    ids,
    generations=2,
    include_ids=True,
    pedigree_labels=("Child", "Father", "Mother"),
)
print("Ancestors of Kristi (including ids)", "\n", ancestors.collect())

ped_jv = ped_jv.pipe(classify_generations, pedigree_labels=("progeny", "sire", "dam"))
print("Classified ped_jv", "\n", ped_jv)

ped_lit = ped_lit.pipe(
    classify_generations, pedigree_labels=("Child", "Father", "Mother")
)
print("Classified ped_lit", "\n", ped_lit)
