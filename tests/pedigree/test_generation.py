from pathlib import Path

import polars as pl
from pedigree.core import null_unknown_parents
from pedigree.generations import classify_generations, get_descendents_of

data_dir = Path("/home/rishe0/dev/pedigree/tests/resources")

pedjv = pl.read_csv(
    data_dir / "ped_jv.csv", dtypes=3 * [pl.Int32], comment_prefix="#"
).pipe(
    null_unknown_parents,
)

pedlit = pl.read_csv(
    data_dir / "ped_literal.csv", dtypes=3 * [pl.Utf8], comment_prefix="#"
).pipe(null_unknown_parents, parent_labels=("Father", "Mother"))

pedjv = pedjv.pipe(classify_generations, pedigree_labels=("progeny", "sire", "dam"))
pedlit = pedlit.pipe(
    classify_generations, pedigree_labels=("Child", "Father", "Mother")
)

ids = [3, 4]
get_descendents_of(
    pedjv,
    ids,
    generations=6,
    include_ids=True,
    pedigree_labels=("progeny", "sire", "dam"),
)

print(pedjv)
print(pedlit)
