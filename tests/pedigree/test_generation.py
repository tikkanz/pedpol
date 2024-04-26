from pathlib import Path

import polars as pl
from pedigree.generations import classify_generations, null_unknown_parents

data_dir = Path("/home/rishe0/dev/pedigree/tests/resources")

ped = pl.read_csv(
    data_dir / "ped_jv.csv", dtypes=3 * [pl.Int32], comment_prefix="#"
).pipe(
    null_unknown_parents,
)

pedlit = pl.read_csv(
    data_dir / "ped_literal.csv", dtypes=3 * [pl.Utf8], comment_prefix="#"
).pipe(null_unknown_parents, parent_labels=("Father", "Mother"))

ped = ped.pipe(classify_generations, pedigree_labels=("progeny", "sire", "dam"))
pedlit = pedlit.pipe(
    classify_generations, pedigree_labels=("Child", "Father", "Mother")
)

print(ped)
print(pedlit)
