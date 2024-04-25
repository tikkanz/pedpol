from pathlib import Path

import polars as pl
from pedigree.generations import classify_generations

data_dir = Path("/home/rishe0/dev/pedigree/tests/resources")

ped = pl.read_csv(data_dir / "ped_jv.csv", comment_prefix="#")

ped = ped.pipe(classify_generations, pedigree_labels=("progeny", "sire", "dam"))

print(ped)
