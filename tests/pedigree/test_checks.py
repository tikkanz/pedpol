from pathlib import Path

import polars as pl
from pedigree.checks import (
    add_missing_records,
    get_animals_are_own_parent,
    get_animals_with_multiple_records,
    get_parents_both_sires_and_dams,
    get_parents_without_own_record,
    recode_pedigree,
)
from pedigree.core import null_unknown_parents

pl.Config.set_tbl_rows(15)
data_dir = Path("/home/rishe0/dev/pedigree/tests/resources")

ped_jv = pl.read_csv(
    data_dir / "ped_jv.csv", dtypes=3 * [pl.Int32], comment_prefix="#"
).pipe(
    null_unknown_parents,
)

ped_lit = pl.read_csv(
    data_dir / "ped_literal.csv", dtypes=3 * [pl.Utf8], comment_prefix="#"
).pipe(null_unknown_parents, parent_labels=("Father", "Mother"))

ped_err = pl.read_csv(
    data_dir / "ped_errors.csv", dtypes=3 * [pl.Int32], comment_prefix="#"
).pipe(null_unknown_parents, parent_labels=("sire", "dam"))


print(
    "Is it's own parent",
    "\n",
    ped_err.pipe(get_animals_are_own_parent, ("anim", "sire", "dam")),
)
print(
    "Has multiple records",
    "\n",
    ped_err.pipe(get_animals_with_multiple_records, ("anim", "sire", "dam")),
)
print(
    "Parent is both sire and dam",
    "\n",
    ped_err.pipe(get_parents_both_sires_and_dams),
)
print(
    "Parents without their own record",
    "\n",
    ped_err.pipe(get_parents_without_own_record, ("anim", "sire", "dam")),
)
print(
    "Added record for any parents without their own record",
    "\n",
    ped_err.pipe(add_missing_records, ("anim", "sire", "dam")),
)

tmp = ped_lit.pipe(add_missing_records, ("Child", "Father", "Mother"))
recoded_ped, id_map = recode_pedigree(tmp, ("Child", "Father", "Mother"))
print(
    "Pedigree with recoded Ids", "\n", recoded_ped, "\n", "Map to old Ids", "\n", id_map
)
