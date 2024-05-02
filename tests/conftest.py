from io import StringIO
from pathlib import Path

import polars as pl
import pytest
from pedigree.core import null_unknown_parents

data_dir = Path("/home/rishe0/dev/pedigree/tests/resources")


@pytest.fixture
def ped_basic():
    return pl.read_csv(
        data_dir / "ped_basic.csv", dtypes=3 * [pl.Int32], comment_prefix="#"
    ).pipe(null_unknown_parents, parent_labels=("Sire", "Dam"))


@pytest.fixture
def ped_jv():
    """pedigree from Zhang et. al. 2009"""
    return pl.read_csv(
        data_dir / "ped_jv.csv", dtypes=3 * [pl.Int32], comment_prefix="#"
    ).pipe(
        null_unknown_parents,
    )


@pytest.fixture
def ped_circular():
    """cannot be correctly sorted
    Test for circular pedigree by sorting then check if sorted?"""
    return pl.read_csv(
        data_dir / "ped_circular.csv", dtypes=3 * [pl.Int32], comment_prefix="#"
    ).pipe(
        null_unknown_parents,
    )


@pytest.fixture
def ped_errors():
    """Pedigree with errors

    Errors in pedigree include:
     * Animals that are their own parent,
     * multiple pedigree records for an animal,
     * multisex parent,
     * parents with no own record"""
    return pl.read_csv(
        data_dir / "ped_errors.csv", dtypes=3 * [pl.Int32], comment_prefix="#"
    ).pipe(null_unknown_parents, parent_labels=("sire", "dam"))


@pytest.fixture
def ped_lit():
    return pl.read_csv(
        data_dir / "ped_literal.csv", dtypes=3 * [pl.Utf8], comment_prefix="#"
    ).pipe(null_unknown_parents, parent_labels=("Father", "Mother"))


ped_literal_str = StringIO("""Child,Father,Mother
Harry,George,Daisey
Gertrude,Jim,Jessica
Nader,Harry,Gloria
Karen,Harry,Michelle
Steve,Harry,Fatma
Frances,Harry,.
Hein,Tom,Gertrude
Emily,Tom,Susan
Barry,Hein,Karen
Scott,Hein,Karen
Kristi,Hein,Karen
Helen,Hein,Emily
""")
