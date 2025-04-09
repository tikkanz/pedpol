from io import StringIO
from pathlib import Path

import polars as pl
import pytest

from pedpol.core import null_unknown_parents
from pedpol.generations import classify_generations
from pedpol.validation import add_missing_records

data_dir = Path(__file__).absolute().parent / "resources"


@pytest.fixture
def ped_basic():
    return pl.read_csv(
        data_dir / "ped_basic.csv", schema_overrides=3 * [pl.Int32], comment_prefix="#"
    ).pipe(null_unknown_parents, parent_labels=("Sire", "Dam"))


@pytest.fixture
def ped_jv():
    """pedigree from Zhang et. al. 2009"""
    ped = pl.read_csv(
        data_dir / "ped_jv.csv", schema_overrides=3 * [pl.Int32], comment_prefix="#"
    ).pipe(
        null_unknown_parents,
    )
    return ped, (ped.columns)


@pytest.fixture
def ped_circular():
    """cannot be correctly sorted
    Test for circular pedigree by sorting then check if sorted?"""
    ped = pl.read_csv(
        data_dir / "ped_circular.csv",
        schema_overrides=3 * [pl.Int32],
        comment_prefix="#",
    ).pipe(
        null_unknown_parents,
    )
    return ped, (ped.columns)


@pytest.fixture
def ped_errors():
    """Pedigree with errors

    Errors in pedigree include:
     * Animals that are their own parent,
     * multiple pedigree records for an animal,
     * multisex parent,
     * parents with no own record"""
    ped = pl.read_csv(
        data_dir / "ped_errors.csv", schema_overrides=3 * [pl.Int32], comment_prefix="#"
    ).pipe(null_unknown_parents, parent_labels=("sire", "dam"))
    return ped, (ped.columns)


@pytest.fixture
def ped_lit():
    ped = pl.read_csv(
        data_dir / "ped_literal.csv",
        schema_overrides=3 * [pl.Utf8],
        comment_prefix="#",
    ).pipe(null_unknown_parents, parent_labels=("Father", "Mother"))
    return ped, (ped.columns)


@pytest.fixture
def ped_lit_valid():
    ped = (
        pl.read_csv(
            data_dir / "ped_literal.csv",
            schema_overrides=3 * [pl.Utf8],
            comment_prefix="#",
        )
        .pipe(null_unknown_parents, parent_labels=("Father", "Mother"))
        .pipe(add_missing_records, pedigree_labels=("Child", "Father", "Mother"))
    )

    return ped, (ped.columns)


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


@pytest.fixture
def ped_jv_classified(ped_jv):
    ped, lbls = ped_jv
    return classify_generations(ped, lbls), lbls


@pytest.fixture
def ped_circular_classified(ped_circular):
    ped, lbls = ped_circular
    return classify_generations(ped, lbls), lbls
