# PedPol
[![Styling, Lint & Test](https://github.com/tikkanz/pedpol/actions/workflows/test_and_lint.yml/badge.svg?event=push)](https://github.com/tikkanz/pedpol/actions/workflows/test_and_lint.yml)

PedPol is a python package for wrangling animal pedigrees
 * Handles large pedigrees (tested for pedigrees with more than 30 million records)
 * Flexible naming/type conventions for individual & parent Ids
 * Pedigree can contain any other columns as desired
 * Comprehensive testing of pedigree validity
 * Tools to create valid pedigrees (null parents without their own record)
 * Filtering based on relationships (parents, progeny, ancestors, descendants)
 * Classify records by generations without birth date/year
 * Recoding of pedigree Ids
 * Uses Polars DataFrames to read, write, store and manipulate pedigrees
 