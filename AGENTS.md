# Repository Guidelines

## Layout

- `irispy/` is the import package; the distribution name is `irispy-lmsal`.
- Main public data APIs live in `irispy/sji.py` and `irispy/spectrograph.py`.
- Level 2 FITS readers live in `irispy/io/`; analysis and calibration helpers live in `irispy/utils/`.
- Packaged calibration and test data live in `irispy/data/`.
- Tests stay beside their code in `irispy/tests/`, `irispy/io/tests/`, and `irispy/utils/tests/`.
- Documentation is in `docs/`, runnable examples are in `examples/`, and Towncrier fragments are in `changelog/`.
- Do not hand-edit generated files such as `irispy/_version.py`, `docs/api/`, or `docs/generated/`.

## Development Commands

The project supports Python 3.12-3.14; Python 3.14 matches the primary CI job.

```bash
micromamba create -n irispy python=3.14
micromamba activate irispy
python -m pip install -e ".[dev]"
```

Run the full offline suite, including documentation doctests:

```bash
pytest
```

Run network and figure tests explicitly:

```bash
pytest --remote-data=any
pytest --pyargs irispy -m "mpl_image_compare" --mpl --remote-data=any
```

CI-style checks are `tox -e py314`, `tox -e codestyle`, and `tox -e build_docs`. To remove generated documentation,
run `make clean` from `docs/`.

## Style and Tests

- Target Python 3.12 syntax. Ruff formats with 120-column lines, double quotes, spaces, and NumPy docstrings.
- Let the `isort` pre-commit hook order imports; Ruff's import-sorting rules are disabled.
- Run `pre-commit run --all-files` before opening a PR.
- Warnings are errors. The default suite excludes `mpl_image_compare` tests and enables RST doctests.
- Mark data downloads with `@pytest.mark.remote_data` and other connectivity-dependent tests with
  `@pytest.mark.online`.
- Add the smallest focused regression test near changed code. Use the existing `figure_test` helper for visual output.

## Commits and Pull Requests

- Describe user-visible impact in the PR and link the relevant issue or PR when one exists.
- For user-visible changes, add a ReST Towncrier fragment named `changelog/<PR_NUMBER>.<TYPE>.rst`.
- Valid fragment types are `breaking`, `deprecation`, `removal`, `feature`, `bugfix`, `doc`, and `trivial`.
- Add a counter for multiple fragments of the same type, for example `123.feature.1.rst`.
