# Publishing `pydaltoolbox` (PyPI)

This guide explains how to publish a Python package, written for someone who
already knows the R/CRAN workflow. The short version: **CRAN → PyPI**, and the
tooling is different but the stages map almost one to one.

## The mental map (R → Python)

| R / CRAN step | Python / PyPI equivalent |
|---|---|
| `DESCRIPTION` | `pyproject.toml` (`[project]` table) |
| `NAMESPACE` (exports) | `__all__` in `__init__.py` (no separate file) |
| `Imports:` / `Depends:` | `dependencies = [...]` in `pyproject.toml` |
| `Suggests:` | `[project.optional-dependencies]` (extras) |
| `R CMD build` → `.tar.gz` | `python -m build` → `sdist` (`.tar.gz`) + `wheel` (`.whl`) |
| `R CMD check` | `pytest` + `ruff` + `twine check` |
| `devtools::check()` | same combo, usually run in CI |
| Test on win-builder / R-hub | upload to **TestPyPI** |
| `devtools::release()` → CRAN | `twine upload` → **PyPI** |
| CRAN manual review by humans | **no human review** — upload is instant and public |
| `install.packages("pkg")` | `pip install pkg` |

Two differences worth internalizing:

1. **No gatekeeper.** PyPI does not review submissions. Whatever you upload is live
   immediately. So the discipline (tests, lint, a TestPyPI dry run) is on *you*, not
   on a reviewer.
2. **Versions are immutable.** Once `0.1.0` is uploaded you can never re-upload
   `0.1.0`, even if you delete it. Bump the version for every upload.

## One-time setup

Install the build/publish tools (already in our `[dev]` extra):

```bash
python -m pip install --upgrade build twine
```

Create accounts:

- PyPI: <https://pypi.org/account/register/>
- TestPyPI (a throwaway mirror for rehearsals): <https://test.pypi.org/account/register/>

Enable 2FA on both (required). For uploading, modern PyPI uses **API tokens**
instead of passwords (and, in CI, **Trusted Publishing** — see below).

## Release checklist (manual)

```bash
# 1. Bump the version in pyproject.toml  (e.g. 0.1.0 -> 0.1.1)

# 2. Run the quality gate (this is our "R CMD check")
python -m pytest
python -m ruff check .

# 3. Clean old artifacts and build sdist + wheel into dist/
rm -rf dist build *.egg-info
python -m build

# 4. Validate the built artifacts' metadata
python -m twine check dist/*

# 5. Rehearse on TestPyPI
python -m twine upload --repository testpypi dist/*
#    then, in a fresh virtualenv, verify it installs and imports:
pip install -i https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple pydaltoolbox
python -c "import pydaltoolbox; print(pydaltoolbox.__version__)"

# 6. Publish for real
python -m twine upload dist/*
```

When `twine` prompts for credentials, use `__token__` as the username and the API
token (starting with `pypi-`) as the password. Or put tokens in `~/.pypirc`.

## Recommended: automate with GitHub Actions + Trusted Publishing

This removes tokens entirely: you authorize the GitHub repo once on PyPI, and
pushing a tag publishes the release. This is the modern equivalent of a CRAN
submission pipeline.

1. On PyPI: *Your project → Publishing → Add a trusted publisher* → GitHub,
   fill in owner (`cefet-rj-dal`), repo (`pydaltoolbox`), and workflow filename
   (`release.yml`).
2. Add `.github/workflows/release.yml`:

```yaml
name: release
on:
  push:
    tags: ["v*"]
jobs:
  build-and-publish:
    runs-on: ubuntu-latest
    permissions:
      id-token: write          # required for Trusted Publishing
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install build
      - run: python -m build
      - uses: pypa/gh-action-pypi-publish@release/v1
```

Then a release is just:

```bash
git tag v0.1.1 && git push origin v0.1.1
```

## Versioning convention

Use [Semantic Versioning](https://semver.org/) (`MAJOR.MINOR.PATCH`). While the
port is incomplete, stay on `0.x` — that signals "API may still move", which is
accurate given we are building the Experiment Line family by family.
