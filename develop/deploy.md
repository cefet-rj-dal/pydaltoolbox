# Publishing `pydaltoolbox` to PyPI — a from-zero guide

This guide assumes you have **no accounts anywhere** and have never published a Python
package. It walks every step. Think of it as the Python equivalent of submitting to
CRAN, with two big differences: there is **no human review** (your upload is live
immediately) and **a version number can never be reused** (once `0.1.0` is uploaded,
the next upload must be `0.1.1`, `0.2.0`, etc.).

Two websites are involved:

- **PyPI** (<https://pypi.org>) — the real, public index. `pip install pydaltoolbox`
  downloads from here.
- **TestPyPI** (<https://test.pypi.org>) — a throwaway *sandbox* that works exactly the
  same way, for rehearsing. It has **separate accounts** from PyPI.

Do the one-time setup once; then Part 1–3 is what you repeat for every release.

---

## Part 0 — One-time setup (~15 minutes)

### 0.1 Create a PyPI account

1. Go to <https://pypi.org/account/register/>.
2. Fill in name, email, username, password. Submit.
3. Open the confirmation email from PyPI and click the verification link.

### 0.2 Turn on two-factor authentication (2FA) — required

PyPI will not let you upload without 2FA.

1. Install an authenticator app on your phone if you don't have one (Google
   Authenticator, Microsoft Authenticator, or Authy all work).
2. On PyPI: top-right menu → **Account settings** → **Two factor authentication** →
   **Add 2FA with authenticator application**.
3. Scan the QR code with the app; type the 6-digit code it shows to confirm.
4. **Save the recovery codes** it gives you somewhere safe (a password manager). They
   are your way back in if you lose the phone.

### 0.3 Create an API token (this is how you log in when uploading)

You never type your website password in the terminal; you use a token instead.

1. On PyPI: **Account settings** → **API tokens** → **Add API token**.
2. Token name: anything, e.g. `pydaltoolbox-upload`.
3. Scope: the first time, choose **"Entire account"** (after the project exists you can
   make a project-scoped token, which is safer — see §4).
4. Click **Create token**, then **copy the token**. It starts with `pypi-` and is shown
   **only once** — paste it into your password manager now.

### 0.4 Repeat 0.1–0.3 on TestPyPI

TestPyPI is a different site with its own account and its own token:

- Register: <https://test.pypi.org/account/register/>
- Enable 2FA the same way.
- Create an API token there too, and save it (this token is different from the PyPI one).

### 0.5 Install the build/upload tools (once, on your machine)

In a terminal:

```bash
python -m pip install --upgrade build twine
```

- `build` turns the project into the two distributable files.
- `twine` uploads them and checks them.

---

## Part 1 — Build the package

From the project root (`c:/Git/dal/pydaltoolbox`):

```bash
# clean any previous build output
#   Git Bash:    rm -rf dist build src/*.egg-info
#   PowerShell:  Remove-Item -Recurse -Force dist, build, src/*.egg-info -ErrorAction SilentlyContinue

python -m build
```

This creates a `dist/` folder with two files:

- `pydaltoolbox-0.1.0.tar.gz` — the *source distribution* (sdist).
- `pydaltoolbox-0.1.0-py3-none-any.whl` — the *wheel* (what `pip` installs).

Then validate the metadata (like a mini `R CMD check` for the package description):

```bash
python -m twine check dist/*
```

Both files should report `PASSED`.

---

## Part 2 — Rehearse on TestPyPI (safe, recommended the first time)

Upload to the sandbox:

```bash
python -m twine upload --repository testpypi dist/*
```

When prompted:

- **username:** type exactly `__token__` (with the double underscores)
- **password:** paste your **TestPyPI** token (the `pypi-...` string)

If it succeeds, check the page it prints, e.g.
`https://test.pypi.org/project/pydaltoolbox/`.

Verify it actually installs, in a throwaway virtual environment:

```bash
python -m venv /tmp/try && . /tmp/try/Scripts/activate   # Windows Git Bash
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple pydaltoolbox
python -c "import pydaltoolbox; print(pydaltoolbox.__version__)"
deactivate
```

(The `--extra-index-url` is needed because TestPyPI does not host numpy/pandas/sklearn,
so pip fetches those dependencies from the real PyPI.)

---

## Part 3 — Publish for real on PyPI

When the rehearsal looked good, upload to the real index:

```bash
python -m twine upload dist/*
```

- **username:** `__token__`
- **password:** your **PyPI** token (the real one, not the TestPyPI one)

Done. Within a minute `pip install pydaltoolbox` works for everyone, and the project
page is `https://pypi.org/project/pydaltoolbox/`.

---

## Part 4 — Good habits and gotchas

- **Name must be free.** The name `pydaltoolbox` is claimed by whoever publishes it
  first. If someone already took it, you'd pick another name in `pyproject.toml`.
- **Versions are immutable.** You cannot re-upload `0.1.0` — not even after deleting it.
  To ship a fix, bump the `version` in `pyproject.toml` (e.g. `0.1.1`) and rebuild.
- **Keep the token secret.** It is as powerful as your password. After the first
  publish, create a **project-scoped** token (PyPI → project → Settings → ... or
  Account → API tokens, scope = `pydaltoolbox`) and delete the account-wide one.
- **Avoid retyping the token** by storing credentials in `~/.pypirc`:

  ```ini
  [pypi]
    username = __token__
    password = pypi-REAL_TOKEN_HERE

  [testpypi]
    username = __token__
    password = pypi-TESTPYPI_TOKEN_HERE
  ```

  Then `twine upload dist/*` won't prompt. Protect this file (it holds a secret).

- **Later, automate it.** Once comfortable, GitHub Actions can publish automatically on
  each version tag using *Trusted Publishing* (no tokens stored at all). Ask when you
  want to set that up.

---

## Quick reference (every release)

```bash
# 1. bump version in pyproject.toml, e.g. 0.1.0 -> 0.1.1
# 2. quality gate
python -m pytest && python -m ruff check .
# 3. build + check
rm -rf dist && python -m build && python -m twine check dist/*
# 4. publish
python -m twine upload dist/*
```

## CRAN → PyPI cheat sheet

| CRAN (R) | PyPI (Python) |
|---|---|
| `DESCRIPTION` | `pyproject.toml` |
| `R CMD build` | `python -m build` |
| `R CMD check` | `pytest` + `ruff` + `twine check` |
| win-builder / R-hub test | upload to TestPyPI |
| `devtools::release()` → CRAN | `twine upload` → PyPI |
| human review, ~days | none — instant, public |
| resubmit same version | impossible — bump the version |
| `install.packages()` | `pip install` |
