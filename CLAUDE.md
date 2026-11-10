# Project Instructions — provost_model

`provost-model` is the tiny-LM layout labeller that provost's `--model` uses when
this package is installed: SmolLM2-135M fine-tuned on provost's labels, cut to 18
layers, stored at 8 bits and run with numpy. The model is built in
[provost](https://github.com/MartinGallagher-code/provost) (`ml/export_lm.py`);
this repository packages and releases it.

## Every task is a ticket

Everything done here goes through a ticket in `MartinGallagher-code/shared_tools`,
which keeps every project's tickets as files and moves each one Backlog → In Progress
→ Review → Done (Adversarial Test only when the owner asks). Before any change:

1. Attach `MartinGallagher-code/shared_tools` with `add_repo` and push access, and
   clone it. If a clone is already here, pull it instead
   (`git -C <shared_tools clone> pull --ff-only`): a stale clone has a stale
   `CLAUDE.md` and a stale `ticket.py`.
2. Read the "Every task is a ticket" section of its `CLAUDE.md`. It sets out what each
   column requires, and none of it is optional.
3. Open the ticket with the shared_tools clone's script, not a path in this
   repository:

   ~~~bash
   <shared_tools clone>/scripts/ticket.py new provost_model "Imperative title" --request "the request, in the user's own words"
   ~~~

**Every change moves the version**, however small: a pull request into `main` raises
it in every place it is written and opens `CHANGELOG.md` with that version's own
dated section, never an "Unreleased" heading left for later. CI fails a pull request
whose version is not greater than its base branch's. This file's own section on the
version says where it is written.

The board: https://claude.ai/artifact/HrB3RpVpqGGQS7UZgZsMJm

## Every change moves the version

The owner's rule (shared_tools BN-002): **whenever this project changes, its version
changes with it**, however small the change: code, tests, docs or CI. A pull request
into `main` carries its own bump and its own `CHANGELOG.md` section, never an
"Unreleased" heading left for later.

- Patch for fixes, docs, tests and CI; minor for anything new a user can use; major
  only when something that worked stops working the way it did.
- In the same pull request: `__version__` in `src/provost_model/__init__.py` (the only
  place; `pyproject.toml` reads it, as hatch's dynamic version), and a
  `## X.Y.Z — YYYY-MM-DD` section at the top of `CHANGELOG.md`.
  `tests/check_version.sh` fails if they disagree.
- The **version** job in CI fails a pull request whose version is not greater than its
  base branch's, and a push to `main` whose version is not greater than what `main` had
  before it. If `main` moves while a pull request is open, rebase and bump again.
- Publishing is a separate step, and a person's call: publishing a GitHub release
  tagged `vX.Y.Z` runs `.github/workflows/release.yml`, which uploads that version to
  PyPI (README, "Releasing"). A number that is never published is fine; a change with
  no number is not.
