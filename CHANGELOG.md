# provost-model — Changelog

## 0.2.1 — 2026-11-10

- **Every change moves the version** (shared_tools BN-002): a pull request
  into `main` now carries its own version bump and its own numbered, dated
  section here, never an "Unreleased" heading. `tests/check_version.sh`
  checks that `__version__` and this file's first section agree, and the
  new `version` job in CI fails a pull request whose version is not greater
  than its base branch's, or a push to `main` that does not move it.
  `CLAUDE.md` and the README's "Releasing" say so. Nothing in the package
  changes: the model is 0.2.0's.

## 0.2.0 — 2026-11-04

- **The model is retrained on every capture** (`--no-val`, PM-001): the same
  `lm-all-l18` recipe (SmolLM2-135M, LoRA rank 16 and two heads, 2 epochs, 18
  layers), now on all 3,674 corpus captures. 0.1.0 held 403 back to pick the
  epoch and never saw them. Trained on the owner's iMac (M4 GPU, 16 min).
- **Faithfulness of the export:** on 200 corpus captures the 8-bit model
  agrees with torch on 1,460 of 1,462 roles and 8,135 of 8,146 starts
  (largest probability gap 0.074).
- **Round 4** (107 captures, model-first): 67 outputs differ from 0.1.0's.
  Rated blind in pairs on round 4's `CRITERIA.md`, 52 rate the same, 9 better
  and 6 worse, so the total moves from 5 / 30 / 72 to 4 / 35 / 68: four fewer
  FAILs, one PASS fewer (`103-apcupsd-apcaccess-status` drops `FIRMWARE`).
  It files 90 captures against 93.
- **CI runs the model with provost** before every release: provost from PyPI
  is installed beside the wheel, loads it with its own numpy runtime and
  labels a short capture (`tests/smoke_provost.py`). provost falls back to its
  decision trees without a word when a model will not load, so nothing else
  would have noticed a broken export.
- **`tests/test_package.py` checks every tensor's shape and dtype** against
  `config.json`, not only its name.

## 0.1.0 — 2026-11-02

The first release: the tiny-LM layout labeller provost's `--model` uses when
this package is installed.

- **The model:** SmolLM2-135M fine-tuned on Claude's line-layout labels of
  provost's 3,674-capture corpus (LoRA rank 16 and two heads, 2 epochs), cut
  to its first 18 of 30 layers. It was trained on every fold, holding back 403
  captures to pick the epoch. In provost it is `lm-all-l18`.
- **Storage:** the adapter is merged into the base weights. Every linear layer
  and the vocabulary table are stored at 8 bits, one scale per row, in three
  `.npz` shards of 95 MB in all. The wheel is 85 MB.
- **Faithfulness:** on provost's round-4 benchmark it rates exactly as the
  float model does (5 / 30 / 72); on fold 0 the 30-layer model exported this
  way scores 0.863, as in float.
- **Built by** provost's `ml/export_lm.py`. provost's `extract/model/lm_numpy.py`
  runs it with numpy alone.
- **Released by** `.github/workflows/release.yml` (PyPI Trusted Publishing, no
  API token) when a GitHub release is published.
