# provost-model

The tiny language model behind [provost](https://github.com/MartinGallagher-code/provost)'s
`--model`, packaged on its own so provost itself stays small.

provost turns command output into tidy datasets. When its parsers can't
read a capture cleanly, `--model` asks a learned layout model which lines
are headers, rows, key/value pairs or sections, and where each field starts.
Every value is still cut from the input; the model only decides where.
provost ships small decision trees for this. This package carries a better
reader: a pretrained language model fine-tuned on provost's own labels.

## Install

```bash
pip install provost[lm]          # provost, numpy and this package
```

or, beside an existing provost:

```bash
pip install provost-model        # pulls in numpy
```

provost finds it on its own: with `provost-model` installed and
`PROVOST_LAYOUT_LM` unset, `provost --model` labels with this model instead
of the trees.
- `PROVOST_LAYOUT_LM=0` keeps the trees.
- `PROVOST_LAYOUT_LM=DIR` uses another model directory.
- provost runs the model under the Python it was installed with; when that
  differs, set `PROVOST_PYTHON` to the interpreter that has this package.

No torch, transformers or GPU is needed: provost runs the model with numpy
alone (`extract/model/lm_numpy.py`).

## What's in it

`provost_model.MODEL_DIR` is a directory holding:

| file | what it is |
|---|---|
| `config.json` | the shape of the model, its format (`provost-lm-numpy/1`) and where it came from |
| `tokenizer.json` | SmolLM2's byte-level BPE tokenizer, read by provost's `bpe.py` |
| `weights-1.npz` … `weights-3.npz` | the weights, 95 MB in three shards |

**The model** is [SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M)
(Apache-2.0):
- fine-tuned with a LoRA adapter and two small heads on Claude's labels of
  provost's 3,674-capture corpus;
- cut to its first 18 of 30 layers, with the adapter merged in;
- stored at 8 bits per weight, one scale per row.

provost's `ml/export_lm.py` builds this directory from a model
`ml/train_lm.py` trains.

## How good it is

**Per line,** five-fold cross-validation on held-out tool families. The
score is how often a line's field split matches the reference label (14,468
lines in all):

| reader | split | hand-labelled lines |
|---|---:|---:|
| provost's parsers | 0.814 | 189/287 |
| provost's shipped trees | 0.819 | 212/287 |
| **this model (18 layers)** | **0.862** | **246/287** |

**The 8-bit export matches the float model:**
- on fold 0, the 30-layer model scores 0.863 in float and 0.863 exported;
- on provost's round-4 benchmark (107 captures, rated blind), this model
  rates exactly as its float original does.

**What it costs to run:** about 280 MB of memory and a 0.5 s load. It takes
under a second per capture on 4 CPU cores, and longer for a capture of
hundreds of long lines.

**On whole captures** it is a second reader, not a replacement for
provost's parsers. That is why provost only asks it about a capture the
quality gate would otherwise hold.

Details: provost's `docs/lm-2026-10-29-tiny-labeller.md`.

## Releasing

`.github/workflows/release.yml` publishes to PyPI by Trusted Publishing, with
no API token, when a GitHub release is published or the workflow is run by
hand.
- **Set it up once on PyPI:** project `provost-model`, owner
  `MartinGallagher-code`, repository `provost_model`, workflow `release.yml`,
  environment `pypi`.
- **For each release:**
  1. bump `__version__` in `src/provost_model/__init__.py` and add a
     `CHANGELOG.md` entry;
  2. merge;
  3. publish a GitHub release tagged `vX.Y.Z`.
- **What CI checks first**, on every push and pull request and again in
  `release.yml` before anything is uploaded: the built wheel installs, ships
  every file with the shapes `config.json` gives (`tests/test_package.py`),
  and provost from PyPI loads and runs it (`tests/smoke_provost.py`). The
  second matters because provost hides a broken model: it falls back to the
  decision trees with no message, so a bad export would otherwise install
  cleanly and quietly turn `provost --model` off. To run both locally:

  ```bash
  python -m build
  python -m pip install dist/*.whl provost
  cd tests && python -m unittest discover -s . -v && python -m unittest -v smoke_provost
  ```

## Licence

Apache-2.0, as the base model is; see `LICENSE` and `NOTICE`. provost itself
is GPL-3.0-or-later.
