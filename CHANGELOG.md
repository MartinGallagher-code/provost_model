# provost-model — Changelog

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
