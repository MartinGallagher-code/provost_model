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
   repository: `<shared_tools clone>/scripts/ticket.py new provost_model
   "Imperative title" --request "the request, in the user's own words"`.

The board: https://claude.ai/artifact/HrB3RpVpqGGQS7UZgZsMJm
