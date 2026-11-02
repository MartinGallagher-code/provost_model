# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Martin J. Gallagher
"""The installed package carries a complete model directory.

    python -m unittest discover -s tests -v

Run it against an installed wheel, not the source tree, to check what pip
ships (CI does).
"""
import json
import os
import unittest

import provost_model

LAYER_KEYS = ("in_norm", "post_norm") + tuple(
    f"{k}.{p}" for k in ("q", "k", "v", "o", "gate", "up", "down") for p in ("q", "s"))


class Package(unittest.TestCase):
    def setUp(self):
        with open(os.path.join(provost_model.MODEL_DIR, "config.json"), encoding="utf-8") as fh:
            self.cfg = json.load(fh)

    def test_config(self):
        self.assertEqual(self.cfg["format"], "provost-lm-numpy/1")
        self.assertEqual(self.cfg["roles"], ["header", "row", "cont", "kv", "section", "skip"])
        self.assertEqual(self.cfg["layers"], 18)

    def test_every_file_is_there(self):
        for f in ["tokenizer.json"] + self.cfg["shards"]:
            self.assertTrue(os.path.isfile(os.path.join(provost_model.MODEL_DIR, f)), f)

    def test_tokenizer_is_byte_level_bpe(self):
        with open(os.path.join(provost_model.MODEL_DIR, "tokenizer.json"), encoding="utf-8") as fh:
            tok = json.load(fh)
        self.assertEqual(tok["model"]["type"], "BPE")
        self.assertEqual(tok["pre_tokenizer"]["type"], "ByteLevel")
        self.assertEqual(len(tok["model"]["vocab"]), self.cfg["vocab"])

    def test_weights_hold_every_tensor(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("needs numpy")
        names = set()
        for shard in self.cfg["shards"]:
            with np.load(os.path.join(provost_model.MODEL_DIR, shard), allow_pickle=False) as z:
                names.update(z.files)
        want = {"embed.q", "embed.s", "norm"} | {f"{h}.{p}" for h in ("start", "role")
                                                  for p in ("w0", "b0", "w1", "b1")}
        want |= {f"l{i}.{k}" for i in range(self.cfg["layers"]) for k in LAYER_KEYS}
        self.assertEqual(names, want)


if __name__ == "__main__":
    unittest.main()
