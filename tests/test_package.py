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

    def test_tensor_shapes_match_the_config(self):
        """Every tensor's shape and dtype is what config.json says provost's
        runtime will reshape it to; a name alone can hide a wrong export."""
        try:
            import numpy as np
        except ImportError:
            self.skipTest("needs numpy")
        c = self.cfg
        h, kv, inter = c["hidden"], c["kv_heads"] * c["head_dim"], c["intermediate"]
        self.assertEqual(c["heads"] * c["head_dim"], h)
        shapes = {"embed.q": ((c["vocab"], h), np.int8), "embed.s": ((c["vocab"],), None),
                  "norm": ((h,), np.float32)}
        out_in = {"q": (h, h), "k": (kv, h), "v": (kv, h), "o": (h, h),
                  "gate": (inter, h), "up": (inter, h), "down": (h, inter)}
        for i in range(c["layers"]):
            shapes[f"l{i}.in_norm"] = shapes[f"l{i}.post_norm"] = ((h,), np.float32)
            for k, oi in out_in.items():
                shapes[f"l{i}.{k}.q"] = (oi, np.int8)
                shapes[f"l{i}.{k}.s"] = ((oi[0],), None)
        got = {}
        for shard in c["shards"]:
            with np.load(os.path.join(provost_model.MODEL_DIR, shard), allow_pickle=False) as z:
                for k in z.files:
                    got[k] = (z[k].shape, z[k].dtype)
        for k, (shape, dtype) in shapes.items():
            self.assertEqual(got[k][0], shape, k)
            if dtype is not None:
                self.assertEqual(got[k][1], dtype, k)
            else:
                self.assertTrue(np.issubdtype(got[k][1], np.floating), k)
        # the two heads: hidden -> width -> (1 start logit | one per role)
        for head, n in (("start", 1), ("role", len(c["roles"]))):
            width = got[f"{head}.w0"][0][0]
            self.assertEqual(got[f"{head}.w0"][0], (width, h), head)
            self.assertEqual(got[f"{head}.b0"][0], (width,), head)
            self.assertEqual(got[f"{head}.w1"][0], (n, width), head)
            self.assertEqual(got[f"{head}.b1"][0], (n,), head)


if __name__ == "__main__":
    unittest.main()
