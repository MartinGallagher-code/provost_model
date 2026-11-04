# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Martin J. Gallagher
"""provost loads and runs the installed model, not only finds its files.

    pip install dist/*.whl provost
    cd tests && python -m unittest -v smoke_provost

Not named test_*.py on purpose: the plain ``unittest discover`` run needs
only this package and numpy, while this needs provost (from PyPI) installed
beside it. CI runs both. With provost missing this fails rather than skips,
since it is only ever run to check exactly that pairing.

provost hides a broken model: ``layout_model.load_lm()`` returns None on any
exception and the decision trees label instead, so a bad export would pass
everything else and quietly switch ``provost --model`` off. This loads the
model with provost's own numpy runtime directly, so the real error shows,
then checks load_lm() does find it, then labels a short capture.

provost's runtime lives inside its wheel at
``provost_data.bundle_dir()/extract/model``. That is provost's internal
layout rather than an API, so if it moves this fails loudly and says where
it looked.
"""
import json
import os
import sys
import unittest

import provost_model

# A short kubectl-style table: a header line and three rows, no blank lines.
CAPTURE = [
    "NAME                     READY   STATUS    RESTARTS   AGE",
    "web-7d4b9c8f6d-2xkqp     1/1     Running   0          3d",
    "web-7d4b9c8f6d-9zt4m     1/1     Running   2          3d",
    "db-0                     1/1     Running   0          12d",
]


def provost_model_dir():
    """provost's bundled extract/model directory, put on sys.path."""
    try:
        import provost_data
    except ImportError as e:
        raise AssertionError("provost is not installed (pip install provost): %s" % e)
    where = os.path.join(provost_data.bundle_dir(), "extract", "model")
    for f in ("lm_numpy.py", "layout_model.py"):
        if not os.path.isfile(os.path.join(where, f)):
            raise AssertionError(
                "provost's model runtime is not where this test looks: no %s in %s "
                "(provost %s moved it?)" % (f, where, getattr(provost_data, "__version__", "?")))
    if where not in sys.path:
        sys.path.insert(0, where)
    return where


class ProvostRunsTheModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        provost_model_dir()
        with open(os.path.join(provost_model.MODEL_DIR, "config.json"), encoding="utf-8") as fh:
            cls.cfg = json.load(fh)
        import lm_numpy
        # Any exception here is the point: load_lm() would swallow it.
        cls.lm = lm_numpy.load(provost_model.MODEL_DIR)

    def test_load_lm_finds_the_installed_model(self):
        import layout_model
        env = {k: v for k, v in os.environ.items() if k != "PROVOST_LAYOUT_LM"}
        lm = layout_model.load_lm(env)
        self.assertIsNotNone(lm, "load_lm() returned None: provost would label with the trees")
        self.assertTrue(getattr(lm[0], "numpy", False), "load_lm() did not pick the numpy model")

    def test_labels_a_capture(self):
        import layout_model
        roles, starts, probs = layout_model.label_lm(self.lm, CAPTURE)
        self.assertEqual(len(roles), len(CAPTURE), roles)
        for r in roles:
            self.assertIn(r, self.cfg["roles"], roles)
        self.assertEqual(roles[0], "header", roles)
        self.assertEqual(roles[1:], ["row"] * (len(CAPTURE) - 1), roles)


if __name__ == "__main__":
    unittest.main()
