# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Martin J. Gallagher
"""The tiny-LM layout labeller for provost's ``--model``: its weights only.

provost (https://github.com/MartinGallagher-code/provost) runs the model with
numpy (``extract/model/lm_numpy.py``); this package carries what it runs.
When ``PROVOST_LAYOUT_LM`` is unset and this package is installed, ``provost
--model`` labels with it instead of the shipped trees.

``MODEL_DIR`` holds ``config.json``, ``tokenizer.json`` and ``weights-N.npz``,
the layout ``ml/export_lm.py`` in provost writes.
"""

import os

__all__ = ["MODEL_DIR", "__version__"]

__version__ = "0.2.1"

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
