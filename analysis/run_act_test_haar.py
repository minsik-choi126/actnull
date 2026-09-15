#!/usr/bin/env python3
"""Run the archived ACT implementation with mathematically Haar QR draws.

The archived implementation calls ``torch.linalg.qr(G)`` directly for Gaussian
matrices.  LAPACK fixes Householder signs, so the returned Q is not Haar unless
the signs of ``diag(R)`` are transferred to Q.  This compatibility runner makes
that correction before dispatching to ``act_test.main``; it leaves the archived
source and all previously reported CSV files untouched.

Usage is identical to ``analysis/act_test.py``.  Set ``ACT_TEST_SOURCE`` to the
directory containing that file when it is not the default sibling repository.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path


DEFAULT_SOURCE = Path(__file__).resolve().with_name("act_test.py")


def _load_source(path: Path):
    spec = importlib.util.spec_from_file_location("prism_archived_act_test", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import ACT source from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    source = Path(os.environ.get("ACT_TEST_SOURCE", DEFAULT_SOURCE))
    act_test = _load_source(source)
    torch = act_test.torch
    original_qr = torch.linalg.qr

    def haar_qr(matrix, *args, **kwargs):
        q, r = original_qr(matrix, *args, **kwargs)
        diagonal = torch.diagonal(r, dim1=-2, dim2=-1)
        one = torch.ones((), dtype=diagonal.dtype, device=diagonal.device)
        phase = torch.where(diagonal >= 0, one, -one)
        return q * phase.unsqueeze(-2), r * phase.unsqueeze(-1)

    torch.linalg.qr = haar_qr
    act_test.main()


if __name__ == "__main__":
    main()
