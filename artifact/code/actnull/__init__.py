"""Calibrated null models for task-vector subspace overlap.

  actnull.null        the overlap statistic, the activation-conditioned null, the null for
                      low-rank adapters, and the block structure the calibration set resolves
  actnull.covariance  the activation second moment C_H of a frozen base model

Import the submodules directly:

    from actnull.null import coupling_destroying_null, overlap, resolvable_blocks

They are not re-exported here, so that ``python -m actnull.null`` runs the command-line entry
point without importing the module twice under different names.
"""
__all__ = ["null", "covariance", "experiment", "positive_control_utils"]
