"""Post-hoc analyses computed from saved per-candidate predictions.

These require no GPU and no retraining: every table reported in the README is derived
from the arrays in ``results/predictions`` by the functions here.
"""
from . import dataset_stats, specification, subgroup

__all__ = ["dataset_stats", "specification", "subgroup"]
