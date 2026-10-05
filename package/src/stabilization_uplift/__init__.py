"""Stabilization Score (SS) and Stabilization Uplift (SU) metrics for model drift under shocks.

Varshavskiy et al., "Mitigating Model Drift in Developing Economies Using Synthetic Data
and Outliers", NeurIPS 2025 Workshop on Generative AI in Finance.
"""

from stabilization_uplift.metrics import stabilization_score, stabilization_uplift
from stabilization_uplift.shift import distribution_shift

__version__ = "0.1.1"

__all__ = ["stabilization_score", "stabilization_uplift", "distribution_shift"]
