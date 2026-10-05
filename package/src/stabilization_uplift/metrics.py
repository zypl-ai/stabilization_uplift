import numpy as np

_EPSILON = 1e-5
_K = 100
_K_SHOCK = 1000


def _sigmoid_weight(x: float, k: float) -> float:
    return 1 - 1 / (1 + np.exp(k * x))


def _symmetric_auc(auc: float, name: str) -> float:
    if not 0 < auc <= 1:
        raise ValueError(f"{name} should be in (0, 1], got {auc}")
    return 1 - auc if auc < 0.5 else auc


def stabilization_score(auc_base: float, auc_shock: float, dist_shift: float) -> float:
    """Stabilization Score (SS) of a single model.

    Measures the change in ROC AUC between the base and shock periods,
    normalized by the severity of the distribution shift:

        SS = 1 - |AUC_base - AUC_shock| / (1 + ln(1 + dist_shift))

    SS = 1 means the model's performance did not change under the shock.

    Args:
        auc_base: ROC AUC on the base (pre-shock) test set.
        auc_shock: ROC AUC on the shock test set.
        dist_shift: Distribution shift between the base and shock data, e.g.
            from :func:`stabilization_uplift.distribution_shift`.

    Returns:
        The Stabilization Score.
    """
    auc_base = _symmetric_auc(auc_base, "auc_base")
    auc_shock = _symmetric_auc(auc_shock, "auc_shock")

    delta_auc = abs(auc_base - auc_shock)
    shift_val = max(1 + np.log(1 + dist_shift + _EPSILON), _EPSILON)

    return float(1 - delta_auc / shift_val)


def stabilization_uplift(auc_base_A: float,
                         auc_shock_A: float,
                         auc_base_B: float,
                         auc_shock_B: float,
                         dist_shift: float) -> float:
    """Stabilization Uplift (SU) of model B over model A.

    Compares the Stabilization Scores of two models with sigmoid weights that
    account for what the plain difference SS_B - SS_A misses:

    * ``w_A``, ``w_B`` distinguish a drop in AUC under the shock (weight close
      to 0) from a gain (weight close to 1), since SS uses ``|delta AUC|``;
    * ``w`` rewards model B only if its shock AUC is higher than model A's;
    * ``w_superiority`` favours the model with higher AUC on both periods.

    Args:
        auc_base_A: ROC AUC of model A on the base test set.
        auc_shock_A: ROC AUC of model A on the shock test set.
        auc_base_B: ROC AUC of model B on the base test set.
        auc_shock_B: ROC AUC of model B on the shock test set.
        dist_shift: Distribution shift between the base and shock data.

    Returns:
        The Stabilization Uplift, non-negative; 0 means no uplift of B over A.
    """
    auc_base_A = _symmetric_auc(auc_base_A, "auc_base_A")
    auc_shock_A = _symmetric_auc(auc_shock_A, "auc_shock_A")
    auc_base_B = _symmetric_auc(auc_base_B, "auc_base_B")
    auc_shock_B = _symmetric_auc(auc_shock_B, "auc_shock_B")

    w_A = _sigmoid_weight(auc_shock_A - auc_base_A, _K)
    w_B = _sigmoid_weight(auc_shock_B - auc_base_B, _K)
    w = _sigmoid_weight(auc_shock_B - auc_shock_A, _K_SHOCK)
    w_superiority = _sigmoid_weight((auc_base_B - auc_base_A) + (auc_shock_B - auc_shock_A), _K)

    w_B = w_B * w_superiority
    w_A = w_A * (1 - w_superiority)

    score_A = stabilization_score(auc_base_A, auc_shock_A, dist_shift)
    score_B = stabilization_score(auc_base_B, auc_shock_B, dist_shift)

    return float(max(w * (w_B * score_B - w_A * score_A), 0.0))
