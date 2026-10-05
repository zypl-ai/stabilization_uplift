import numpy as np


def distribution_shift(base_data, shock_data) -> float:
    """Average per-column distribution shift between two datasets.

    Column types are detected with SDV on ``base_data``. The shift of a
    categorical column is the Total Variation Distance (1 - TVComplement),
    of a numerical column the Kolmogorov-Smirnov statistic (1 - KSComplement).
    Columns of other types (e.g. datetime, id) are ignored.

    In the paper, ``base_data`` is the training data concatenated with the base
    test set and ``shock_data`` the training data concatenated with the shock
    test set.

    Requires the optional dependencies: ``pip install "stabilization-uplift[shift]"``.

    Args:
        base_data: pandas DataFrame from the base (pre-shock) period.
        shock_data: pandas DataFrame from the shock period, same columns.

    Returns:
        Mean shift over the categorical and numerical columns in [0, 1],
        or 0.0 if there are no such columns.
    """
    try:
        from sdmetrics.single_column import KSComplement, TVComplement
        from sdv.metadata import SingleTableMetadata
    except ImportError as e:
        raise ImportError(
            'distribution_shift requires sdv and sdmetrics: pip install "stabilization-uplift[shift]"'
        ) from e

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(base_data)
    columns = metadata.to_dict()["columns"]

    categorical_columns = [c for c, p in columns.items() if p["sdtype"] == "categorical"]
    numeric_columns = [c for c, p in columns.items() if p["sdtype"] == "numerical"]

    shifts = []
    for column in categorical_columns:
        shifts.append(1 - TVComplement.compute(real_data=base_data[column], synthetic_data=shock_data[column]))
    for column in numeric_columns:
        shifts.append(1 - KSComplement.compute(real_data=base_data[column], synthetic_data=shock_data[column]))

    return float(np.mean(shifts)) if shifts else 0.0
