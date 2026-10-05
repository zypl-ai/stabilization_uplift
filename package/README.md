# stabilization-uplift

[![PyPI](https://img.shields.io/pypi/v/stabilization-uplift)](https://pypi.org/project/stabilization-uplift/)
[![NeurIPS 2025 Workshop](https://img.shields.io/badge/NeurIPS%202025-GenAI%20in%20Finance%20Workshop-blue)](https://openreview.net/forum?id=zfTaFD0B5Z)
[![arXiv](https://img.shields.io/badge/arXiv-2510.09294-b31b1b)](https://arxiv.org/abs/2510.09294)
[![Hugging Face Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-zyplai%2Fstabilization--uplift-yellow)](https://huggingface.co/datasets/zyplai/stabilization-uplift)

Metrics for measuring how stable a classifier stays when the data distribution suddenly shifts, for example after a macroeconomic shock, and whether a modified model is more stable than a baseline.

- **Stabilization Score (SS)**: how much one model's ROC AUC changed between the base (pre-shock) and shock periods, relative to how strong the shift was.
- **Stabilization Uplift (SU)**: whether model B is more stable *and* not worse than model A under the shock. Use it to evaluate a drift-mitigation method such as training on synthetic data with outliers.
- **Distribution shift**: how strong the shift between the base and shock data is, computed from the data itself.

Introduced in *"Mitigating Model Drift in Developing Economies Using Synthetic Data and Outliers"*, [NeurIPS 2025 Workshop on Generative AI in Finance](https://openreview.net/forum?id=zfTaFD0B5Z) ([arXiv:2510.09294](https://arxiv.org/abs/2510.09294)).

## Installation

```sh
pip install stabilization-uplift
```

`stabilization_score` and `stabilization_uplift` only need NumPy. To compute the distribution shift from data, install the optional dependencies (pandas, SDV, SDMetrics):

```sh
pip install "stabilization-uplift[shift]"
```

Requires Python 3.9+.

## Quick start

If you already have the ROC AUCs of your models and the distribution shift:

```python
from stabilization_uplift import stabilization_score, stabilization_uplift

# Model A: baseline. Model B: e.g. trained with synthetic outliers.
auc_base_A, auc_shock_A = 0.80, 0.70   # A loses 0.10 AUC under the shock
auc_base_B, auc_shock_B = 0.80, 0.81   # B holds up
dist_shift = 0.2

stabilization_score(auc_base_A, auc_shock_A, dist_shift)   # 0.915
stabilization_score(auc_base_B, auc_shock_B, dist_shift)   # 0.992
stabilization_uplift(auc_base_A, auc_shock_A, auc_base_B, auc_shock_B, dist_shift)   # 0.725
```

## End-to-end example

The full evaluation protocol from the paper on the open [Lending Club data](https://huggingface.co/datasets/zyplai/stabilization-uplift): train a baseline A on real data and a model B on real plus synthetic data with outliers, evaluate both before and after the 2018 shock, and compare their stability.

```sh
pip install "stabilization-uplift[shift]" scikit-learn huggingface_hub
```

```python
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from stabilization_uplift import distribution_shift, stabilization_score, stabilization_uplift

REPO = "hf://datasets/zyplai/stabilization-uplift@v1.0"
TARGET = "loan_condition_int"

# Synthetic data defines the feature set used in the paper
synthetic = pd.read_parquet(f"{REPO}/synthetic/outliers_0_05.parquet").drop(columns="issue_d")
columns = list(synthetic.columns)


def load(split, n=4000):
    data = pd.read_parquet(f"{REPO}/lending_club/{split}.parquet")
    return data[columns].sample(n=n, random_state=100)


train, base_test, shock_test = load("train"), load("base_test"), load("shock_test")


def fit(data):
    X = data.drop(columns=TARGET)
    X = X.astype({c: "category" for c in X.select_dtypes("object").columns})
    return HistGradientBoostingClassifier(categorical_features="from_dtype", random_state=0).fit(X, data[TARGET])


def auc(model, data):
    X = data.drop(columns=TARGET)
    X = X.astype({c: "category" for c in X.select_dtypes("object").columns})
    return roc_auc_score(data[TARGET], model.predict_proba(X)[:, 1])


model_A = fit(train)                                # real data only
model_B = fit(pd.concat([train, synthetic]))        # real + synthetic data with outliers

auc_base_A, auc_shock_A = auc(model_A, base_test), auc(model_A, shock_test)
auc_base_B, auc_shock_B = auc(model_B, base_test), auc(model_B, shock_test)

shift = distribution_shift(pd.concat([train, base_test]), pd.concat([train, shock_test]))

print(f"AUC A: base {auc_base_A:.3f}, shock {auc_shock_A:.3f}")
print(f"AUC B: base {auc_base_B:.3f}, shock {auc_shock_B:.3f}")
print(f"Distribution shift: {shift:.3f}")
print(f"SS of A: {stabilization_score(auc_base_A, auc_shock_A, shift):.3f}")
print(f"SS of B: {stabilization_score(auc_base_B, auc_shock_B, shift):.3f}")
print(f"SU of B over A: {stabilization_uplift(auc_base_A, auc_shock_A, auc_base_B, auc_shock_B, shift):.3f}")
```

Output:

```
AUC A: base 0.649, shock 0.658
AUC B: base 0.650, shock 0.649
Distribution shift: 0.073
SS of A: 0.992
SS of B: 0.999
SU of B over A: 0.000
```

This run shows why SU is needed on top of SS. Model B has the higher SS: its AUC barely moved under the shock. But its shock AUC (0.649) is lower than model A's (0.658), so B is not an improvement, and SU is 0. To compare several synthetic datasets, repeat the evaluation for each `synthetic/*` file and compare their SU values.

## How to read the results

**Stabilization Score** is in [0.5, 1].

- SS = 1: the model's AUC did not change under the shock.
- Lower SS: a larger change in AUC. A change of the same size is penalized less when the distribution shift is stronger, because a strong shock makes some change expected.
- SS is symmetric: a drop and a gain of the same size give the same SS. Use SU to tell them apart.

**Stabilization Uplift** is in [0, 1).

- SU = 0: model B is not more stable than A, or B's AUC under the shock is not higher than A's.
- Larger SU: a stronger improvement. SU is high when B keeps or improves its AUC under the shock while A loses AUC.
- SU is asymmetric: it measures B over A. To check A over B, swap the arguments.

Examples with `dist_shift = 0.2`:

| Scenario | AUC A (base, shock) | AUC B (base, shock) | SS A | SS B | SU |
|---|---|---|---|---|---|
| B improves under the shock, A drops | 0.80, 0.70 | 0.80, 0.81 | 0.915 | 0.992 | **0.725** |
| B stays flat, A drops slightly | 0.80, 0.79 | 0.80, 0.80 | 0.992 | 1.000 | **0.294** |
| Both drop, B less than A | 0.80, 0.70 | 0.81, 0.78 | 0.915 | 0.975 | **0.046** |
| Both drop equally | 0.80, 0.70 | 0.80, 0.70 | 0.915 | 0.915 | **0.000** |
| B is worse than A | 0.81, 0.78 | 0.80, 0.70 | 0.975 | 0.915 | **0.000** |

Note the third row: B drops less than A, so its SS is higher, but SU stays close to 0. SU rewards a model whose AUC does not decrease under the shock, not one that merely degrades more slowly.

## API reference

### `stabilization_score(auc_base, auc_shock, dist_shift) -> float`

Stabilization Score of a single model.

| Argument | Description |
|---|---|
| `auc_base` | ROC AUC on the base (pre-shock) test set, in (0, 1] |
| `auc_shock` | ROC AUC on the shock test set, in (0, 1] |
| `dist_shift` | Distribution shift between the base and shock data, normally in [0, 1] (see `distribution_shift`) |

```
SS = 1 - |AUC_base - AUC_shock| / (1 + ln(1 + dist_shift))
```

### `stabilization_uplift(auc_base_A, auc_shock_A, auc_base_B, auc_shock_B, dist_shift) -> float`

Stabilization Uplift of model B over model A. Arguments are the base and shock ROC AUCs of both models and the distribution shift, as above.

```
SU = max(w * (w_B' * SS_B - w_A' * SS_A), 0)

w_A           = sigmoid(100  * (AUC_shock_A - AUC_base_A))    # ~0 if A's AUC dropped, ~1 if it rose
w_B           = sigmoid(100  * (AUC_shock_B - AUC_base_B))    # same for B
w             = sigmoid(1000 * (AUC_shock_B - AUC_shock_A))   # ~1 only if B beats A under the shock
w_superiority = sigmoid(100  * ((AUC_base_B - AUC_base_A) + (AUC_shock_B - AUC_shock_A)))
w_B'          = w_B * w_superiority
w_A'          = w_A * (1 - w_superiority)
```

### `distribution_shift(base_data, shock_data) -> float`

Mean per-column shift between two pandas DataFrames with the same columns, in [0, 1]. Requires `pip install "stabilization-uplift[shift]"`.

- Column types are detected with [SDV](https://github.com/sdv-dev/SDV) on `base_data`.
- Categorical columns: Total Variation Distance. Numerical columns: Kolmogorov-Smirnov statistic.
- Other column types (datetime, IDs, addresses, etc.) are ignored.
- As in the paper, pass the training data concatenated with each test set (see the example above), so that both sides share the same reference data.

### Notes

- An AUC below 0.5 is replaced with `1 - AUC`: a model with inverted predictions is treated as equally informative.
- An AUC outside (0, 1] raises `ValueError`.
- All functions return a Python `float`.

## Data and experiments

- **Dataset:** [`zyplai/stabilization-uplift`](https://huggingface.co/datasets/zyplai/stabilization-uplift) on Hugging Face: Lending Club splits before and after the shock, and synthetic training data with different proportions of outliers.
- **Code and notebooks from the paper:** [github.com/zypl-ai/stabilization_uplift](https://github.com/zypl-ai/stabilization_uplift).

## Citation

```bibtex
@inproceedings{varshavskiy2025mitigating,
  title     = {Mitigating Model Drift in Developing Economies Using Synthetic Data and Outliers},
  author    = {Varshavskiy, Ilyas and Boboeva, Bonu and Khalilbekov, Shuhrat and Azimi, Azizjon and Shulgin, Sergey and Nizamitdinov, Akhlitdin and S{\'a}ez de Oc{\'a}riz Borde, Haitz},
  booktitle = {NeurIPS 2025 Workshop on Generative AI in Finance},
  year      = {2025},
  url       = {https://openreview.net/forum?id=zfTaFD0B5Z}
}
```

## Credits

Author: [zypl.ai](https://zypl.ai)

## License

MIT
