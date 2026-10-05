# stabilization-uplift

Stabilization Score (SS) and Stabilization Uplift (SU): metrics for evaluating how stable a model's performance is under sudden distribution shifts, such as macroeconomic shocks.

Introduced in *"Mitigating Model Drift in Developing Economies Using Synthetic Data and Outliers"*, [NeurIPS 2025 Workshop on Generative AI in Finance](https://openreview.net/forum?id=zfTaFD0B5Z) ([arXiv:2510.09294](https://arxiv.org/abs/2510.09294)).

## Installation

```sh
pip install stabilization-uplift
```

To also compute the distribution shift from data (requires SDV):

```sh
pip install "stabilization-uplift[shift]"
```

## Metrics

**Stabilization Score (SS)** measures how much a model's ROC AUC changes between the base (pre-shock) and shock periods, normalized by the severity of the distribution shift:

$$SS = 1 - \frac{|AUC_{base} - AUC_{shock}|}{1 + \ln(1 + \text{shift})}$$

SS = 1 means the model's performance did not change under the shock.

**Stabilization Uplift (SU)** compares two models, A (e.g. a baseline) and B (e.g. trained with synthetic outliers). It is a weighted difference of their Stabilization Scores, where sigmoid weights distinguish a drop in AUC from a gain and favour the model with higher AUC. SU = 0 means B is not more stable than A; larger values mean a stronger uplift.

**Distribution shift** is the mean per-column shift between the base and shock data: Total Variation Distance for categorical columns and the Kolmogorov-Smirnov statistic for numerical columns.

## Usage

```python
from stabilization_uplift import stabilization_score, stabilization_uplift

# ROC AUC of model A (baseline) and model B on the base and shock test sets
auc_base_A, auc_shock_A = 0.80, 0.70
auc_base_B, auc_shock_B = 0.81, 0.78
dist_shift = 0.2

stabilization_score(auc_base_A, auc_shock_A, dist_shift)  # SS of model A
stabilization_score(auc_base_B, auc_shock_B, dist_shift)  # SS of model B
stabilization_uplift(auc_base_A, auc_shock_A, auc_base_B, auc_shock_B, dist_shift)  # SU of B over A
```

Computing the distribution shift from data, as in the paper:

```python
import pandas as pd
from stabilization_uplift import distribution_shift

base_data = pd.concat([train_data, base_test_data])
shock_data = pd.concat([train_data, shock_test_data])
dist_shift = distribution_shift(base_data, shock_data)
```

The data used in the paper is available on Hugging Face: [`zyplai/stabilization-uplift`](https://huggingface.co/datasets/zyplai/stabilization-uplift). The experiments are in the [GitHub repository](https://github.com/zypl-ai/stabilization_uplift).

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
