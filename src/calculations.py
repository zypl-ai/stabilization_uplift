import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.base import BaseEstimator  

from sdmetrics.single_column import KSComplement
from sdmetrics.single_column import TVComplement

from src.utilities import (detect_stable_metadata, create_categorical_columns, calc_auc)

def distribution_shift(base_data: pd.DataFrame, shock_data: pd.DataFrame) -> float:

    metadata = detect_stable_metadata(base_data)

    categorical_columns = [column for column, properties in metadata.to_dict()["columns"].items() if properties["sdtype"] == "categorical"]
    numeric_columns = [column for column, properties in metadata.to_dict()["columns"].items() if properties["sdtype"] == "numerical"]

    shifts = []
    for column_name in categorical_columns:
      shifts.append(1 - TVComplement.compute(real_data=base_data[column_name], synthetic_data=shock_data[column_name]))
    for column_name in numeric_columns:
      shifts.append(1 - KSComplement.compute(real_data=base_data[column_name], synthetic_data=shock_data[column_name]))
    return np.mean(shifts) if shifts else 0.0

def stabilization_score(auc_base: float, auc_shock: float, dist_shift: float) -> float:

    assert 0 != auc_base, "auc_base should be more than 0"
    assert 0 != auc_shock, "auc_shock should be more than 0"

    if auc_base < 0.5:
      auc_base = 1-auc_base
    if auc_shock < 0.5:
      auc_shock = 1-auc_shock

    epsilon = 1e-5

    shift_scale_func = lambda x: 1 + np.log(1 + x + epsilon)

    delta_auc = abs(auc_base - auc_shock)
    shift_val = shift_scale_func(dist_shift)

    shift_val = max(shift_val, epsilon)

    score = 1 - (delta_auc / shift_val)

    return score

def stabilization_uplift(auc_base_A: float,
                         auc_shock_A: float,
                         auc_base_B: float,
                         auc_shock_B: float,
                         dist_shift: float) -> float:

      '''
      Problem 1: Delta = |auc_base - auc_shock| in stabilization_score (SS) equal if auc_base > auc_shock and
      auc_base < auc_shock both. Finally stabilization_uplift shown uplift in cases with auc_base > auc_shock.

      Solution: w_A, w_B - based on sigmoid function, if auc_base > auc_shock close 0, if auc_base = auc_shock,
      close to 0.5, if auc_base < auc_shock close to 1.

      k - speed of sigmoid grows. When auc_base > auc_shock in some epsion, regulated by k, SS shown non zerro value.
      Possibly it will be fair when auc_base \approx auc_shock, then stabilization_uplift (SU) shown small uplift.

      Problem 2: SU shown equal uplift when auc_shock_B >= auc_shock_A and auc_shock_B <= auc_shock_A, due to the base values.

      Solution: w - based on sigmoid with high k, for reduction epsilon.

      Problem 3: in B and H cases uplift not shown, even though B models values > A models values.

      Solution: w_superiority

      '''

      assert 0 != auc_base_A, "auc_base_A should be more than 0"
      assert 0 != auc_shock_A, "auc_shock_A should be more than 0"

      if auc_base_A < 0.5:
        auc_base_A = 1-auc_base_A
      if auc_shock_A < 0.5:
        auc_shock_A = 1-auc_shock_A
        
      assert 0 != auc_base_B, "auc_base_B should be more than 0"
      assert 0 != auc_shock_B, "auc_shock_B should be more than 0"

      if auc_base_B < 0.5:
        auc_base_B = 1-auc_base_A
      if auc_shock_B < 0.5:
        auc_shock_B = 1-auc_shock_A

      k = 100
      w_A = 1 - 1 / (1 + np.exp(k * (auc_shock_A - auc_base_A)))
      w_B = 1 - 1 / (1 + np.exp(k * (auc_shock_B - auc_base_B)))

      w = 1 - 1 / (1 + np.exp(1000 * (auc_shock_B - auc_shock_A)))

      w_superiority = 1 - 1 / (1 + np.exp(k * ((auc_base_B - auc_base_A) + (auc_shock_B - auc_shock_A))))

      w_B = w_B * w_superiority

      w_A = w_A *  (1 - w_superiority)

      score_A = stabilization_score(auc_base_A, auc_shock_A, dist_shift)
      score_B = stabilization_score(auc_base_B, auc_shock_B, dist_shift)

      score = max(w*(w_B*score_B - w_A*score_A), 0.)

      return score

def compute_auc_scores(train_data: pd.DataFrame,
                       base_test_data: pd.DataFrame,
                       shock_test_data: pd.DataFrame, 
                       synthetic_data_dict: dict,
                       target: str, model: BaseEstimator) -> dict:
    
    scores_dict = {}

    auc_base_A_list, auc_shock_A_list = [], []
    auc_base_B_list, auc_shock_B_list = [], []
    train_auc_base_A_list, train_auc_shock_A_list = [], []
    train_auc_base_B_list, train_auc_shock_B_list = [], []

    train_auc_base_A, auc_base_A, model_A = calc_auc(train_data.copy(), base_test_data.copy(), model=model,
                                                      target=target)


    train_auc_shock_A, auc_shock_A, _ = calc_auc(train_data.copy(), shock_test_data.copy(), model=model_A,
                                                  target=target)

    for key in tqdm(synthetic_data_dict.keys(), desc="AUC Calculations"):
        
        auc_base_A_list.append(auc_base_A)
        auc_shock_A_list.append(auc_shock_A)
        train_auc_base_A_list.append(train_auc_base_A)
        train_auc_shock_A_list.append(train_auc_shock_A)

        synthetic_real_data_mix = pd.concat([synthetic_data_dict[key], train_data])
        synthetic_real_data_mix.reset_index(drop=True, inplace=True)
        synthetic_real_data_mix.drop_duplicates(inplace=True)

        train_auc_base_B, auc_base_B, model_B = calc_auc(synthetic_real_data_mix, base_test_data, model=model,
                                                         target=target)
        auc_base_B_list.append(auc_base_B)
        train_auc_base_B_list.append(train_auc_base_B)

        train_auc_shock_B, auc_shock_B, _ = calc_auc(synthetic_real_data_mix, shock_test_data, model=model_B,
                                                     target=target)
        auc_shock_B_list.append(auc_shock_B)
        train_auc_shock_B_list.append(train_auc_shock_B)

    scores_dict['train_auc_base_A'] = train_auc_base_A_list
    scores_dict['auc_base_A'] = auc_base_A_list
    scores_dict['train_auc_shock_A'] = train_auc_shock_A_list
    scores_dict['auc_shock_A'] = auc_shock_A_list
    scores_dict['train_auc_base_B'] = train_auc_base_B_list
    scores_dict['auc_base_B'] = auc_base_B_list
    scores_dict['train_auc_shock_B'] = train_auc_shock_B_list
    scores_dict['auc_shock_B'] = auc_shock_B_list

    return scores_dict

def compute_uplift(scores_dict: dict, 
                   train_data: pd.DataFrame, 
                   base_test_data: pd.DataFrame, 
                   shock_test_data: pd.DataFrame) -> dict:
    
    base_data = pd.concat([train_data, base_test_data])
    shock_data = pd.concat([train_data, shock_test_data])

    dist_shift = distribution_shift(base_data, shock_data)

    score_A_list = []
    score_B_list = []
    difference_uplift_list = []
    uplift_score_list = []
    

    for i in  tqdm(range(len(scores_dict['auc_base_A'])), desc='Uplift Calculations'):
        
        score_A_list.append(stabilization_score(scores_dict['auc_base_A'][i], scores_dict['auc_shock_A'][i], dist_shift))
        score_B_list.append(stabilization_score(scores_dict['auc_base_B'][i], scores_dict['auc_shock_B'][i], dist_shift))
        uplift_score_list.append(stabilization_uplift(scores_dict['auc_base_A'][i],
                                                           scores_dict['auc_shock_A'][i],
                                                           scores_dict['auc_base_B'][i],
                                                           scores_dict['auc_shock_B'][i], dist_shift))
        difference_uplift_list.append(max(score_B_list[i] - score_A_list[i], 0.0))
    
    scores_dict['dist_shift'] = dist_shift
    scores_dict['score_A'] = score_A_list
    scores_dict['score_B'] = score_B_list
    scores_dict['difference_uplift'] = difference_uplift_list
    scores_dict['uplift_score'] = uplift_score_list

    return scores_dict

def culc_uplift_by_monthes(train_data: pd.DataFrame, 
                        base_test_data: pd.DataFrame, 
                        shock_test_data: pd.DataFrame,
                        synthetic_data_dict: dict, 
                        date_column_name: str, 
                        target: str,
                        model: BaseEstimator) -> pd.DataFrame:
    
    shock_test_data[date_column_name] = pd.to_datetime(shock_test_data[date_column_name])
    monthly_dataframes = {str(period): df for period, df in shock_test_data.groupby(shock_test_data[date_column_name].dt.to_period('M'))}
    
    _, auc_base_A, model_A = calc_auc(train_data, base_test_data, model=model, target=target)
    
    auc_base_A = max(auc_base_A, 1 - auc_base_A)
    
    base_data = pd.concat([train_data, base_test_data])
    
    results_list = []
    
    for key in tqdm(synthetic_data_dict.keys(), desc='Each Outliers Calculations'):
        synthetic_real_data_mix = pd.concat([synthetic_data_dict[key], train_data]).drop_duplicates().reset_index(drop=True)
        _, auc_base_B, model_B = calc_auc(synthetic_real_data_mix, base_test_data, model=model, target=target)
        auc_base_B = max(auc_base_B, 1 - auc_base_B)
        
        for month in tqdm(monthly_dataframes.keys(), desc='Monthly Calculations', leave=True):
            
            if len(monthly_dataframes[month][target].value_counts().index) >= 2:
                
                _, auc_shock_A, _ = calc_auc(train_data, monthly_dataframes[month], model=model_A, target=target)
                auc_shock_A = max(auc_shock_A, 1 - auc_shock_A)
                
                shock_data = pd.concat([train_data, monthly_dataframes[month]])
                dist_shift = distribution_shift(base_data, shock_data)
                score_A = stabilization_score(auc_base_A, auc_shock_A, dist_shift)
                
                _, auc_shock_B, _ = calc_auc(synthetic_real_data_mix, monthly_dataframes[month], 
                                             model=model_B, target=target)
                auc_shock_B = max(auc_shock_B, 1 - auc_shock_B)
                score_B = stabilization_score(auc_base_B, auc_shock_B, dist_shift)
                
                difference_uplift = max(score_B - score_A, 0.0)
                uplift_score = stabilization_uplift(auc_base_A, auc_shock_A,
                                                    auc_base_B, auc_shock_B, dist_shift)
                
                results_list.append({
                    "month": month,
                    "outliers": key,
                    "auc_base_A": auc_base_A,
                    "auc_shock_A": auc_shock_A,
                    "auc_base_B": auc_base_B,
                    "auc_shock_B": auc_shock_B,
                    "dist_shift": dist_shift,
                    "score_A": score_A,
                    "score_B": score_B,
                    "difference_uplift": difference_uplift,
                    "uplift_score": uplift_score
                })
    
    final_results = pd.DataFrame(results_list)
    return final_results
