import pandas as pd
import numpy as np
from sdv.metadata import SingleTableMetadata
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator

def detect_stable_metadata(data: pd.DataFrame):

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data)

    return metadata

def create_categorical_columns(data: pd.DataFrame, target: str) -> list:

  metadata = detect_stable_metadata(data)

  categorical_columns = [column for column, properties in metadata.to_dict()["columns"].items() if properties["sdtype"] == "categorical"]
  categorical_columns.remove(target)

  return categorical_columns

def calc_auc(train_data: pd.DataFrame, 
             val_data: pd.DataFrame, 
             model: BaseEstimator,
             target: str) -> float:
    
    X_train, y_train = train_data.drop(columns=[target]), train_data[target]

    X_train = X_train.copy()
    for col in X_train.select_dtypes(include=['datetime64', 'timedelta64']):
        X_train[col] = X_train[col].astype(str)
        
    model_clone = clone(model)
    model_clone.fit(X_train, y_train)

    train_prediction_probabilities = model_clone.predict_proba(X_train)
    transformers_train_auc = roc_auc_score(y_train, train_prediction_probabilities[:, 1])
    
    X_test, y_test = val_data.drop(columns=[target]), val_data[target]

    X_test = X_test.copy()

    for col in X_test.select_dtypes(include=['datetime64', 'timedelta64']):
        X_test[col] = X_test[col].astype(str)

    test_prediction_probabilities = model_clone.predict_proba(X_test)
    transformers_test_auc = roc_auc_score(y_test, test_prediction_probabilities[:, 1])
    
    return transformers_train_auc, transformers_test_auc, model_clone

def preprocess_for_model(train, base, shock, synth_dict, model_name):
    
    """
    Preprocesses the data for a specific model:
      Dates → year/month
    Categorical features:
      XGB → OrdinalEncoder with unknown_value=-1
      LightGBM, HGB, NGBoost, TabNet, FTTransformer → OrdinalEncoder with unknown_value=-1
    """
    
    train, base, shock = train.copy(), base.copy(), shock.copy()
    synth_dict = {k: v.copy() for k, v in synth_dict.items()}

    all_dfs = [train, base, shock] + list(synth_dict.values())

    date_cols = [c for c in train.columns if np.issubdtype(train[c].dtype, np.datetime64)]
    for df in all_dfs:
        for col in date_cols:
            if col in df:
                df[col] = pd.to_datetime(df[col], errors="coerce")
                df[col + "_year"] = df[col].dt.year
                df[col + "_month"] = df[col].dt.month
                df.drop(columns=[col], inplace=True)

    cat_cols = train.select_dtypes(include=["object", "category"]).columns.tolist()

    if cat_cols:
        enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        train[cat_cols] = enc.fit_transform(train[cat_cols].astype(str))
        base[cat_cols] = enc.transform(base[cat_cols].astype(str))
        shock[cat_cols] = enc.transform(shock[cat_cols].astype(str))
        for k in synth_dict:
            synth_dict[k][cat_cols] = enc.transform(synth_dict[k][cat_cols].astype(str))

    return train, base, shock, synth_dict
    
def fill_missing_with_mode(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if df[col].isnull().any():
            mode_value = df[col].mode(dropna=True)
            if not mode_value.empty:
                fill_value = mode_value[0]
                if pd.api.types.is_numeric_dtype(df[col]) and isinstance(fill_value, float):
                    fill_value = round(fill_value, 0)
                df[col] = df[col].fillna(fill_value)
    return df