import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from ngboost import NGBClassifier
from sklearn.tree import DecisionTreeRegressor
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
import rtdl
import torch.nn as nn



class NGBoostWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=500, learning_rate=0.03,
                 max_depth=3, min_samples_leaf=20, minibatch_frac=0.8, random_state=42):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.minibatch_frac = minibatch_frac
        self.random_state = random_state
        self.model = None

    def fit(self, X, y):
        self.model = NGBClassifier(
            Base=DecisionTreeRegressor(max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf),
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            minibatch_frac=self.minibatch_frac,
            natural_gradient=True,
            verbose=False,
            random_state=self.random_state
        )
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)
        
class TabNetWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self,
                 n_d=16,
                 n_a=16,
                 n_steps=5,
                 gamma=1.3,
                 lambda_sparse=1e-4,
                 optimizer_fn=torch.optim.Adam,
                 optimizer_params=dict(lr=0.02),
                 mask_type='sparsemax',
                 verbose=0,
                 seed=42):
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.lambda_sparse = lambda_sparse
        self.optimizer_fn = optimizer_fn
        self.optimizer_params = optimizer_params
        self.mask_type = mask_type
        self.verbose = verbose
        self.seed = seed
        self.model = None

    def fit(self, X, y):
        """
        X: pd.DataFrame with numerical features after preprocessing  
        y: pd.Series or np.array with labels
        """
        X_np = X.values.astype(np.float32)
        y_np = y.values.astype(np.int64)
        self.model = TabNetClassifier(
            n_d=self.n_d,
            n_a=self.n_a,
            n_steps=self.n_steps,
            gamma=self.gamma,
            lambda_sparse=self.lambda_sparse,
            optimizer_fn=self.optimizer_fn,
            optimizer_params=self.optimizer_params,
            mask_type=self.mask_type,
            verbose=self.verbose,
            seed=self.seed
        )
        self.model.fit(X_np, y_np)
        return self

    def predict_proba(self, X):
        X_np = X.values.astype(np.float32)
        return self.model.predict_proba(X_np)
        
class FTTransformerWrapper(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        d_token=64,
        n_blocks=3,
        ffn_d_hidden=128,
        ff_dropout=0.2,
        attention_dropout=0.2,
        residual_dropout=0.1,
        lr=1e-3,
        batch_size=512,
        n_epochs=200,
        patience=20,
        verbose=1,
        seed=42
    ):
        self.d_token = d_token
        self.n_blocks = n_blocks
        self.ffn_d_hidden = ffn_d_hidden
        self.ff_dropout = ff_dropout
        self.attention_dropout = attention_dropout
        self.residual_dropout = residual_dropout
        self.lr = lr
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.patience = patience
        self.verbose = verbose
        self.seed = seed

        self.enc = None
        self.cat_cols = None

    def fit(self, X, y):
        torch.manual_seed(self.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        num_cols = [c for c in X.columns if c not in self.cat_cols]

        X_num = X[num_cols].astype(float).values
        if self.cat_cols:
            self.enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            X_cat = self.enc.fit_transform(X[self.cat_cols]).astype(int)
        else:
            X_cat = None

        X_num = torch.tensor(X_num, dtype=torch.float32).to(device)
        X_cat = torch.tensor(X_cat, dtype=torch.long).to(device) if X_cat is not None else None
        y = torch.tensor(y.values, dtype=torch.long).to(device)

        n_features = X_num.shape[1]
        n_classes = len(torch.unique(y))
        cat_cardinalities = [int(X[self.cat_cols].nunique()[c]) for c in self.cat_cols] if self.cat_cols else None

        self.model = rtdl.FTTransformer.make_baseline(
            n_num_features=n_features,
            cat_cardinalities=cat_cardinalities,
            d_token=self.d_token,
            n_blocks=self.n_blocks,
            attention_dropout=self.attention_dropout,
            ffn_d_hidden=self.ffn_d_hidden,
            ffn_dropout=self.ff_dropout,
            residual_dropout=self.residual_dropout,
            d_out=n_classes
        ).to(device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.CrossEntropyLoss()

        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.n_epochs):
            self.model.train()
            permutation = torch.randperm(X_num.size(0))
            for i in range(0, X_num.size(0), self.batch_size):
                idx = permutation[i:i+self.batch_size]
                batch_x_num = X_num[idx]
                batch_x_cat = X_cat[idx] if X_cat is not None else None
                batch_y = y[idx]

                self.optimizer.zero_grad()
                logits = self.model(batch_x_num, batch_x_cat)
                loss = loss_fn(logits, batch_y)
                loss.backward()
                self.optimizer.step()

            if self.verbose:
                print(f"Epoch {epoch}, loss: {loss.item():.4f}")

            if loss.item() < best_loss:
                best_loss = loss.item()
                patience_counter = 0
                best_state = self.model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    if self.verbose:
                        print("Early stopping")
                    break

        self.model.load_state_dict(best_state)
        return self

    def predict_proba(self, X):
        self.model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        num_cols = [c for c in X.columns if c not in self.cat_cols]
        X_num = X[num_cols].astype(float).values
        X_num = torch.tensor(X_num, dtype=torch.float32).to(device)

        if self.cat_cols:
            X_cat = self.enc.transform(X[self.cat_cols]).astype(int)
            X_cat = torch.tensor(X_cat, dtype=torch.long).to(device)
        else:
            X_cat = None

        with torch.no_grad():
            logits = self.model(X_num, X_cat)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)