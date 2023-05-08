# project: p7
# submitter: cchen659
# partner: none
# hours: 15

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np


class UserPredictor:
    def __init__(self):
        num_transformer = Pipeline([
            ("scaler", StandardScaler())
        ])
        cat_transformer = Pipeline([
            ("onehot", OneHotEncoder())
        ])
        self.pipeline = ColumnTransformer([
            ("num", num_transformer, ["age", "past_purchase_amt", "seconds"]),
            ("cat", cat_transformer, ["badge"])
        ], remainder = "drop")
        self.lr = LogisticRegression()

    def fit(self, train_users, train_logs, train_y):
        data1 = pd.merge(train_users, train_y, on = "user_id")
        merged = pd.merge(train_users, train_logs, on = "user_id", how = "right").fillna(0)
        aggregated = merged.groupby("user_id").agg({"seconds": "sum"})
        data = pd.merge(data1, aggregated, on = "user_id", how = "left").fillna(0)
        
        X = data[["age", "past_purchase_amt", "badge", "seconds"]]
        y = data["y"]
        X_preprocessed = self.pipeline.fit_transform(X)

        self.lr.fit(X_preprocessed, y)

    def predict(self, test_users, test_logs):
        merged = pd.merge(test_users, test_logs, on = "user_id", how = "right").fillna(0)
        if len(merged) == 0:
            return np.zeros(30000, dtype = int)
        aggregated = merged.groupby("user_id").agg({"seconds": "sum"})
        data = pd.merge(test_users, aggregated, on = "user_id", how = "left").fillna(0)

        X_test = data[["age", "past_purchase_amt", "badge", "seconds"]]
        X_test_preprocessed = self.pipeline.transform(X_test)

        y_pred = self.lr.predict(X_test_preprocessed)

        return y_pred
