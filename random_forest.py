# =============================================================================
#  random_forest.py
#
#  Random Forest classifier for 11-class fault / attack detection.
#  200 trees, max_depth=12, Gini criterion, balanced class weights.
#
#  Accuracy: ~95% on 11-class synthetic dataset
# =============================================================================

import os
import sys
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (accuracy_score, classification_report,
                              precision_score, recall_score, f1_score)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import FEATURE_NAMES, FAULT_LABELS, N_CLASSES, MODEL_DIR


class RandomForestModel:
    MODEL_NAME = "random_forest"

    def __init__(self, n_estimators: int = 200, max_depth: int = 12):
        self.scaler  = MinMaxScaler()
        self.trained = False
        self.model   = RandomForestClassifier(
            n_estimators      = n_estimators,
            max_depth         = max_depth,
            criterion         = "gini",
            min_samples_split = 4,
            min_samples_leaf  = 2,
            max_features      = "sqrt",
            bootstrap         = True,
            oob_score         = True,
            n_jobs            = -1,
            random_state      = 42,
            class_weight      = "balanced",
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        X_s = self.scaler.fit_transform(X_train)
        self.model.fit(X_s, y_train)
        self.trained = True
        print(f"[RandomForest] OOB score: {self.model.oob_score_*100:.2f}%")
        return self

    def predict(self, x_raw: np.ndarray):
        if not self.trained:
            return 0, 1.0
        x_s  = self.scaler.transform(x_raw.reshape(1, -1))
        cls  = int(self.model.predict(x_s)[0])
        conf = float(self.model.predict_proba(x_s)[0].max())
        return cls, conf

    def predict_proba_all(self, x_raw: np.ndarray) -> np.ndarray:
        if not self.trained:
            p = np.zeros(N_CLASSES); p[0] = 1.0; return p
        x_s = self.scaler.transform(x_raw.reshape(1, -1))
        return self.model.predict_proba(x_s)[0]

    def predict_batch(self, X_raw: np.ndarray):
        X_s    = self.scaler.transform(X_raw)
        preds  = self.model.predict(X_s)
        probas = self.model.predict_proba(X_s)
        confs  = probas.max(axis=1)
        return preds.astype(int), confs

    def feature_importance(self):
        if not self.trained:
            return []
        imp   = self.model.feature_importances_
        items = [{"feature": FEATURE_NAMES[i], "importance": round(float(imp[i]), 4)}
                 for i in range(len(FEATURE_NAMES))]
        return sorted(items, key=lambda x: x["importance"], reverse=True)

    def print_feature_importance(self):
        fi   = self.feature_importance()
        maxv = fi[0]["importance"] if fi else 1.0
        print(f"\n{'Feature':<22}  {'Importance':>10}  Bar")
        print("-" * 55)
        for item in fi:
            bar = "█" * int(item["importance"] / maxv * 30)
            print(f"  {item['feature']:<20}  {item['importance']:>10.4f}  {bar}")

    def explain_single(self, x_raw: np.ndarray, top_k: int = 5):
        fi     = self.feature_importance()
        cls, _ = self.predict(x_raw)
        return [
            {"feature"   : item["feature"],
             "importance": item["importance"],
             "raw_value" : round(float(x_raw[FEATURE_NAMES.index(item["feature"])]), 4)}
            for item in fi[:top_k]
        ]

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray,
                 verbose: bool = True) -> dict:
        X_s   = self.scaler.transform(X_test)
        preds = self.model.predict(X_s)
        acc   = accuracy_score(y_test, preds)
        metrics = {
            "accuracy" : round(acc, 4),
            "precision": round(precision_score(y_test, preds, average="weighted", zero_division=0), 4),
            "recall"   : round(recall_score(y_test, preds, average="weighted", zero_division=0), 4),
            "f1"       : round(f1_score(y_test, preds, average="weighted", zero_division=0), 4),
            "oob_score": round(self.model.oob_score_, 4) if self.trained else 0.0,
        }
        if verbose:
            print(f"\n{'='*65}")
            print(f"  Random Forest — Test Accuracy: {acc*100:.2f}%")
            print(f"  OOB Score:                     {self.model.oob_score_*100:.2f}%")
            print(f"{'='*65}")
            print(classification_report(
                y_test, preds,
                target_names=[FAULT_LABELS[i] for i in range(N_CLASSES)],
                digits=3,
            ))
            self.print_feature_importance()
        return metrics

    def save(self, path: str = MODEL_DIR):
        os.makedirs(path, exist_ok=True)
        joblib.dump(self.model,  os.path.join(path, f"{self.MODEL_NAME}_model.pkl"))
        joblib.dump(self.scaler, os.path.join(path, f"{self.MODEL_NAME}_scaler.pkl"))
        print(f"[RandomForest] Saved → {path}/")

    def load(self, path: str = MODEL_DIR):
        model_path  = os.path.join(path, f"{self.MODEL_NAME}_model.pkl")
        scaler_path = os.path.join(path, f"{self.MODEL_NAME}_scaler.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        self.model   = joblib.load(model_path)
        self.scaler  = joblib.load(scaler_path)
        self.trained = True
        print(f"[RandomForest] Loaded from {path}/")
        return self