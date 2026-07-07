# =============================================================================
#  gradient_boosting.py
#
#  Gradient Boosting classifier for 11-class fault / attack detection.
#  Uses XGBoost as primary implementation, falls back to sklearn if unavailable.
#
#  Accuracy: ~96.6% on 11-class synthetic dataset
# =============================================================================

import os
import sys
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (accuracy_score, classification_report,
                              precision_score, recall_score, f1_score)

try:
    import xgboost as xgb
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False
    from sklearn.ensemble import GradientBoostingClassifier
    print("[WARNING] xgboost not found — using sklearn GradientBoostingClassifier")

try:
    import shap
    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import FEATURE_NAMES, FAULT_LABELS, N_CLASSES, MODEL_DIR


class GradientBoostingModel:
    MODEL_NAME = "gradient_boosting"

    def __init__(self):
        self.scaler    = MinMaxScaler()
        self.trained   = False
        self.explainer = None
        self._build_model()

    def _build_model(self):
        if _XGB_AVAILABLE:
            self.model = xgb.XGBClassifier(
                n_estimators      = 400,
                max_depth         = 6,
                learning_rate     = 0.05,
                subsample         = 0.8,
                colsample_bytree  = 0.8,
                gamma             = 0.1,
                reg_alpha         = 0.1,
                reg_lambda        = 1.0,
                min_child_weight  = 2,
                eval_metric       = "mlogloss",
                use_label_encoder = False,
                n_jobs            = -1,
                random_state      = 42,
            )
        else:
            self.model = GradientBoostingClassifier(
                n_estimators  = 200,
                max_depth     = 5,
                learning_rate = 0.05,
                subsample     = 0.8,
                random_state  = 42,
            )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray = None, y_val: np.ndarray = None):
        X_s = self.scaler.fit_transform(X_train)
        if _XGB_AVAILABLE and X_val is not None:
            eval_set = [(X_s, y_train), (self.scaler.transform(X_val), y_val)]
            self.model.fit(X_s, y_train, eval_set=eval_set, verbose=50)
        else:
            self.model.fit(X_s, y_train)
        self.trained = True
        if _SHAP_AVAILABLE:
            self.explainer = shap.TreeExplainer(self.model)
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

    def explain(self, x_raw: np.ndarray, top_k: int = 5):
        if not (self.trained and _SHAP_AVAILABLE and self.explainer):
            return []
        x_s      = self.scaler.transform(x_raw.reshape(1, -1))
        pred_cls = int(self.model.predict(x_s)[0])
        sv       = self.explainer.shap_values(x_s)
        shap_vec = sv[pred_cls][0]
        top_idx  = np.argsort(np.abs(shap_vec))[::-1][:top_k]
        return [
            {"feature": FEATURE_NAMES[i],
             "shap_value": round(float(shap_vec[i]), 4),
             "raw_value": round(float(x_raw[i]), 4)}
            for i in top_idx
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
        }
        if verbose:
            print(f"\n{'='*65}")
            print(f"  Gradient Boosting — Test Accuracy: {acc*100:.2f}%")
            print(f"{'='*65}")
            print(classification_report(
                y_test, preds,
                target_names=[FAULT_LABELS[i] for i in range(N_CLASSES)],
                digits=3,
            ))
        return metrics

    def save(self, path: str = MODEL_DIR):
        os.makedirs(path, exist_ok=True)
        if _XGB_AVAILABLE:
            self.model.save_model(os.path.join(path, f"{self.MODEL_NAME}_model.json"))
        else:
            joblib.dump(self.model, os.path.join(path, f"{self.MODEL_NAME}_model.pkl"))
        joblib.dump(self.scaler, os.path.join(path, f"{self.MODEL_NAME}_scaler.pkl"))
        print(f"[GradientBoosting] Saved → {path}/")

    def load(self, path: str = MODEL_DIR):
        scaler_path = os.path.join(path, f"{self.MODEL_NAME}_scaler.pkl")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler not found: {scaler_path}")
        self.scaler = joblib.load(scaler_path)
        self._build_model()
        if _XGB_AVAILABLE:
            self.model.load_model(os.path.join(path, f"{self.MODEL_NAME}_model.json"))
        else:
            self.model = joblib.load(os.path.join(path, f"{self.MODEL_NAME}_model.pkl"))
        self.trained = True
        if _SHAP_AVAILABLE:
            self.explainer = shap.TreeExplainer(self.model)
        print(f"[GradientBoosting] Loaded from {path}/")
        return self

    def feature_importance(self):
        if not self.trained:
            return []
        imp   = self.model.feature_importances_
        items = [{"feature": FEATURE_NAMES[i], "importance": round(float(imp[i]), 4)}
                 for i in range(len(FEATURE_NAMES))]
        return sorted(items, key=lambda x: x["importance"], reverse=True)