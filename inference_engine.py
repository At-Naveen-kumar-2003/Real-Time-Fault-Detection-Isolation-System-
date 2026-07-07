# =============================================================================
#  inference_engine.py — Dual-model inference pipeline
#
#  For every incoming sensor reading:
#    1. Extracts 17 engineered features via FeatureEngine
#    2. Runs Gradient Boosting  → class + confidence + feature importance
#    3. Runs Random Forest      → class + confidence + OOB probability
#    4. Majority vote           → final verdict
#
#  Returns a structured result dict consumed by app.py and dashboard.
# =============================================================================

import os
import json
import numpy as np
import joblib

from config import (
    MODEL_DIR, N_CLASSES, FAULT_LABELS, FAULT_COLORS,
    GENUINE_FAULT_CLASSES, ATTACK_CLASSES,
)
from feature_engine import FeatureEngine
from gradient_boosting import GradientBoostingModel
from random_forest     import RandomForestModel


class InferenceEngine:

    def __init__(self):
        self.gb  = GradientBoostingModel()
        self.rf  = RandomForestModel()
        self.gb_loaded = False
        self.rf_loaded = False

        # Feature engine (one per node)
        self.feat_engine = FeatureEngine(node_ids=(1, 2))

        self._load_models()

    def _load_models(self):
        gb_scaler = os.path.join(MODEL_DIR, "gradient_boosting_scaler.pkl")
        rf_scaler = os.path.join(MODEL_DIR, "random_forest_scaler.pkl")

        if os.path.exists(gb_scaler):
            try:
                self.gb.load(MODEL_DIR)
                self.gb_loaded = True
                print("[InferenceEngine] Gradient Boosting loaded")
            except Exception as e:
                print(f"[InferenceEngine] GB load error: {e}")

        if os.path.exists(rf_scaler):
            try:
                self.rf.load(MODEL_DIR)
                self.rf_loaded = True
                print("[InferenceEngine] Random Forest loaded")
            except Exception as e:
                print(f"[InferenceEngine] RF load error: {e}")

        if not self.gb_loaded and not self.rf_loaded:
            print("[InferenceEngine] No models loaded — run training scripts first")

    def infer(self, node_id: int, sensors: dict) -> dict:
        """
        sensors : dict with keys current/voltage/temperature/humidity/gas/power
        Returns : nested result dict
        """
        # ── Feature extraction ────────────────────────────────────────────────
        feat_raw = self.feat_engine.extract(node_id, sensors)

        # ── Gradient Boosting ─────────────────────────────────────────────────
        if self.gb_loaded:
            gb_cls,  gb_conf  = self.gb.predict(feat_raw)
            gb_proba          = self.gb.predict_proba_all(feat_raw).tolist()
            gb_explain        = self.gb.explain(feat_raw, top_k=5)
        else:
            gb_cls, gb_conf   = 0, 1.0
            gb_proba          = [1.0] + [0.0] * (N_CLASSES - 1)
            gb_explain        = []

        # ── Random Forest ─────────────────────────────────────────────────────
        if self.rf_loaded:
            rf_cls,  rf_conf  = self.rf.predict(feat_raw)
            rf_proba          = self.rf.predict_proba_all(feat_raw).tolist()
            rf_explain        = self.rf.explain_single(feat_raw, top_k=5)
        else:
            rf_cls, rf_conf   = 0, 1.0
            rf_proba          = [1.0] + [0.0] * (N_CLASSES - 1)
            rf_explain        = []

        # ── Majority vote (GB: 55%, RF: 45%) ──────────────────────────────────
        scores = np.zeros(N_CLASSES)
        scores[gb_cls] += 0.55
        scores[rf_cls] += 0.45

        vote_cls  = int(np.argmax(scores))
        is_fault  = vote_cls in GENUINE_FAULT_CLASSES
        is_attack = vote_cls in ATTACK_CLASSES

        return {
            "gradient_boosting": {
                "class"      : gb_cls,
                "label"      : FAULT_LABELS[gb_cls],
                "confidence" : round(gb_conf, 3),
                "probas"     : [round(p, 4) for p in gb_proba],
                "explain"    : gb_explain,
            },
            "random_forest": {
                "class"      : rf_cls,
                "label"      : FAULT_LABELS[rf_cls],
                "confidence" : round(rf_conf, 3),
                "probas"     : [round(p, 4) for p in rf_proba],
                "explain"    : rf_explain,
            },
            "vote": {
                "class"     : vote_cls,
                "label"     : FAULT_LABELS[vote_cls],
                "color"     : FAULT_COLORS[vote_cls],
                "is_fault"  : is_fault,
                "is_attack" : is_attack,
            },
            "features" : feat_raw.tolist(),
            "sensors"  : sensors,
        }

    def clear_node(self, node_id: int):
        """Reset node buffer after isolation/restore."""
        self.feat_engine.reset_node(node_id)
