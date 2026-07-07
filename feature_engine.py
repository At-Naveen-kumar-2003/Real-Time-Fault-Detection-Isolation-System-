# =============================================================================
#  feature_engine.py — Real-time 17-feature extractor
#
#  Called on every incoming MQTT sensor message.
#  Maintains per-node rolling buffers so delta, rolling-std and z-score
#  are computed against recent history — not just the current sample.
#
#  Features (17 total):
#    Raw       : current, voltage, temperature, gas, power         (5)
#    Delta     : delta_current, delta_voltage, delta_temp, delta_gas (4)
#    Rolling   : roll_std_current, roll_std_voltage,               (4)
#                roll_std_temp, roll_std_gas
#    Physics   : physics_error = |P - V*I| / (V*I)                (1)
#    Z-scores  : z_current, z_voltage                              (2)
#    Cross-node: cross_delta_temp                                   (1)
# =============================================================================

import numpy as np
import collections
from config import FEATURE_NAMES


class NodeBuffer:
    """Rolling history buffer for one ESP32 node."""

    WINDOW       = 20   # rolling window size
    BASELINE_LEN = 30   # samples used to build z-score baseline

    def __init__(self):
        self.history  = collections.deque(maxlen=self.WINDOW)
        self.baseline = {
            "current": collections.deque(maxlen=self.BASELINE_LEN),
            "voltage": collections.deque(maxlen=self.BASELINE_LEN),
        }

    def extract(self, sensors: dict, other_temp: float = None) -> np.ndarray:
        curr = float(sensors.get("current",     2.0) or 2.0)
        volt = float(sensors.get("voltage",    11.0) or 11.0)
        temp = float(sensors.get("temperature",32.0) or 32.0)
        gas  = float(sensors.get("gas",       100.0) or 100.0)
        pwr  = float(sensors.get("power", curr * volt) or curr * volt)

        # ── Deltas ────────────────────────────────────────────────────────────
        if self.history:
            prev   = self.history[-1]
            d_curr = curr - prev[0]
            d_volt = volt - prev[1]
            d_temp = temp - prev[2]
            d_gas  = gas  - prev[3]
        else:
            d_curr = d_volt = d_temp = d_gas = 0.0

        self.history.append((curr, volt, temp, gas))

        # ── Rolling std ───────────────────────────────────────────────────────
        h = np.array(self.history)
        rs_curr = float(h[:, 0].std()) if len(h) > 1 else 0.0
        rs_volt = float(h[:, 1].std()) if len(h) > 1 else 0.0
        rs_temp = float(h[:, 2].std()) if len(h) > 1 else 0.0
        rs_gas  = float(h[:, 3].std()) if len(h) > 1 else 0.0

        # ── Physics error ─────────────────────────────────────────────────────
        pwr_expected  = curr * volt
        physics_error = abs(pwr - pwr_expected) / max(pwr_expected, 0.1)

        # ── Z-scores ──────────────────────────────────────────────────────────
        self.baseline["current"].append(curr)
        self.baseline["voltage"].append(volt)
        bc = np.array(self.baseline["current"])
        bv = np.array(self.baseline["voltage"])
        z_curr = (curr - bc.mean()) / max(bc.std(), 0.01)
        z_volt = (volt - bv.mean()) / max(bv.std(), 0.01)

        # ── Cross-node temperature delta ──────────────────────────────────────
        cross = abs(temp - other_temp) if other_temp is not None else abs(temp - 32.0)

        return np.array([
            curr,   volt,   temp,   gas,   pwr,
            d_curr, d_volt, d_temp, d_gas,
            rs_curr, rs_volt, rs_temp, rs_gas,
            physics_error,
            z_curr, z_volt,
            cross,
        ], dtype=np.float32)

    def reset(self):
        self.history.clear()
        self.baseline["current"].clear()
        self.baseline["voltage"].clear()


class FeatureEngine:
    """Manages one NodeBuffer per node and exposes a single extract() call."""

    def __init__(self, node_ids=(1, 2)):
        self.buffers     = {nid: NodeBuffer() for nid in node_ids}
        self.latest_temp = {nid: 32.0        for nid in node_ids}

    def extract(self, node_id: int, sensors: dict) -> np.ndarray:
        """
        sensors : dict with keys current/voltage/temperature/gas/power/humidity
        Returns : (17,) numpy float32 feature vector
        """
        other_ids  = [n for n in self.buffers if n != node_id]
        other_temp = self.latest_temp[other_ids[0]] if other_ids else None

        feats = self.buffers[node_id].extract(sensors, other_temp)
        self.latest_temp[node_id] = float(sensors.get("temperature", 32.0) or 32.0)
        return feats

    def reset_node(self, node_id: int):
        """Call when a node is restored after isolation."""
        if node_id in self.buffers:
            self.buffers[node_id].reset()
