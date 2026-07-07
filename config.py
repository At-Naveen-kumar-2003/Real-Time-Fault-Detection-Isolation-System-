# =============================================================================
#  config.py — Central configuration for the Edge AI fault detection system
# =============================================================================

# ── MQTT ──────────────────────────────────────────────────────────────────────
MQTT_BROKER = "localhost"       # ← Change to your Raspberry Pi IP address
MQTT_PORT   = 1883

TOPIC_SENSOR_N1 = "sensors/node1"
TOPIC_SENSOR_N2 = "sensors/node2"
TOPIC_NODE3     = "sensors/node3"

TOPIC_RELAY_N1  = "control/relay/node1"
TOPIC_RELAY_N2  = "control/relay/node2"

# ── GPIO ──────────────────────────────────────────────────────────────────────
BUZZER_PIN = 22

# ── SENSOR KEYS ───────────────────────────────────────────────────────────────
SENSOR_KEYS = ["current", "voltage", "temperature", "humidity", "gas", "power"]

FEATURE_NAMES = [
    "current", "voltage", "temperature", "gas", "power",
    "delta_current", "delta_voltage", "delta_temp", "delta_gas",
    "roll_std_current", "roll_std_voltage", "roll_std_temp", "roll_std_gas",
    "physics_error",
    "z_current", "z_voltage",
    "cross_delta_temp",
]
N_FEATURES = len(FEATURE_NAMES)   # 17

# ── FAULT / ATTACK CLASSES ────────────────────────────────────────────────────
FAULT_LABELS = {
    0:  "Normal",
    1:  "Overcurrent",
    2:  "Overvoltage",
    3:  "Undervoltage",
    4:  "Overtemperature",
    5:  "Gas Leak",
    6:  "Spoofing Attack",
    7:  "Replay Attack",
    8:  "Gradual Drift",
    9:  "Pulse Attack",
    10: "Physics Attack",
}
N_CLASSES = len(FAULT_LABELS)   # 11

# FIX: compare_models.py imports ATTACK_LABELS — this alias makes both names work
ATTACK_LABELS = FAULT_LABELS

FAULT_COLORS = {
    0:  "#22c55e",  1:  "#ef4444",  2:  "#f97316",  3:  "#eab308",
    4:  "#a855f7",  5:  "#78716c",  6:  "#dc2626",  7:  "#b45309",
    8:  "#0284c7",  9:  "#db2777",  10: "#7c3aed",
}

GENUINE_FAULT_CLASSES = {1, 2, 3, 4, 5}
ATTACK_CLASSES        = {6, 7, 8, 9, 10}

# ── HARDWARE THRESHOLDS ───────────────────────────────────────────────────────
THRESHOLDS = {
    "current_high"  : 4.5,
    "voltage_high"  : 13.0,
    "voltage_low"   : 9.0,
    "temp_high"     : 60.0,
    "gas_high"      : 350.0,
    "power_high"    : 3500.0,
}

# ── MODEL SETTINGS ────────────────────────────────────────────────────────────
WINDOW_SIZE     = 10
FL_INTERVAL_SEC = 30
FL_LOCAL_EPOCHS = 5
FL_LR           = 0.001

MODEL_DIR = "saved_models"

# ── EMAIL ALERTS (optional — set EMAIL_ENABLED=True and fill credentials) ─────
EMAIL_ENABLED = False
SMTP_HOST     = "smtp.gmail.com"
SMTP_PORT     = 587
SMTP_USER     = "your_email@gmail.com"
SMTP_PASS     = "your_app_password"
EMAIL_TO      = "recipient@gmail.com"
