# Real-Time Fault Detection and Cyber Attack Classification in IoT Sensor Nodes Using Edge AI

An edge computing system that detects **11 classes of faults and cyber attacks** in real-time using two ESP32 sensor nodes and a Raspberry Pi 4 running dual machine learning models (Gradient Boosting + Random Forest).

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Raspberry Pi 4                              │
│                                                                 │
│   Mosquitto MQTT Broker (port 1883)                            │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │  app.py — Flask + SocketIO Dashboard (port 5000)        │  │
│   │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │  │
│   │  │  Feature Eng │  │ Gradient     │  │ Random       │  │  │
│   │  │  17 features │→ │ Boosting     │  │ Forest       │  │  │
│   │  └──────────────┘  │ ~97% acc     │  │ ~95% acc     │  │  │
│   │                    └──────────────┘  └──────────────┘  │  │
│   │                           ↓ Majority Vote               │  │
│   │                    Final Classification (11 classes)    │  │
│   └─────────────────────────────────────────────────────────┘  │
│                                                                 │
│   GPIO22 → Active Buzzer                                       │
└────────────────────────────────┬────────────────────────────────┘
                                 │ WiFi / MQTT
              ┌──────────────────┼──────────────────┐
              │                  │                  │
   ┌──────────▼──────┐  ┌────────▼────────┐  ┌─────▼───────────┐
   │   ESP32 Node 1  │  │  ESP32 Node 2   │  │  ESP32 Node 3   │
   │   INA219 (I2C)  │  │  ACS712 (ADC)   │  │  Attacker Node  │
   │   DHT11, MQ-135 │  │  DHT11, MQ-135  │  │  Injects faults │
   │   Relay, LED    │  │  Relay, LED     │  │  for testing    │
   └─────────────────┘  └─────────────────┘  └─────────────────┘
```

---

## Features

- **11-class detection** — Normal + 5 genuine faults + 5 cyber attacks
- **Dual ML models** — Gradient Boosting (97.78%) + Random Forest (94.29%)
- **17 engineered features** — raw sensors, deltas, rolling std, physics error, z-scores
- **Real-time dashboard** — live sensor gauges, model predictions, alert log
- **Relay isolation** — automatic power cut on fault/attack detection
- **Buzzer alerts** — different patterns for faults vs cyber attacks
- **SQLite database** — all readings stored for history and export
- **CSV/Excel export** — download historical data from dashboard
- **Node 3 attacker** — dedicated ESP32 for injecting simulated attacks

---

## Attack Classes

| Class | Label | Type | Primary Feature |
|-------|-------|------|----------------|
| 0 | Normal | — | All normal |
| 1 | Overcurrent | Genuine Fault | current > 4.5A |
| 2 | Overvoltage | Genuine Fault | voltage > 13V |
| 3 | Undervoltage | Genuine Fault | voltage < 9V |
| 4 | Overtemperature | Genuine Fault | temp > 60°C |
| 5 | Gas Leak | Genuine Fault | gas > 350ppm |
| 6 | Spoofing Attack | Cyber Attack | huge delta_current |
| 7 | Replay Attack | Cyber Attack | roll_std ≈ 0 |
| 8 | Gradual Drift | Cyber Attack | slow ramp |
| 9 | Pulse Attack | Cyber Attack | brief spikes |
| 10 | Physics Attack | Cyber Attack | P ≠ V×I |

---

## Hardware

### Node 1 — INA219 (I2C)
| Component | ESP32 Pin |
|-----------|-----------|
| INA219 SDA | GPIO21 |
| INA219 SCL | GPIO22 |
| INA219 VCC | 3.3V |
| DHT11 DATA | GPIO4 |
| MQ-135 AOUT | GPIO32 |
| Relay IN | GPIO26 |
| LED | GPIO2 |
| Voltage Divider (33kΩ+10kΩ) | GPIO35 |

### Node 2 — ACS712 (ADC)
| Component | ESP32 Pin |
|-----------|-----------|
| ACS712 AOUT | GPIO34 |
| Voltage Divider (33kΩ+10kΩ) | GPIO27 |
| DHT11 DATA | GPIO4 |
| MQ-135 AOUT | GPIO32 |
| Relay IN | GPIO26 |
| LED | GPIO2 |

### Raspberry Pi
| Component | Pin |
|-----------|-----|
| Active Buzzer (+) | GPIO22 (BCM) |
| Active Buzzer (−) | GND |

---

## Project Structure

```
Edge-AI-Fault-Detection/
│
├── app.py                      ← Main server (Flask + MQTT + ML)
├── config.py                   ← All settings (topics, classes, pins)
├── feature_engine.py           ← 17-feature real-time extractor
├── inference_engine.py         ← Dual-model inference pipeline
├── mqtt_handler.py             ← MQTT client (pub/sub)
├── gpio_controller.py          ← Buzzer driver
├── generate_dataset.py         ← Synthetic 11-class dataset generator
├── gradient_boosting.py        ← Gradient Boosting model class
├── random_forest.py            ← Random Forest model class
├── train_gradient_boosting.py  ← Train GB model
├── train_random_forest.py      ← Train RF model
├── compare_models.py           ← Algorithm comparison + graphs
├── requirements.txt
├── setup.sh                    ← One-shot Pi setup script
├── README.md
├── LICENSE
├── .gitignore
│
├── firmware/
│   ├── Esp32_Node1.ino         ← Node 1 firmware (INA219)
│   ├── Esp32_Node2.ino         ← Node 2 firmware (ACS712)
│   └── Esp32_Node3.ino         ← Attacker node firmware
│
└── saved_models/               ← Created after training
    ├── gradient_boosting_model.json
    ├── gradient_boosting_scaler.pkl
    ├── random_forest_model.pkl
    ├── random_forest_scaler.pkl
    ├── model_accuracy.json
    ├── model_metrics.json
    ├── training_data.csv
    ├── training_data.xlsx
    └── graphs/
```

---

## Installation

### Step 1 — Clone and setup
```bash
git clone https://github.com/yourusername/Edge-AI-Fault-Detection.git
cd Edge-AI-Fault-Detection
chmod +x setup.sh && ./setup.sh
```

### Step 2 — Manual setup (if setup.sh fails)
```bash
# Install Mosquitto
sudo apt install mosquitto mosquitto-clients -y
echo "listener 1883 0.0.0.0" | sudo tee -a /etc/mosquitto/mosquitto.conf
echo "allow_anonymous true"   | sudo tee -a /etc/mosquitto/mosquitto.conf
sudo systemctl restart mosquitto

# Create venv
python3 -m venv ~/edgeai_env --system-site-packages
source ~/edgeai_env/bin/activate

# Install packages
pip install -r requirements.txt --no-cache-dir
```

### Step 3 — Generate dataset and train models
```bash
source ~/edgeai_env/bin/activate
python3 generate_dataset.py
python3 train_gradient_boosting.py
python3 train_random_forest.py
```

### Step 4 — Flash ESP32 nodes
In each `.ino` file set your Pi's IP:
```cpp
#define MQTT_BROKER "10.48.223.2"   // ← your Pi IP from: hostname -I
```
Flash via Arduino IDE.

### Step 5 — Run
```bash
python3 app.py
```
Open: `http://<PI_IP>:5000`

---

## Model Comparison

Run to generate 7 graphs + HTML report:
```bash
python3 compare_models.py
# Open: saved_models/comparison_report.html
```

| Metric | Gradient Boosting | Random Forest |
|--------|------------------|---------------|
| Accuracy | **97.78%** | 94.29% |
| F1 Macro | **97.1%** | 92.3% |
| Training Time | ~3 min | ~1 min |
| Inference | Single sample | Single sample |
| Explainability | Feature importance | Feature importance |

---

## MQTT Topics

| Direction | Topic | Description |
|-----------|-------|-------------|
| ESP32 → Pi | `sensors/node1` | Node 1 sensor data (1 Hz) |
| ESP32 → Pi | `sensors/node2` | Node 2 sensor data (1 Hz) |
| ESP32 → Pi | `sensors/node3` | Node 3 attack injection log |
| Pi → ESP32 | `control/relay/node1` | Relay cut/restore Node 1 |
| Pi → ESP32 | `control/relay/node2` | Relay cut/restore Node 2 |

---

## Dashboard URLs

| URL | Description |
|-----|-------------|
| `http://<PI_IP>:5000` | Live dashboard |
| `http://<PI_IP>:5000/api/state` | Full JSON state |
| `http://<PI_IP>:5000/api/stats` | Statistics |
| `http://<PI_IP>:5000/api/history/node1` | Node 1 history |
| `http://<PI_IP>:5000/api/export/csv` | Export CSV |
| `http://<PI_IP>:5000/api/relay` | Relay control (POST) |

---

## Results

- Gradient Boosting: **97.78% accuracy** on 11-class synthetic dataset
- Random Forest: **94.29% accuracy** on 11-class synthetic dataset
- Majority vote combines both for robust final decision
- Real-time inference: < 5ms per sample on Raspberry Pi 4

---

## License

MIT License — see [LICENSE](LICENSE)
