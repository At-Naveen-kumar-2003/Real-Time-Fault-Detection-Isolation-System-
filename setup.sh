#!/bin/bash
# =============================================================================
#  setup.sh — One-shot setup script for Raspberry Pi
#  Run: chmod +x setup.sh && ./setup.sh
# =============================================================================

echo "============================================="
echo "  Edge AI Fault Detection — Pi Setup"
echo "============================================="

# ── Step 1: Install Mosquitto MQTT broker ─────────────────────────────────────
echo "[1/6] Installing Mosquitto..."
sudo apt-get update -qq
sudo apt-get install -y mosquitto mosquitto-clients

# Allow external connections
CONF="/etc/mosquitto/mosquitto.conf"
if ! grep -q "listener 1883 0.0.0.0" "$CONF"; then
    echo "listener 1883 0.0.0.0" | sudo tee -a "$CONF"
    echo "allow_anonymous true"  | sudo tee -a "$CONF"
fi
sudo systemctl enable mosquitto
sudo systemctl restart mosquitto
echo "  Mosquitto running on port 1883"

# ── Step 2: Create virtual environment ────────────────────────────────────────
echo "[2/6] Creating Python virtual environment..."
python3 -m venv ~/edgeai_env --system-site-packages
source ~/edgeai_env/bin/activate

# ── Step 3: Install Python packages ───────────────────────────────────────────
echo "[3/6] Installing Python packages..."
pip install --upgrade pip --quiet
pip install --no-cache-dir \
    numpy pandas scikit-learn joblib \
    matplotlib openpyxl \
    flask flask-socketio simple-websocket \
    paho-mqtt eventlet

echo "  Python packages installed"

# ── Step 4: Add venv to .bashrc ───────────────────────────────────────────────
echo "[4/6] Adding venv auto-activation to .bashrc..."
if ! grep -q "edgeai_env" ~/.bashrc; then
    echo "source ~/edgeai_env/bin/activate" >> ~/.bashrc
fi

# ── Step 5: Create saved_models directory ─────────────────────────────────────
echo "[5/6] Creating saved_models directory..."
mkdir -p saved_models/graphs
mkdir -p saved_models/fl_graphs

# ── Step 6: Create systemd service ────────────────────────────────────────────
echo "[6/6] Creating systemd service..."
SERVICE="/etc/systemd/system/edgeai.service"
sudo tee "$SERVICE" > /dev/null << EOF
[Unit]
Description=Edge AI Fault Detection Server
After=network.target mosquitto.service

[Service]
User=pi
WorkingDirectory=$(pwd)
ExecStart=/home/pi/edgeai_env/bin/python3 $(pwd)/app.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable edgeai
echo "  Service created (start with: sudo systemctl start edgeai)"

echo ""
echo "============================================="
echo "  Setup Complete!"
echo "============================================="
echo ""
echo "  Next steps:"
echo "  1. source ~/edgeai_env/bin/activate"
echo "  2. python3 generate_dataset.py"
echo "  3. python3 train_gradient_boosting.py"
echo "  4. python3 train_random_forest.py"
echo "  5. python3 app.py"
echo "  6. Open: http://$(hostname -I | awk '{print $1}'):5000"
echo ""
