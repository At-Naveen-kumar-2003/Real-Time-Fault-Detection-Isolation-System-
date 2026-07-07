# =============================================================================
#  mqtt_handler.py — MQTT client for the Raspberry Pi
#
#  Subscribes:
#    sensors/node1   — Node 1 (INA219)  sensor data  (1 Hz)
#    sensors/node2   — Node 2 (ACS712)  sensor data  (1 Hz)
#    sensors/node3   — Node 3 (attacker) injection log
#
#  Publishes:
#    control/relay/node1   {"action": "cut" | "restore"}
#    control/relay/node2   {"action": "cut" | "restore"}
# =============================================================================

import json
import threading
import time

try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False
    print("[MQTT] paho-mqtt not installed — run: pip install paho-mqtt")

from config import (
    MQTT_BROKER, MQTT_PORT,
    TOPIC_SENSOR_N1, TOPIC_SENSOR_N2, TOPIC_NODE3,
    TOPIC_RELAY_N1,  TOPIC_RELAY_N2,
)


class MQTTHandler:

    def __init__(self):
        self._callbacks  = {}
        self._lock       = threading.Lock()
        self._connected  = False

        if not MQTT_AVAILABLE:
            return

        self._client = mqtt.Client(
            client_id   = "pi_edge_server",
            clean_session = True,
        )
        self._client.on_connect    = self._on_connect
        self._client.on_message    = self._on_message
        self._client.on_disconnect = self._on_disconnect

    # ── Internal callbacks ────────────────────────────────────────────────────

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            subs = [
                (TOPIC_SENSOR_N1, 0),
                (TOPIC_SENSOR_N2, 0),
                (TOPIC_NODE3,     0),
            ]
            client.subscribe(subs)
            self._connected = True
            print(f"[MQTT] Connected to {MQTT_BROKER}:{MQTT_PORT} "
                  f"— subscribed to {len(subs)} topics")
        else:
            print(f"[MQTT] Connection refused (rc={rc})")

    def _on_message(self, client, userdata, msg):
        topic = msg.topic
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
        except Exception:
            payload = msg.payload.decode("utf-8", errors="replace")

        with self._lock:
            cbs = list(self._callbacks.get(topic, []))

        for cb in cbs:
            threading.Thread(
                target=cb, args=(topic, payload), daemon=True
            ).start()

    def _on_disconnect(self, client, userdata, rc):
        self._connected = False
        print(f"[MQTT] Disconnected (rc={rc}) — reconnecting …")
        while True:
            try:
                self._client.reconnect()
                break
            except Exception:
                time.sleep(5)

    # ── Public API ────────────────────────────────────────────────────────────

    def register(self, topic: str, callback):
        """Register callback(topic, payload_dict) for a MQTT topic."""
        with self._lock:
            self._callbacks.setdefault(topic, []).append(callback)

    def publish(self, topic: str, payload: dict, qos: int = 1):
        if not MQTT_AVAILABLE or not self._connected:
            return
        self._client.publish(topic, json.dumps(payload), qos=qos)

    def relay_cut(self, node_id: int):
        topic = TOPIC_RELAY_N1 if node_id == 1 else TOPIC_RELAY_N2
        self.publish(topic, {"action": "cut"})
        print(f"[MQTT] Relay CUT   → Node {node_id}")

    def relay_restore(self, node_id: int):
        topic = TOPIC_RELAY_N1 if node_id == 1 else TOPIC_RELAY_N2
        self.publish(topic, {"action": "restore"})
        print(f"[MQTT] Relay RESTORE → Node {node_id}")

    def start(self):
        if not MQTT_AVAILABLE:
            print("[MQTT] Not available — skipping MQTT connection")
            return
        try:
            self._client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
            self._client.loop_start()
        except Exception as e:
            print(f"[MQTT] Failed to connect: {e}")
            print("[MQTT] Check: sudo systemctl start mosquitto")
            print("[MQTT] Check: listener 1883 0.0.0.0 in /etc/mosquitto/mosquitto.conf")

    def stop(self):
        if MQTT_AVAILABLE:
            self._client.loop_stop()
            self._client.disconnect()

    @property
    def connected(self) -> bool:
        return self._connected
