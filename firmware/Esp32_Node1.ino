#include <WiFi.h>
#include <PubSubClient.h>
#include <DHT.h>
#include <ArduinoJson.h>
#include <Wire.h>
#include <Adafruit_INA219.h>

// ── CONFIGURATION — CHANGE THESE ─────────────────────────────────────────────
#define NODE_ID         1
#define WIFI_SSID       "YOUR_WIFI_SSID"
#define WIFI_PASSWORD   "YOUR_WIFI_PASSWORD"
#define MQTT_BROKER     "192.168.X.X"     // ← Replace with your Pi IP
#define MQTT_PORT       1883

// ── PIN DEFINITIONS ──────────────────────────────────────────────────────────
// INA219 uses I2C: SDA=GPIO21, SCL=GPIO22 (ESP32 default I2C pins)
#define PIN_DHT         4     // DHT11 DATA
#define PIN_GAS         32    // MQ-135 AOUT
#define PIN_RELAY       26    // Relay IN  (HIGH = cut power to load)
#define PIN_LED         2     // Status LED

// ── TIMING ───────────────────────────────────────────────────────────────────
#define PUBLISH_INTERVAL_MS  1000     // 1 Hz publish rate

// ── OBJECTS ──────────────────────────────────────────────────────────────────
Adafruit_INA219 ina219;               // default address 0x40
DHT             dht(PIN_DHT, DHT11);
WiFiClient      wifiClient;
PubSubClient    mqtt(wifiClient);

// ── MQTT TOPICS ──────────────────────────────────────────────────────────────
char TOPIC_SENSOR[32];    // sensors/node1  — we publish here
char TOPIC_RELAY[32];     // control/relay/node1 — Pi sends commands here

// ── STATE ─────────────────────────────────────────────────────────────────────
bool          relayTripped = false;
unsigned long lastPublish  = 0;
float         lastTemp     = 30.0f;
float         lastHumi     = 50.0f;

// ══════════════════════════════════════════════════════════════════════════════
// SETUP
// ══════════════════════════════════════════════════════════════════════════════
void setup() {
  Serial.begin(115200);
  delay(500);
  Serial.println("\n=== Edge AI Node 1 — INA219 Firmware ===");

  sprintf(TOPIC_SENSOR, "sensors/node%d",       NODE_ID);
  sprintf(TOPIC_RELAY,  "control/relay/node%d",  NODE_ID);

  // GPIO
  pinMode(PIN_RELAY, OUTPUT);
  pinMode(PIN_LED,   OUTPUT);
  digitalWrite(PIN_RELAY, LOW);   // Power ON at startup
  digitalWrite(PIN_LED,   HIGH);  // LED ON = active

  // ADC attenuation for MQ-135 (GPIO32)
  analogSetPinAttenuation(PIN_GAS, ADC_11db);

  // INA219 init over I2C
  Wire.begin(21, 22);   // SDA=21, SCL=22 (ESP32 default — explicit is safer)
  if (!ina219.begin()) {
    Serial.println("ERROR: INA219 not found! Check wiring:");
    Serial.println("  SDA → GPIO21, SCL → GPIO22, VCC → 3.3V, GND → GND");
    Serial.println("  Also check I2C address — default is 0x40");
    // Blink LED rapidly to signal hardware error
    while (true) {
      digitalWrite(PIN_LED, !digitalRead(PIN_LED));
      delay(200);
    }
  }
  // Set measurement range — use 16V/400mA for 11V system with INA219 default shunt
  // If you need higher current range, call ina219.setCalibration_32V_2A()
  ina219.setCalibration_16V_400mA();   // most precise for low-current 11V systems
  Serial.println("INA219 initialised at 0x40 (16V/400mA mode)");

  dht.begin();
  connectWiFi();

  mqtt.setServer(MQTT_BROKER, MQTT_PORT);
  mqtt.setCallback(mqttCallback);
  connectMQTT();

  Serial.printf("Publishing to : %s\n", TOPIC_SENSOR);
  Serial.printf("Listening on  : %s\n", TOPIC_RELAY);
  Serial.println("Ready. Sending sensor data to Pi at 1 Hz.\n");
}

// ══════════════════════════════════════════════════════════════════════════════
// MAIN LOOP
// ══════════════════════════════════════════════════════════════════════════════
void loop() {
  if (!mqtt.connected()) connectMQTT();
  mqtt.loop();

  if (millis() - lastPublish >= PUBLISH_INTERVAL_MS) {
    lastPublish = millis();

    if (!relayTripped) {
      // ── Read INA219 (current + voltage) ───────────────────────────────────
      // INA219 returns bus voltage (load side) and current through shunt resistor
      float current     = readCurrent_INA219();   // Amperes
      float voltage     = readVoltage_INA219();   // Volts (bus voltage)
      float power_mw    = ina219.getPower_mW();   // INA219 internal power calc (mW)
      float power       = current * voltage;      // Recalculate in Watts for consistency

      // ── Read DHT11 ────────────────────────────────────────────────────────
      float temperature = dht.readTemperature();
      float humidity    = dht.readHumidity();
      if (!isnan(temperature)) lastTemp = temperature;
      else                     temperature = lastTemp;
      if (!isnan(humidity))    lastHumi = humidity;
      else                     humidity = lastHumi;

      // ── Read MQ-135 ───────────────────────────────────────────────────────
      float gas = readGas();

      // ── Serial debug ──────────────────────────────────────────────────────
      Serial.printf("[Node1/INA219] I=%.3fA  V=%.2fV  T=%.1f°C  H=%.0f%%  Gas=%.0fppm  P=%.2fW\n",
                    current, voltage, temperature, humidity, gas, power);

      // ── Build JSON and publish ─────────────────────────────────────────────
      // Pi's mqtt_handler.py expects these exact keys
      StaticJsonDocument<256> doc;
      doc["node_id"]     = NODE_ID;
      doc["current"]     = serialized(String(current,     3));
      doc["voltage"]     = serialized(String(voltage,     2));
      doc["temperature"] = serialized(String(temperature, 1));
      doc["humidity"]    = serialized(String(humidity,    0));
      doc["gas"]         = serialized(String(gas,         0));
      doc["power"]       = serialized(String(power,       2));

      char buf[256];
      serializeJson(doc, buf);

      bool ok = mqtt.publish(TOPIC_SENSOR, buf);
      if (!ok) Serial.println("  ✗ MQTT publish failed");

    } else {
      // Isolated — slow blink to show the node is still alive
      digitalWrite(PIN_LED, (millis() / 500) % 2);
    }
  }
}

// ══════════════════════════════════════════════════════════════════════════════
// SENSOR READING FUNCTIONS
// ══════════════════════════════════════════════════════════════════════════════

/*
 * readCurrent_INA219()
 * INA219 directly measures current through the shunt resistor.
 * Averages 10 readings for stability.
 * Returns current in Amperes (absolute value).
 *
 * IMPORTANT: If you get negative current, the INA219 is wired backwards.
 * Swap VIN+ and VIN- connections.
 */
float readCurrent_INA219() {
  float sum = 0;
  for (int i = 0; i < 10; i++) {
    sum += ina219.getCurrent_mA();
    delay(2);
  }
  float avg_mA = sum / 10.0f;
  return abs(avg_mA / 1000.0f);   // mA → A, always positive
}

/*
 * readVoltage_INA219()
 * INA219 bus voltage = voltage on the load side (after shunt resistor).
 * This is your actual supply/battery voltage seen by the load.
 * Averages 10 readings.
 */
float readVoltage_INA219() {
  float sum = 0;
  for (int i = 0; i < 10; i++) {
    sum += ina219.getBusVoltage_V();
    delay(2);
  }
  return sum / 10.0f;
}

/*
 * readGas()
 * MQ-135: averages 20 ADC samples.
 * Approximate linear mapping: ADC[200..3500] → ppm[50..900]
 */
float readGas() {
  long sum = 0;
  for (int i = 0; i < 20; i++) {
    sum += analogRead(PIN_GAS);
    delayMicroseconds(100);
  }
  float raw = sum / 20.0f;
  long ppm  = map((long)raw, 200, 3500, 50, 900);
  ppm = constrain(ppm, 50, 900);
  return (float)ppm;
}

// ══════════════════════════════════════════════════════════════════════════════
// MQTT CALLBACK — Relay commands from Raspberry Pi
// ══════════════════════════════════════════════════════════════════════════════
/*
 * Pi sends JSON to control/relay/node1:
 *   {"action": "cut"}     → HIGH on relay pin, cuts power to load
 *   {"action": "restore"} → LOW on relay pin, restores power
 */
void mqttCallback(char* topic, byte* payload, unsigned int length) {
  String topicStr = String(topic);
  String msg = "";
  for (unsigned int i = 0; i < length; i++) msg += (char)payload[i];
  Serial.printf("MQTT IN [%s]: %s\n", topic, msg.c_str());

  if (topicStr == TOPIC_RELAY) {
    StaticJsonDocument<128> doc;
    if (deserializeJson(doc, msg) != DeserializationError::Ok) {
      Serial.println("  ✗ JSON parse error on relay command");
      return;
    }
    String action = doc["action"] | "";

    if (action == "cut") {
      Serial.println("  ⛔ RELAY: Power CUT — node isolated by Pi");
      digitalWrite(PIN_RELAY, HIGH);
      digitalWrite(PIN_LED,   LOW);
      relayTripped = true;

    } else if (action == "restore") {
      Serial.println("  ✅ RELAY: Power RESTORED by Pi");
      digitalWrite(PIN_RELAY, LOW);
      digitalWrite(PIN_LED,   HIGH);
      relayTripped = false;

    } else {
      Serial.printf("  ? Unknown relay action: %s\n", action.c_str());
    }
  }
}

// ══════════════════════════════════════════════════════════════════════════════
// WiFi & MQTT CONNECTION
// ══════════════════════════════════════════════════════════════════════════════
void connectWiFi() {
  Serial.printf("Connecting to WiFi: %s", WIFI_SSID);
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 40) {
    delay(500); Serial.print("."); attempts++;
  }
  if (WiFi.status() == WL_CONNECTED) {
    Serial.printf("\nWiFi connected — IP: %s\n", WiFi.localIP().toString().c_str());
  } else {
    Serial.println("\nWiFi FAILED. Restarting...");
    delay(3000); ESP.restart();
  }
}

void connectMQTT() {
  char clientId[24];
  sprintf(clientId, "esp32_node_%d", NODE_ID);
  int retries = 0;
  while (!mqtt.connected() && retries < 5) {
    Serial.printf("MQTT connecting as [%s]...", clientId);
    if (mqtt.connect(clientId)) {
      Serial.println(" connected ✓");
      mqtt.subscribe(TOPIC_RELAY);
      Serial.printf("Subscribed: %s\n", TOPIC_RELAY);
    } else {
      Serial.printf(" failed (rc=%d). Retry in 3s\n", mqtt.state());
      delay(3000); retries++;
    }
  }
  if (!mqtt.connected()) {
    Serial.println("MQTT failed after 5 attempts. Restarting...");
    delay(1000); ESP.restart();
  }
}