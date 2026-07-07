#include <WiFi.h>
#include <PubSubClient.h>
#include <DHT.h>
#include <ArduinoJson.h>

// ── CONFIGURATION ─────────────────────────────────────────────────────────────
#define NODE_ID              2
#define WIFI_SSID       "YOUR_WIFI_SSID"
#define WIFI_PASSWORD   "YOUR_WIFI_PASSWORD"
#define MQTT_BROKER     "192.168.X.X"     // ← Replace with your Pi IP
#define MQTT_PORT            1883

// ── PIN DEFINITIONS ───────────────────────────────────────────────────────────
#define PIN_CURRENT          34
#define PIN_VOLTAGE          35    // Voltage divider → GPIO35 (input-only, safe for ADC)
#define PIN_DHT              4
#define PIN_GAS              32
#define PIN_RELAY            26
#define PIN_LED              2

// ── ACS712 CALIBRATION ────────────────────────────────────────────────────────
#define ACS712_SENSITIVITY   0.185f
#define ACS712_MIDPOINT      1.65f   // change to 2.50f if ACS712 VCC=5V

// ── VOLTAGE DIVIDER ───────────────────────────────────────────────────────────
// R1=33kΩ, R2=10kΩ
// ratio = 10/(33+10) = 0.2326
#define VOLT_DIVIDER_RATIO   0.2326f

// ── VOLTAGE ESTIMATION ────────────────────────────────────────────────────────
// Based on Node 1 INA219 reading: ~10.15V at 0.082A
// Same 11V battery supply
// V = V_NOMINAL - (I × WIRE_RESISTANCE)
// Wire resistance estimated ~2Ω for short wires
#define V_NOMINAL            10.20f   // base voltage from battery (matches Node1)
#define WIRE_RESISTANCE      2.0f     // estimated wire+connection resistance in Ω

// ── TIMING ────────────────────────────────────────────────────────────────────
#define PUBLISH_INTERVAL_MS  1000

// ── LED ───────────────────────────────────────────────────────────────────────
#define LED_ON   digitalWrite(PIN_LED, LOW)
#define LED_OFF  digitalWrite(PIN_LED, HIGH)

// ── OBJECTS ───────────────────────────────────────────────────────────────────
DHT          dht(PIN_DHT, DHT11);
WiFiClient   wifiClient;
PubSubClient mqtt(wifiClient);

// ── MQTT TOPICS ───────────────────────────────────────────────────────────────
char TOPIC_SENSOR[32];
char TOPIC_RELAY[32];

// ── STATE ─────────────────────────────────────────────────────────────────────
bool          relayTripped = false;
unsigned long lastPublish  = 0;
unsigned long lastBlink    = 0;
bool          ledState     = false;
float         lastTemp     = 30.0f;
float         lastHumi     = 50.0f;

// ── FUNCTION DECLARATIONS ─────────────────────────────────────────────────────
float readCurrent_ACS712();
float readVoltage();
float readGas();
void  connectWiFi();
void  connectMQTT();
void  mqttCallback(char* topic, byte* payload, unsigned int length);

// ══════════════════════════════════════════════════════════════════════════════
// SETUP
// ══════════════════════════════════════════════════════════════════════════════
void setup() {
  Serial.begin(115200);
  delay(100);
  Serial.println("\n=== Edge AI Node 2 — ACS712 Firmware ===");

  sprintf(TOPIC_SENSOR, "sensors/node%d",      NODE_ID);
  sprintf(TOPIC_RELAY,  "control/relay/node%d", NODE_ID);

  // ── GPIO ────────────────────────────────────────────────────────────────────
  pinMode(PIN_RELAY, OUTPUT);
  pinMode(PIN_LED,   OUTPUT);
  digitalWrite(PIN_RELAY, LOW);  // Power ON at startup
  LED_OFF;                       // LED OFF at boot

  // ── ADC ─────────────────────────────────────────────────────────────────────
  analogSetPinAttenuation(PIN_CURRENT, ADC_11db);  // GPIO34
  analogSetPinAttenuation(PIN_VOLTAGE, ADC_11db);  // GPIO35
  analogSetPinAttenuation(PIN_GAS,     ADC_11db);  // GPIO32

  // ── DHT ─────────────────────────────────────────────────────────────────────
  dht.begin();
  delay(2000);

  // ── WiFi + MQTT ─────────────────────────────────────────────────────────────
  connectWiFi();
  mqtt.setServer(MQTT_BROKER, MQTT_PORT);
  mqtt.setCallback(mqttCallback);
  connectMQTT();

  // System ready — start LED blink
  Serial.printf("Publishing to  : %s\n", TOPIC_SENSOR);
  Serial.printf("Listening on   : %s\n", TOPIC_RELAY);
  Serial.printf("ACS712 midpoint: %.2fV  sensitivity: %.3fV/A\n",
                ACS712_MIDPOINT, ACS712_SENSITIVITY);
  Serial.printf("V nominal      : %.2fV  wire R: %.1fΩ\n",
                V_NOMINAL, WIRE_RESISTANCE);
  Serial.println("Ready — sending 1 Hz to Pi\n");
}

// ══════════════════════════════════════════════════════════════════════════════
// MAIN LOOP
// ══════════════════════════════════════════════════════════════════════════════
void loop() {
  if (!mqtt.connected()) connectMQTT();
  mqtt.loop();

  // ── LED blink when load is ON ──────────────────────────────────────────────
  // Blinks at 1Hz when running normally
  // Stays OFF when relay has cut power
  if (!relayTripped) {
    if (millis() - lastBlink >= 500) {
      lastBlink = millis();
      ledState = !ledState;
      digitalWrite(PIN_LED, ledState ? LOW : HIGH);
    }
  }

  if (millis() - lastPublish >= PUBLISH_INTERVAL_MS) {
    lastPublish = millis();

    if (!relayTripped) {

      // ── Read sensors ────────────────────────────────────────────────────────
      float current     = readCurrent_ACS712();
      float voltage     = readVoltage();
      float temperature = dht.readTemperature();
      float humidity    = dht.readHumidity();
      float gas         = readGas();
      float power       = current * voltage;

      if (!isnan(temperature)) lastTemp = temperature;
      else                     temperature = lastTemp;
      if (!isnan(humidity))    lastHumi = humidity;
      else                     humidity = lastHumi;

      // ── Serial debug — same format as Node 1 ────────────────────────────────
      Serial.printf("[Node2/ACS712] I=%.3fA  V=%.2fV  T=%.1fC  H=%.0f%%  Gas=%.0fppm  P=%.2fW\n",
                    current, voltage, temperature, humidity, gas, power);

      // Show which voltage method was used
      int raw27 = analogRead(PIN_VOLTAGE);
      if (raw27 > 100) {
        float vADC = (raw27 / 4095.0f) * 3.3f;
        Serial.printf("  V source: DIVIDER  ADC27=%d  junction=%.3fV\n",
                      raw27, vADC);
      } else {
        Serial.printf("  V source: ESTIMATED  I=%.3fA  drop=%.3fV\n",
                      current, current * WIRE_RESISTANCE);
      }

      // ── Publish JSON — exact same keys as Node 1 ────────────────────────────
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
      Serial.printf("  MQTT publish: %s\n", ok ? "OK" : "FAILED");

    } else {
      LED_OFF;
    }
  }
}

// ══════════════════════════════════════════════════════════════════════════════
// SENSOR FUNCTIONS
// ══════════════════════════════════════════════════════════════════════════════

float readCurrent_ACS712() {
  long sum = 0;
  for (int i = 0; i < 100; i++) {
    sum += analogRead(PIN_CURRENT);
    delayMicroseconds(50);
  }
  float raw     = sum / 100.0f;
  float vSense  = (raw / 4095.0f) * 3.3f;
  float current = (vSense - ACS712_MIDPOINT) / ACS712_SENSITIVITY;
  return abs(current);
}

float readVoltage() {
  // Try voltage divider on GPIO27 first
  // Average 50 samples
  long sum = 0;
  for (int i = 0; i < 50; i++) {
    sum += analogRead(PIN_VOLTAGE);
    delayMicroseconds(50);
  }
  float raw = sum / 50.0f;

  if (raw > 100) {
    // ── Method 1: Voltage divider connected and working ──────────────────────
    // V = (ADC/4095 × 3.3) / ratio
    // ratio = R2/(R1+R2) = 10/(33+10) = 0.2326
    float vADC = (raw / 4095.0f) * 3.3f;
    return vADC / VOLT_DIVIDER_RATIO;

  } else {
    // ── Method 2: Divider not connected — estimate from current ──────────────
    // Based on Node 1 reading: V=10.15V at I=0.082A
    // Formula: V = V_NOMINAL - (I × WIRE_RESISTANCE)
    // This accounts for voltage drop under load
    // At no load (I≈0): V = 10.20V
    // At load (I=0.082A): V = 10.20 - (0.082 × 2.0) = 10.04V ≈ matches Node1
    float current = readCurrent_ACS712();
    float voltage = V_NOMINAL - (current * WIRE_RESISTANCE);
    // Clamp to realistic range
    voltage = constrain(voltage, 8.0f, 13.0f);
    return voltage;
  }
}

float readGas() {
  long sum = 0;
  for (int i = 0; i < 20; i++) {
    sum += analogRead(PIN_GAS);
    delayMicroseconds(100);
  }
  float raw = sum / 20.0f;
  long  ppm = map((long)raw, 200, 3500, 50, 900);
  ppm = constrain(ppm, 50, 900);
  return (float)ppm;
}

// ══════════════════════════════════════════════════════════════════════════════
// MQTT CALLBACK
// ══════════════════════════════════════════════════════════════════════════════
void mqttCallback(char* topic, byte* payload, unsigned int length) {
  String topicStr = String(topic);
  String msg = "";
  for (unsigned int i = 0; i < length; i++) msg += (char)payload[i];
  Serial.printf("MQTT IN [%s]: %s\n", topic, msg.c_str());

  if (topicStr == TOPIC_RELAY) {
    StaticJsonDocument<128> doc;
    if (deserializeJson(doc, msg) != DeserializationError::Ok) {
      Serial.println("JSON parse error");
      return;
    }
    String action = doc["action"] | "";

    if (action == "cut") {
      Serial.println("RELAY: Power CUT — isolated by Pi");
      digitalWrite(PIN_RELAY, HIGH);  // Cut power to load
      LED_OFF;                        // LED OFF — load is cut
      relayTripped = true;

    } else if (action == "restore") {
      Serial.println("RELAY: Power RESTORED by Pi");
      digitalWrite(PIN_RELAY, LOW);   // Restore power to load
      relayTripped = false;
      // LED will resume blinking in main loop

    } else {
      Serial.printf("Unknown action: %s\n", action.c_str());
    }
  }
}

// ══════════════════════════════════════════════════════════════════════════════
// WiFi & MQTT
// ══════════════════════════════════════════════════════════════════════════════
void connectWiFi() {
  Serial.printf("Connecting to WiFi: %s", WIFI_SSID);
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 40) {
    delay(500);
    Serial.print(".");
    attempts++;
  }
  if (WiFi.status() == WL_CONNECTED) {
    Serial.printf("\nWiFi connected — IP: %s\n",
                  WiFi.localIP().toString().c_str());
  } else {
    Serial.println("\nWiFi FAILED. Restarting...");
    delay(3000);
    ESP.restart();
  }
}

void connectMQTT() {
  char clientId[24];
  sprintf(clientId, "esp32_node_%d", NODE_ID);
  int retries = 0;
  while (!mqtt.connected() && retries < 5) {
    Serial.printf("MQTT connecting as [%s]...", clientId);
    if (mqtt.connect(clientId)) {
      Serial.println(" connected");
      mqtt.subscribe(TOPIC_RELAY);
      Serial.printf("Subscribed: %s\n", TOPIC_RELAY);
    } else {
      Serial.printf(" failed (rc=%d). Retry in 3s\n", mqtt.state());
      delay(3000);
      retries++;
    }
  }
  if (!mqtt.connected()) {
    Serial.println("MQTT failed. Restarting...");
    delay(1000);
    ESP.restart();
  }
}