# =============================================================================
#  app.py — ESP32 Edge AI Monitor | Raspberry Pi
#  Models  : Random Forest + Gradient Boosting (XGBoost)
#  Classes : 11 (Normal + 5 genuine faults + 5 cyber attacks)
#  Features: 17 engineered features per sample
#  Run     : python3 app.py
#  Open    : http://<PI_IP>:5000
# =============================================================================

import os, json, time, threading, collections, datetime, sqlite3, csv, io
import numpy as np
import joblib

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

from flask import Flask, render_template_string, jsonify, request as freq, Response
from flask_socketio import SocketIO

try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
MQTT_BROKER    = "localhost"
MQTT_PORT      = 1883
TOPIC_NODE1    = "sensors/node1"
TOPIC_NODE2    = "sensors/node2"
TOPIC_NODE3    = "sensors/node3"
TOPIC_RELAY_N1 = "control/relay/node1"
TOPIC_RELAY_N2 = "control/relay/node2"
MODEL_DIR      = "saved_models"
DB_PATH        = "energy_monitor.db"

FAULT_LABELS = {
    0:"Normal",      1:"Overcurrent",    2:"Overvoltage",
    3:"Undervoltage",4:"Overtemperature",5:"Gas Leak",
    6:"Spoofing Attack",7:"Replay Attack",8:"Gradual Drift",
    9:"Pulse Attack",10:"Physics Attack",
}
GENUINE_FAULTS = {1,2,3,4,5}
ATTACK_CLASSES = {6,7,8,9,10}

FEATURE_NAMES = [
    "current","voltage","temperature","gas","power",
    "delta_current","delta_voltage","delta_temp","delta_gas",
    "roll_std_current","roll_std_voltage","roll_std_temp","roll_std_gas",
    "physics_error","z_current","z_voltage","cross_delta_temp",
]

NODE_BASELINES = {
    "node1": {"current":0.092,"voltage":10.73,"temperature":31.8,"gas":457.0,"power":0.99},
    "node2": {"current":0.980,"voltage":8.24, "temperature":31.7,"gas":383.0,"power":8.08},
}

THRESHOLDS = {
    "current_high":3.0,"voltage_high":20.0,"voltage_low":3.0,
    "temp_warn":50.0,"temp_high":70.0,"gas_warn":600.0,"gas_high":1000.0,
    "power_high":30.0,
}

# ---------------------------------------------------------------------------
# SERIALIZATION FIX
# ---------------------------------------------------------------------------
def to_py(obj):
    if isinstance(obj, dict):          return {k: to_py(v) for k,v in obj.items()}
    if isinstance(obj, (list,tuple)):  return [to_py(v) for v in obj]
    if isinstance(obj, np.integer):    return int(obj)
    if isinstance(obj, np.floating):   return float(obj)
    if isinstance(obj, np.ndarray):    return obj.tolist()
    return obj

# ---------------------------------------------------------------------------
# FEATURE ENGINE — 17 features from rolling sensor history
# ---------------------------------------------------------------------------
class FeatureEngine:
    WINDOW       = 20
    BASELINE_LEN = 20
    NORMAL_TEMP  = 32.0

    def __init__(self):
        self._hist = {
            "node1": collections.deque(maxlen=self.WINDOW),
            "node2": collections.deque(maxlen=self.WINDOW),
        }
        self._bl = {
            "node1": {"current":[],"voltage":[]},
            "node2": {"current":[],"voltage":[]},
        }

    def update(self, node_key: str, payload: dict) -> np.ndarray:
        curr = float(payload.get("current",    0) or 0)
        volt = float(payload.get("voltage",    0) or 0)
        temp = float(payload.get("temperature",0) or 0)
        gas  = float(payload.get("gas",        0) or 0)
        pwr  = float(payload.get("power",      0) or 0)

        h = self._hist[node_key]
        h.append({"c":curr,"v":volt,"t":temp,"g":gas})

        c_arr = np.array([x["c"] for x in h])
        v_arr = np.array([x["v"] for x in h])
        t_arr = np.array([x["t"] for x in h])
        g_arr = np.array([x["g"] for x in h])

        d_c = curr - (c_arr[-2] if len(c_arr)>1 else curr)
        d_v = volt - (v_arr[-2] if len(v_arr)>1 else volt)
        d_t = temp - (t_arr[-2] if len(t_arr)>1 else temp)
        d_g = gas  - (g_arr[-2] if len(g_arr)>1 else gas)

        rs_c = float(c_arr.std()) if len(c_arr)>1 else 0.0
        rs_v = float(v_arr.std()) if len(v_arr)>1 else 0.0
        rs_t = float(t_arr.std()) if len(t_arr)>1 else 0.0
        rs_g = float(g_arr.std()) if len(g_arr)>1 else 0.0

        phys_err = abs(pwr - curr*volt) / (curr*volt + 1e-6)

        bl = self._bl[node_key]
        if len(bl["current"]) < self.BASELINE_LEN:
            bl["current"].append(curr)
            bl["voltage"].append(volt)
        b_c = np.array(bl["current"]) if bl["current"] else np.array([curr])
        b_v = np.array(bl["voltage"]) if bl["voltage"] else np.array([volt])
        z_c = (curr - b_c.mean()) / max(b_c.std(), 0.01)
        z_v = (volt - b_v.mean()) / max(b_v.std(), 0.01)

        cross = abs(temp - self.NORMAL_TEMP)

        return np.array([
            curr, volt, temp, gas, pwr,
            d_c, d_v, d_t, d_g,
            rs_c, rs_v, rs_t, rs_g,
            phys_err, z_c, z_v, cross,
        ], dtype=np.float32)

feature_engine = FeatureEngine()

# ---------------------------------------------------------------------------
# MODEL LOADING
# ---------------------------------------------------------------------------
models = {
    "gradient_boosting": None, "gb_scaler": None,
    "random_forest":     None, "rf_scaler": None,
}
MODEL_ACCURACY = {"gradient_boosting": 0.0, "random_forest": 0.0}
MODEL_METRICS  = {}

def load_models():
    global MODEL_ACCURACY, MODEL_METRICS
    if not os.path.exists(MODEL_DIR):
        print(f"[ML] '{MODEL_DIR}' not found — run training scripts first")
        return

    gb_sc = os.path.join(MODEL_DIR, "gradient_boosting_scaler.pkl")
    gb_js = os.path.join(MODEL_DIR, "gradient_boosting_model.json")
    gb_pk = os.path.join(MODEL_DIR, "gradient_boosting_model.pkl")
    if os.path.exists(gb_sc):
        models["gb_scaler"] = joblib.load(gb_sc)
        if XGB_AVAILABLE and os.path.exists(gb_js):
            clf = xgb.XGBClassifier(); clf.load_model(gb_js)
            models["gradient_boosting"] = clf
            print("[ML] Gradient Boosting (XGBoost) loaded")
        elif os.path.exists(gb_pk):
            models["gradient_boosting"] = joblib.load(gb_pk)
            print("[ML] Gradient Boosting (sklearn) loaded")

    rf_md = os.path.join(MODEL_DIR, "random_forest_model.pkl")
    rf_sc = os.path.join(MODEL_DIR, "random_forest_scaler.pkl")
    if os.path.exists(rf_md) and os.path.exists(rf_sc):
        models["random_forest"] = joblib.load(rf_md)
        models["rf_scaler"]     = joblib.load(rf_sc)
        print("[ML] Random Forest loaded")

    for fname, target in [("model_accuracy.json", MODEL_ACCURACY),
                           ("model_metrics.json",  MODEL_METRICS)]:
        p = os.path.join(MODEL_DIR, fname)
        if os.path.exists(p):
            with open(p) as f: target.update(json.load(f))

load_models()

# ---------------------------------------------------------------------------
# ML INFERENCE
# ---------------------------------------------------------------------------
def run_ml(node_key: str, payload: dict) -> dict:
    features = feature_engine.update(node_key, payload)
    results  = {}
    for mname, mkey, skey in [
        ("gradient_boosting","gradient_boosting","gb_scaler"),
        ("random_forest",    "random_forest",    "rf_scaler"),
    ]:
        clf = models.get(mkey)
        sc  = models.get(skey)
        if clf is None or sc is None:
            continue
        try:
            X_s   = sc.transform(features.reshape(1, -1))
            cls   = int(clf.predict(X_s)[0])
            proba = clf.predict_proba(X_s)[0]
            conf  = float(proba.max()) * 100
            results[mname] = {
                "label":      cls,
                "class_name": FAULT_LABELS.get(cls, f"Class {cls}"),
                "confidence": round(conf, 1),
                "proba_all":  [round(float(p)*100, 1) for p in proba],
                "is_fault":   cls > 0,
                "is_attack":  cls in ATTACK_CLASSES,
            }
        except Exception as e:
            print(f"[ML] {mname} inference error: {e}")
    return results

# ---------------------------------------------------------------------------
# DATABASE
# ---------------------------------------------------------------------------
def db_connect():
    c = sqlite3.connect(DB_PATH, check_same_thread=False)
    c.row_factory = sqlite3.Row
    return c

def db_init():
    c = db_connect()
    c.executescript("""
        CREATE TABLE IF NOT EXISTS readings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL, node TEXT NOT NULL,
            current REAL, voltage REAL, power REAL,
            temperature REAL, humidity REAL, gas REAL,
            gb_class INTEGER DEFAULT -1, gb_conf REAL DEFAULT 0,
            rf_class INTEGER DEFAULT -1, rf_conf REAL DEFAULT 0,
            fault_name TEXT DEFAULT '',
            spoof INTEGER DEFAULT 0, n_faults INTEGER DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS faults (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL, node TEXT NOT NULL,
            fault_type TEXT NOT NULL, severity TEXT NOT NULL,
            value REAL DEFAULT 0, unit TEXT DEFAULT ''
        );
        CREATE INDEX IF NOT EXISTS idx_r_ts   ON readings(ts);
        CREATE INDEX IF NOT EXISTS idx_r_node ON readings(node);
    """)
    c.commit(); c.close()
    print(f"[DB] Ready: {DB_PATH}")

db_init()
_db_lock = threading.Lock()

def db_insert(node, payload, ml, faults, spoof):
    ts  = payload.get("_ts", datetime.datetime.now().isoformat())
    gb  = ml.get("gradient_boosting", {})
    rf  = ml.get("random_forest",     {})
    fn  = gb.get("class_name", rf.get("class_name", "Unknown"))
    with _db_lock:
        c = db_connect()
        c.execute(
            "INSERT INTO readings(ts,node,current,voltage,power,temperature,humidity,gas,"
            "gb_class,gb_conf,rf_class,rf_conf,fault_name,spoof,n_faults)"
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (ts, node,
             payload.get("current"), payload.get("voltage"), payload.get("power"),
             payload.get("temperature"), payload.get("humidity"), payload.get("gas"),
             gb.get("label",-1), gb.get("confidence",0),
             rf.get("label",-1), rf.get("confidence",0),
             fn, 1 if spoof else 0, len(faults)))
        for f in faults:
            c.execute("INSERT INTO faults(ts,node,fault_type,severity,value,unit)"
                      "VALUES(?,?,?,?,?,?)",
                      (ts,node,f["type"],f["severity"],f.get("value",0),f.get("unit","")))
        c.commit(); c.close()

# ---------------------------------------------------------------------------
# SPOOF / FAULT DETECTION
# ---------------------------------------------------------------------------
_spoof_win = {
    "node1": collections.deque(maxlen=10),
    "node2": collections.deque(maxlen=10),
}

def check_spoof(node_key, payload):
    V = float(payload.get("voltage",0) or 0)
    I = float(payload.get("current",0) or 0)
    G = float(payload.get("gas",   0) or 0)
    P = float(payload.get("power", 0) or 0)
    _spoof_win[node_key].append({"V":V,"I":I,"G":G,"P":P})
    bl = NODE_BASELINES.get(node_key, {})
    for key,val in [("voltage",V),("current",I),("gas",G),("power",P)]:
        base = bl.get(key, 0)
        if base > 0 and val >= 2.0 * base and val > 0:
            return True
    return False

def check_stuck(node_key):
    win = _spoof_win[node_key]
    if len(win) < 5: return False
    vs = [round(r["V"],3) for r in list(win)[-5:]]
    return len(set(vs))==1 and vs[0]>0

def detect_faults(node_key, payload, ml):
    faults = []
    for mname in ("gradient_boosting","random_forest"):
        r = ml.get(mname, {})
        if r.get("is_fault") and r.get("label", 0) > 0:
            cls = r["label"]
            sev = "critical" if cls in {1,2,3,4,5,6,9} else "warning"
            faults.append({
                "type":      r["class_name"].upper().replace(" ","_"),
                "severity":  sev,
                "value":     round(r["confidence"],1),
                "unit":      f"% conf [{mname[:2].upper()}]",
                "ml_source": mname,
            })
    if not models["gradient_boosting"] and not models["random_forest"]:
        V = float(payload.get("voltage",    0) or 0)
        I = float(payload.get("current",    0) or 0)
        T = float(payload.get("temperature",0) or 0)
        G = float(payload.get("gas",        0) or 0)
        P = float(payload.get("power",      0) or 0)
        if V > THRESHOLDS["voltage_high"]:
            faults.append({"type":"OVERVOLTAGE","severity":"critical","value":round(V,2),"unit":"V"})
        elif 0 < V < THRESHOLDS["voltage_low"]:
            faults.append({"type":"UNDERVOLTAGE","severity":"warning","value":round(V,2),"unit":"V"})
        if I > THRESHOLDS["current_high"]:
            faults.append({"type":"OVERCURRENT","severity":"critical","value":round(I,3),"unit":"A"})
        if T > THRESHOLDS["temp_high"]:
            faults.append({"type":"OVERTEMPERATURE","severity":"critical","value":round(T,1),"unit":"°C"})
        if G > THRESHOLDS["gas_high"]:
            faults.append({"type":"GAS_LEAK","severity":"critical","value":round(G,0),"unit":"ppm"})
        if P > THRESHOLDS["power_high"]:
            faults.append({"type":"OVERPOWER","severity":"critical","value":round(P,1),"unit":"W"})
    if check_stuck(node_key):
        faults.append({"type":"SENSOR_STUCK","severity":"warning","value":0,"unit":""})
    return faults

# ---------------------------------------------------------------------------
# TERMINAL BATCH PRINTER
# ---------------------------------------------------------------------------
batch_counters = {"node1":0,"node2":0}
C = {"R":"\033[91m","G":"\033[92m","Y":"\033[93m","C":"\033[96m",
     "P":"\033[95m","W":"\033[97m","D":"\033[90m","B":"\033[1m","X":"\033[0m"}
def cc(col,t): return f"{C.get(col,'')}{t}{C['X']}"

def print_batch(node_key, payload, ml, faults, spoof):
    batch_counters[node_key] += 1
    bn = batch_counters[node_key]
    ts = payload.get("_ts","")[:19].replace("T"," ")
    lbl = "Node 1 — INA219" if node_key=="node1" else "Node 2 — ACS712"
    sep = "─"*70
    print(f"\n{cc('C',sep)}")
    print(f"  {cc('B',f'BATCH #{bn:<4}')} │ {cc('C',lbl):<32} │ {cc('D',ts)}")
    print(cc('C',sep))
    V = float(payload.get("voltage",    0) or 0)
    I = float(payload.get("current",    0) or 0)
    P = float(payload.get("power",      0) or 0)
    T = float(payload.get("temperature",0) or 0)
    H = float(payload.get("humidity",   0) or 0)
    G = float(payload.get("gas",        0) or 0)
    print(f"  I={cc('W',f'{I:.3f}A'):<18} V={cc('W',f'{V:.2f}V'):<18} P={cc('W',f'{P:.2f}W')}")
    print(f"  T={cc('W',f'{T:.1f}°C'):<18} H={cc('W',f'{H:.0f}%'):<18} Gas={cc('W',f'{G:.0f}ppm')}")
    for mname,short in [("gradient_boosting","GB"),("random_forest","RF")]:
        r = ml.get(mname,{})
        if r:
            cls_name = r.get("class_name","?")
            conf     = r.get("confidence",0)
            is_f     = r.get("is_fault",False)
            is_a     = r.get("is_attack",False)
            col = "R" if is_a else ("Y" if is_f else "G")
            print(f"  [{short}] {cc(col,f'{cls_name:<22}')} {conf:.1f}% confidence")
    unique_faults = {f["type"]:f for f in faults}.values()
    if unique_faults:
        for f2 in unique_faults:
            sc = "R" if f2["severity"]=="critical" else "Y"
            print(f"  {cc(sc,f'[{f2[\"severity\"].upper():<8}]')} {f2['type'].replace('_',' ')}: {cc('W',str(f2['value']))} {f2['unit']}")
    else:
        print(f"  {cc('G','All clear')} — no faults")
    if spoof: print(f"  {cc('P','SPOOF DETECTED')} — value exceeds 2x node baseline")
    relay = state[node_key]["relay"]
    print(f"  Relay: {cc('R','CUT') if relay=='cut' else cc('G','Normal')}")
    print(cc('D',sep))

# ---------------------------------------------------------------------------
# FLASK + SOCKETIO
# ---------------------------------------------------------------------------
app      = Flask(__name__)
app.config["SECRET_KEY"] = "edge_ai_2025"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# ---------------------------------------------------------------------------
# SHARED STATE
# ---------------------------------------------------------------------------
state = {
    "node1": {"latest":{}, "relay":"normal", "faults":[], "ml":{},
              "history":collections.deque(maxlen=60), "spoof_active":False},
    "node2": {"latest":{}, "relay":"normal", "faults":[], "ml":{},
              "history":collections.deque(maxlen=60), "spoof_active":False},
    "alerts":    collections.deque(maxlen=500),
    "total_msg": 0,
    "start_time":time.time(),
    "confusion": {
        "gradient_boosting": {"tp":0,"tn":0,"fp":0,"fn":0},
        "random_forest":     {"tp":0,"tn":0,"fp":0,"fn":0},
    },
}
_state_lock = threading.Lock()

# ---------------------------------------------------------------------------
# AUTO RELAY
# ---------------------------------------------------------------------------
mqtt_client = None
_relay_lock = threading.Lock()

def auto_relay_cut(node_key, faults):
    if not any(f["severity"]=="critical" for f in faults): return
    with _relay_lock:
        if state[node_key]["relay"]=="cut": return
        state[node_key]["relay"] = "cut"
    topic  = TOPIC_RELAY_N1 if node_key=="node1" else TOPIC_RELAY_N2
    reason = next(f["type"] for f in faults if f["severity"]=="critical")
    if mqtt_client: mqtt_client.publish(topic, json.dumps({"action":"cut"}), qos=1)
    print(f"  {cc('R','[AUTO-CUT]')} Relay → {node_key} ({reason})")
    socketio.emit("relay_update",{"node":node_key,"relay":"cut","reason":reason,"auto":True})

# ---------------------------------------------------------------------------
# SENSOR HANDLER
# ---------------------------------------------------------------------------
def _parse(payload):
    out = {}
    for k,v in payload.items():
        try:    out[k] = float(v)
        except: out[k] = v
    return out

def handle_sensor(node_key, raw):
    if isinstance(raw, str):
        try:    raw = json.loads(raw)
        except: return
    payload = _parse(raw)
    payload["_ts"] = datetime.datetime.now().isoformat()

    spoof  = check_spoof(node_key, payload)
    ml     = run_ml(node_key, payload)
    faults = detect_faults(node_key, payload, ml)

    if spoof and not any(f["type"]=="SPOOF_DETECTED" for f in faults):
        faults.insert(0,{"type":"SPOOF_DETECTED","severity":"critical",
                         "value":0,"unit":"values 2x baseline"})

    is_real_fault = any(f["severity"]=="critical" and "ML_" not in f["type"] for f in faults)
    with _state_lock:
        for mn in ("gradient_boosting","random_forest"):
            r = ml.get(mn,{})
            if "is_fault" in r:
                pred_f = r["is_fault"]
                cm = state["confusion"][mn]
                if is_real_fault and pred_f:       cm["tp"]+=1
                elif is_real_fault and not pred_f: cm["fn"]+=1
                elif not is_real_fault and pred_f: cm["fp"]+=1
                else:                              cm["tn"]+=1

        nd = state[node_key]
        nd["latest"]       = payload
        nd["ml"]           = ml
        nd["faults"]       = faults
        nd["spoof_active"] = spoof
        nd["history"].append({
            "ts":          payload["_ts"][11:19],
            "current":     round(float(payload.get("current",    0) or 0),3),
            "voltage":     round(float(payload.get("voltage",    0) or 0),2),
            "power":       round(float(payload.get("power",      0) or 0),2),
            "temperature": round(float(payload.get("temperature",0) or 0),1),
            "gas":         round(float(payload.get("gas",        0) or 0),0),
            "humidity":    round(float(payload.get("humidity",   0) or 0),1),
            "gb_class":    ml.get("gradient_boosting",{}).get("label",-1),
            "rf_class":    ml.get("random_forest",    {}).get("label",-1),
        })
        state["total_msg"] += 1
        for f in faults:
            state["alerts"].appendleft({"node":node_key,"ts":payload["_ts"],**f})

    print_batch(node_key, payload, ml, faults, spoof)
    threading.Thread(target=db_insert, args=(node_key,payload,ml,faults,spoof),daemon=True).start()
    auto_relay_cut(node_key, faults)

    socketio.emit("sensor_update", to_py({
        "node":  node_key,
        "data":  {k:(round(float(v),4) if isinstance(v,(int,float,np.number)) else v)
                  for k,v in payload.items()},
        "ml":    ml,
        "faults":faults,
        "relay": state[node_key]["relay"],
        "spoof": spoof,
    }))

def handle_node3(topic, payload):
    ts  = datetime.datetime.now().isoformat()
    atk = payload.get("attack_type","") if isinstance(payload,dict) else ""
    if atk == "RESTORED":
        nk = "node1" if "node1" in payload.get("target_node","") else "node2"
        with _state_lock:
            state[nk]["relay"]        = "normal"
            state[nk]["spoof_active"] = False
            state[nk]["faults"]       = []
        socketio.emit("relay_update",{"node":nk,"relay":"normal","auto":False,"reason":"RESTORED"})
        socketio.emit("node_restored",{"node":nk})
        return
    alert = {"node":"node3","ts":ts,"type":"SPOOF_INJECTION","severity":"critical",
             "value":str(payload),"unit":""}
    with _state_lock: state["alerts"].appendleft(alert)
    socketio.emit("alert", alert)

# ---------------------------------------------------------------------------
# MQTT
# ---------------------------------------------------------------------------
def start_mqtt():
    global mqtt_client
    if not MQTT_AVAILABLE: print("[MQTT] paho-mqtt not installed"); return
    def _on_connect(c,ud,flags,rc):
        if rc==0:
            c.subscribe([(TOPIC_NODE1,0),(TOPIC_NODE2,0),(TOPIC_NODE3,0)])
            print(f"[MQTT] Connected → {MQTT_BROKER}:{MQTT_PORT}")
        else: print(f"[MQTT] Refused rc={rc}")
    def _on_message(c,ud,msg):
        topic = msg.topic
        try:    payload = json.loads(msg.payload.decode())
        except: payload = msg.payload.decode(errors="replace")
        if topic==TOPIC_NODE1:
            threading.Thread(target=handle_sensor,args=("node1",payload),daemon=True).start()
        elif topic==TOPIC_NODE2:
            threading.Thread(target=handle_sensor,args=("node2",payload),daemon=True).start()
        elif topic==TOPIC_NODE3:
            threading.Thread(target=handle_node3, args=(topic,payload), daemon=True).start()
    def _on_disconnect(c,ud,rc):
        print(f"[MQTT] Disconnected rc={rc} — retrying...")
        while True:
            try: c.reconnect(); break
            except: time.sleep(5)
    mqtt_client = mqtt.Client(client_id="pi_server",clean_session=True)
    mqtt_client.on_connect    = _on_connect
    mqtt_client.on_message    = _on_message
    mqtt_client.on_disconnect = _on_disconnect
    try: mqtt_client.connect(MQTT_BROKER,MQTT_PORT,keepalive=60); mqtt_client.loop_start()
    except Exception as e: print(f"[MQTT] {e}")

# ---------------------------------------------------------------------------
# REST API
# ---------------------------------------------------------------------------
@app.route("/api/state")
def api_state():
    with _state_lock:
        return jsonify(to_py({
            "node1":     {k:state["node1"][k] for k in ["latest","relay","faults","ml","spoof_active"]},
            "node2":     {k:state["node2"][k] for k in ["latest","relay","faults","ml","spoof_active"]},
            "alerts":    list(state["alerts"])[:60],
            "total_msg": state["total_msg"],
            "uptime":    int(time.time()-state["start_time"]),
            "confusion": dict(state["confusion"]),
        }))

@app.route("/api/stream_history/<node>")
def api_stream_history(node):
    if node not in ("node1","node2"): return jsonify([])
    with _state_lock: return jsonify(list(state[node]["history"]))

@app.route("/api/history/<node>")
def api_history(node):
    if node not in ("node1","node2","all"): return jsonify({"error":"invalid"}),400
    limit  = int(freq.args.get("limit",100))
    offset = int(freq.args.get("offset",0))
    df_    = freq.args.get("from","")
    dt_    = freq.args.get("to","")
    with _db_lock:
        conn=db_connect(); q="SELECT * FROM readings"; conds=[]; params=[]
        if node!="all": conds.append("node=?"); params.append(node)
        if df_: conds.append("ts>=?"); params.append(df_)
        if dt_: conds.append("ts<=?"); params.append(dt_+" 23:59:59")
        if conds: q+=" WHERE "+" AND ".join(conds)
        q+=" ORDER BY id DESC LIMIT ? OFFSET ?"; params+=[limit,offset]
        rows  = conn.execute(q,params).fetchall()
        total = conn.execute(("SELECT COUNT(*) FROM readings"+(
                " WHERE "+" AND ".join(conds) if conds else "")),params[:-2]).fetchone()[0]
        conn.close()
    return jsonify({"rows":[dict(r) for r in rows],"total":total})

@app.route("/api/relay", methods=["POST"])
def api_relay():
    data   = freq.json or {}
    node   = data.get("node")
    action = data.get("action")
    if node not in ("node1","node2") or action not in ("cut","restore"):
        return jsonify({"error":"bad params"}),400
    topic = TOPIC_RELAY_N1 if node=="node1" else TOPIC_RELAY_N2
    if mqtt_client: mqtt_client.publish(topic,json.dumps({"action":action}),qos=1)
    with _state_lock:
        state[node]["relay"] = "cut" if action=="cut" else "normal"
        if action=="restore": state[node]["spoof_active"]=False; state[node]["faults"]=[]
    socketio.emit("relay_update",{"node":node,"relay":state[node]["relay"],"auto":False})
    return jsonify({"ok":True})

@app.route("/api/stats")
def api_stats():
    with _db_lock:
        conn = db_connect()
        tot  = conn.execute("SELECT COUNT(*) FROM readings").fetchone()[0]
        flt  = conn.execute("SELECT COUNT(*) FROM faults").fetchone()[0]
        sp   = conn.execute("SELECT COUNT(*) FROM readings WHERE spoof=1").fetchone()[0]
        gba  = conn.execute("SELECT COUNT(*) FROM readings WHERE gb_class>0").fetchone()[0]
        rfa  = conn.execute("SELECT COUNT(*) FROM readings WHERE rf_class>0").fetchone()[0]
        conn.close()
    return jsonify({"total_readings":tot,"total_faults":flt,"spoof_count":sp,
                    "gb_anomalies":gba,"rf_anomalies":rfa})

@app.route("/api/model/accuracy")
def api_model_accuracy():
    return jsonify({"accuracy":MODEL_ACCURACY,"metrics":MODEL_METRICS,
                    "gb_loaded": models["gradient_boosting"] is not None,
                    "rf_loaded": models["random_forest"] is not None})

@app.route("/api/export/csv")
def export_csv():
    node  = freq.args.get("node","all")
    limit = int(freq.args.get("limit",5000))
    with _db_lock:
        conn = db_connect()
        rows = conn.execute(
            "SELECT * FROM readings WHERE node=? ORDER BY id DESC LIMIT ?" if node!="all"
            else "SELECT * FROM readings ORDER BY id DESC LIMIT ?",
            (node,limit) if node!="all" else (limit,)).fetchall()
        conn.close()
    buf = io.StringIO()
    if rows:
        w = csv.DictWriter(buf, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows([dict(r) for r in rows])
    return Response(buf.getvalue(), mimetype="text/csv",
        headers={"Content-Disposition":
                 f"attachment; filename=readings_{node}_{datetime.date.today()}.csv"})

@app.route("/")
def dashboard():
    return render_template_string(HTML)

# ---------------------------------------------------------------------------
# DASHBOARD HTML
# ---------------------------------------------------------------------------
_FAULT_COLORS = {
    0:"#22c55e",1:"#ef4444",2:"#f97316",3:"#eab308",4:"#a855f7",
    5:"#78716c",6:"#dc2626",7:"#b45309",8:"#0284c7",9:"#db2777",10:"#7c3aed"
}
_FAULT_COLORS_JS = json.dumps(_FAULT_COLORS)
_FAULT_LABELS_JS = json.dumps(FAULT_LABELS)

HTML = r"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Edge AI Monitor</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet"/>
<script src="https://cdn.socket.io/4.7.2/socket.io.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg0:#f5f6fa;--bg1:#fff;--bg2:#f0f2f8;--bd:#dde2ef;--bd2:#c5ccdf;
  --t1:#1a1f36;--t2:#5a637a;--t3:#9199ae;
  --cy:#0969da;--gn:#1a7f37;--rd:#cf222e;--or:#bc4c00;--pu:#6639ba;
  --cy-bg:#dbeafe;--gn-bg:#dcfce7;--rd-bg:#fee2e2;--or-bg:#ffedd5;--pu-bg:#ede9fe;
  --mn:'JetBrains Mono',monospace;--ss:'Inter',sans-serif;--r:8px;--r2:12px;
}
html{font-size:15px;-webkit-font-smoothing:antialiased}
body{background:var(--bg0);color:var(--t1);font-family:var(--ss);min-height:100vh}
.topbar{background:var(--bg1);border-bottom:1.5px solid var(--bd);height:60px;
  display:flex;align-items:center;padding:0 24px;justify-content:space-between;
  position:sticky;top:0;z-index:200;box-shadow:0 1px 4px rgba(0,0,0,.06)}
.brand{display:flex;align-items:center;gap:12px}
.b-ico{width:36px;height:36px;background:var(--cy);border-radius:8px;display:flex;align-items:center;justify-content:center}
.b-ico svg{width:20px;height:20px;fill:#fff}
.b-title{font-family:var(--mn);font-size:1rem;font-weight:500}
.b-sub{font-size:.75rem;color:var(--t3);margin-top:2px}
.topright{display:flex;align-items:center;gap:8px;flex-wrap:wrap}
.pill{display:inline-flex;align-items:center;gap:6px;font-family:var(--mn);font-size:.75rem;
  padding:5px 12px;border-radius:99px;border:1px solid var(--bd);background:var(--bg2);color:var(--t2)}
.pd{width:8px;height:8px;border-radius:50%}
.pill.ok .pd{background:var(--gn);animation:glw 2s ease-in-out infinite}
.pill.err .pd{background:var(--rd);animation:blk .8s step-end infinite}
@keyframes glw{0%,100%{opacity:1}50%{opacity:.3}}
@keyframes blk{50%{opacity:.2}}
.tabbar{background:var(--bg1);border-bottom:1.5px solid var(--bd);display:flex;padding:0 24px;overflow-x:auto}
.tbtn{font-size:.85rem;font-weight:500;padding:16px 20px;cursor:pointer;color:var(--t3);
  border-bottom:2.5px solid transparent;border:none;background:none;white-space:nowrap}
.tbtn:hover{color:var(--t2)}.tbtn.active{color:var(--cy);border-bottom-color:var(--cy)}
.tcnt{display:inline-block;background:var(--rd);color:#fff;font-size:.65rem;font-family:var(--mn);
  padding:2px 6px;border-radius:99px;margin-left:6px;min-width:19px;text-align:center}
.page{display:none;padding:20px 24px;max-width:1600px;margin:0 auto}
.page.active{display:block}
.g2{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px}
.g3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px;margin-bottom:16px}
.g5{display:grid;grid-template-columns:repeat(5,1fr);gap:14px;margin-bottom:16px}
@media(max-width:1100px){.g5{grid-template-columns:1fr 1fr}}
@media(max-width:700px){.g2,.g3,.g5{grid-template-columns:1fr}}
.card{background:var(--bg1);border:1.5px solid var(--bd);border-radius:var(--r2);padding:18px 20px}
.chdr{display:flex;align-items:center;justify-content:space-between;margin-bottom:14px;
  padding-bottom:12px;border-bottom:1px solid var(--bd)}
.ctitle{font-size:.75rem;font-weight:600;letter-spacing:.1em;text-transform:uppercase;color:var(--t3)}
.cbdg{font-family:var(--mn);font-size:.72rem;padding:3px 10px;border-radius:5px;font-weight:600}
.cb-ok{background:var(--gn-bg);color:var(--gn)}.cb-err{background:var(--rd-bg);color:var(--rd)}
.cb-warn{background:var(--or-bg);color:var(--or)}.cb-sp{background:var(--pu-bg);color:var(--pu)}
.cb-atk{background:#fee2e2;color:#7c3aed}
.kpi{background:var(--bg1);border:1.5px solid var(--bd);border-radius:var(--r2);padding:18px 20px}
.kv{font-family:var(--mn);font-size:1.9rem;font-weight:500;color:var(--t1);line-height:1}
.kl{font-size:.72rem;font-weight:600;text-transform:uppercase;letter-spacing:.1em;color:var(--t3);margin-top:6px}
.ks{font-size:.78rem;color:var(--t3);margin-top:4px}
.mg{display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px}
.mc{background:var(--bg2);border:1.5px solid var(--bd);border-radius:var(--r);padding:11px 13px;transition:all .3s}
.mc.al{background:var(--rd-bg);border-color:#f9a8a8}.mc.wn{background:var(--or-bg);border-color:#fed7aa}
.ml_{font-size:.68rem;font-weight:600;text-transform:uppercase;letter-spacing:.09em;color:var(--t3);margin-bottom:5px}
.mv{font-family:var(--mn);font-size:1.25rem;font-weight:500;color:var(--t1);line-height:1}
.mu{font-size:.68rem;color:var(--t3);margin-left:2px}
.rrow{display:flex;align-items:center;justify-content:space-between;margin-top:14px;padding-top:14px;border-top:1px solid var(--bd)}
.rst{display:flex;align-items:center;gap:8px;font-size:.85rem;font-weight:500}
.rdot{width:10px;height:10px;border-radius:50%}.rok{background:var(--gn)}.rcu{background:var(--rd)}
.btn{font-size:.78rem;font-weight:500;padding:7px 15px;border-radius:6px;cursor:pointer;border:1.5px solid;transition:opacity .2s}
.btn:hover{opacity:.75}.b-cut{background:var(--rd-bg);color:var(--rd);border-color:#f9a8a8}
.b-rst{background:var(--gn-bg);color:var(--gn);border-color:#86efac}
.b-pri{background:var(--cy-bg);color:var(--cy);border-color:#93c5fd}
.brow{display:flex;gap:8px}
.fl{display:flex;flex-direction:column;gap:4px;margin-top:10px;max-height:110px;overflow-y:auto}
.fi{display:flex;align-items:center;gap:8px;padding:7px 11px;border-radius:6px;font-size:.8rem;font-weight:500}
.fi-c{background:var(--rd-bg);color:var(--rd)}.fi-h{background:var(--or-bg);color:var(--or)}
.fi-w{background:#fef9c3;color:#9a6700}.fi-a{background:var(--pu-bg);color:var(--pu)}
.fi-n{font-size:.8rem;color:var(--t3);padding:4px 0}
.div{height:1px;background:var(--bd);margin:14px 0}
.sec{font-size:.72rem;font-weight:600;text-transform:uppercase;letter-spacing:.12em;color:var(--t3);margin-bottom:10px}
.cbox{position:relative;width:100%}
.ml-panel{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:10px}
.ml-card{border-radius:8px;padding:10px 12px;border:1.5px solid var(--bd);background:var(--bg2)}
.ml-card.fault{border-color:#f9a8a8;background:var(--rd-bg)}
.ml-card.attack{border-color:#c4b5fd;background:var(--pu-bg)}
.ml-name{font-size:.68rem;font-weight:600;text-transform:uppercase;letter-spacing:.08em;color:var(--t3)}
.ml-cls{font-family:var(--mn);font-size:1rem;font-weight:600;margin-top:3px;color:var(--t1)}
.ml-conf{font-size:.75rem;color:var(--t2);margin-top:3px}
.ml-bar{height:6px;border-radius:3px;background:var(--bd);margin-top:6px}
.ml-barfill{height:100%;border-radius:3px;transition:width .4s ease}
.fr{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:12px}
.fr input,.fr select{background:var(--bg2);border:1.5px solid var(--bd);color:var(--t1);
  border-radius:6px;padding:7px 11px;font-family:var(--mn);font-size:.78rem}
.tw{overflow-x:auto;max-height:480px;overflow-y:auto;border-radius:8px;border:1.5px solid var(--bd)}
table{width:100%;border-collapse:collapse;font-size:.82rem}
thead{position:sticky;top:0;background:var(--bg2);z-index:10}
th{padding:10px 12px;font-size:.7rem;font-weight:600;letter-spacing:.09em;text-transform:uppercase;
  color:var(--t3);border-bottom:1.5px solid var(--bd);text-align:left}
td{padding:8px 12px;border-bottom:1px solid var(--bd);white-space:nowrap;font-variant-numeric:tabular-nums}
tr:hover td{background:var(--bg2)}
.cok{color:var(--gn)}.cbad{color:var(--rd)}.cwrn{color:#9a6700}
.np{font-family:var(--mn);font-size:.68rem;padding:2px 8px;border-radius:4px;font-weight:600}
.np1{background:var(--cy-bg);color:var(--cy)}.np2{background:var(--pu-bg);color:var(--pu)}
.cls-tag{font-family:var(--mn);font-size:.68rem;padding:2px 8px;border-radius:4px;font-weight:600;border:1px solid currentColor}
.pag{display:flex;align-items:center;gap:9px;margin-top:12px}
.pb{background:var(--bg2);border:1.5px solid var(--bd);color:var(--t2);border-radius:5px;padding:5px 12px;cursor:pointer;font-size:.78rem}
.al-f{display:flex;gap:8px;margin-bottom:14px;flex-wrap:wrap}
.af{background:var(--bg2);border:1.5px solid var(--bd);border-radius:6px;padding:6px 14px;
  font-size:.8rem;font-weight:500;cursor:pointer;color:var(--t3);transition:all .2s}
.af.active{border-color:var(--cy);color:var(--cy);background:var(--cy-bg)}
.al-list{display:flex;flex-direction:column;gap:5px;max-height:600px;overflow-y:auto}
.ai{padding:12px 16px;border-radius:8px;border-left:4px solid;background:var(--bg1);border:1.5px solid var(--bd);border-left-width:4px}
.ai-c{border-left-color:var(--rd)}.ai-h{border-left-color:var(--or)}
.ai-a{border-left-color:var(--pu)}.ai-w{border-left-color:#9a6700}
.an{font-size:.65rem;font-weight:600;text-transform:uppercase;color:var(--t3)}
.at{font-weight:600;margin:3px 0;color:var(--t1);font-size:.85rem}
.av{color:var(--t2);font-size:.8rem}.ats{font-family:var(--mn);font-size:.68rem;color:var(--t3);margin-top:3px}
.noal{text-align:center;color:var(--t3);padding:42px 0}
.cm-grid{display:grid;grid-template-columns:1fr 1fr;gap:6px;margin:10px 0}
.cm-cell{border-radius:6px;padding:12px 8px;text-align:center;border:1.5px solid}
.cm-tp{background:#dcfce7;border-color:#86efac;color:#15803d}.cm-tn{background:#dbeafe;border-color:#93c5fd;color:#1d4ed8}
.cm-fp{background:#ffedd5;border-color:#fed7aa;color:#c2410c}.cm-fn{background:#fee2e2;border-color:#f9a8a8;color:#b91c1c}
.cm-l{font-size:.6rem;font-weight:600;text-transform:uppercase;margin-bottom:3px}
.cm-v{font-family:var(--mn);font-size:1.3rem;font-weight:600}
::-webkit-scrollbar{width:4px;height:4px}
::-webkit-scrollbar-thumb{background:var(--bd2);border-radius:4px}
</style></head><body>

<div class="topbar">
  <div class="brand">
    <div class="b-ico"><svg viewBox="0 0 16 16"><path d="M8 1L1 5v6l7 4 7-4V5L8 1zM8 3.4l4.5 2.6V10L8 12.6 3.5 10V6L8 3.4z"/></svg></div>
    <div><div class="b-title">Edge AI Monitor</div><div class="b-sub">ESP32 · INA219 · ACS712 · RF + XGBoost · 11 Classes</div></div>
  </div>
  <div class="topright">
    <div class="pill err" id="cpill"><div class="pd"></div><span id="clbl">Connecting</span></div>
    <div class="pill"><span id="mcnt">0 msgs</span></div>
    <div class="pill"><span id="ucnt">Up: 00:00:00</span></div>
    <div class="pill">GB: <b id="gbs" style="margin-left:4px;color:var(--rd)">—</b></div>
    <div class="pill">RF: <b id="rfs" style="margin-left:4px;color:var(--rd)">—</b></div>
  </div>
</div>

<div class="tabbar">
  <button class="tbtn active" onclick="swTab('live',this)">Live Data</button>
  <button class="tbtn"        onclick="swTab('models',this)">ML Models</button>
  <button class="tbtn"        onclick="swTab('storage',this)">Storage &amp; Export</button>
  <button class="tbtn"        onclick="swTab('alerts',this)">Alerts<span class="tcnt" id="acnt">0</span></button>
</div>

<!-- TAB: LIVE DATA -->
<div class="page active" id="tab-live">
  <div class="g5">
    <div class="kpi"><div class="kv" id="k-tot">0</div><div class="kl">Total readings</div></div>
    <div class="kpi"><div class="kv" id="k-flt">0</div><div class="kl">Faults &amp; Attacks</div><div class="ks" id="k-sp">0 spoof events</div></div>
    <div class="kpi"><div class="kv" id="k-gba">0</div><div class="kl">GB detections</div></div>
    <div class="kpi"><div class="kv" id="k-rfa">0</div><div class="kl">RF detections</div></div>
    <div class="kpi"><div class="kv" id="k-up">—</div><div class="kl">Uptime</div></div>
  </div>
  <div class="g2">
    <div class="card">
      <div class="chdr"><span class="ctitle">Node 1 — INA219 (I²C)</span><span class="cbdg cb-ok" id="n1b">OK</span></div>
      <div class="mg">
        <div class="mc" id="m1c"><div class="ml_">Current</div><div class="mv" id="v1c">—<span class="mu">A</span></div></div>
        <div class="mc" id="m1v"><div class="ml_">Voltage</div><div class="mv" id="v1v">—<span class="mu">V</span></div></div>
        <div class="mc" id="m1p"><div class="ml_">Power</div><div class="mv" id="v1p">—<span class="mu">W</span></div></div>
        <div class="mc" id="m1t"><div class="ml_">Temperature</div><div class="mv" id="v1t">—<span class="mu">°C</span></div></div>
        <div class="mc" id="m1h"><div class="ml_">Humidity</div><div class="mv" id="v1h">—<span class="mu">%</span></div></div>
        <div class="mc" id="m1g"><div class="ml_">Gas</div><div class="mv" id="v1g">—<span class="mu">ppm</span></div></div>
      </div>
      <div class="div"></div><div class="sec">ML Classification</div>
      <div class="ml-panel" id="ml1"></div>
      <div class="div"></div><div class="sec">Fault status</div>
      <div class="fl" id="fl1"><div class="fi-n">No faults — nominal</div></div>
      <div class="rrow">
        <div class="rst"><div class="rdot rok" id="rd1"></div><span id="rl1">Relay: Normal</span></div>
        <div class="brow">
          <button class="btn b-cut" onclick="relayAction('node1','cut')">Cut power</button>
          <button class="btn b-rst" onclick="relayAction('node1','restore')">Restore</button>
        </div>
      </div>
    </div>
    <div class="card">
      <div class="chdr"><span class="ctitle">Node 2 — ACS712 + Voltage Divider</span><span class="cbdg cb-ok" id="n2b">OK</span></div>
      <div class="mg">
        <div class="mc" id="m2c"><div class="ml_">Current</div><div class="mv" id="v2c">—<span class="mu">A</span></div></div>
        <div class="mc" id="m2v"><div class="ml_">Voltage</div><div class="mv" id="v2v">—<span class="mu">V</span></div></div>
        <div class="mc" id="m2p"><div class="ml_">Power</div><div class="mv" id="v2p">—<span class="mu">W</span></div></div>
        <div class="mc" id="m2t"><div class="ml_">Temperature</div><div class="mv" id="v2t">—<span class="mu">°C</span></div></div>
        <div class="mc" id="m2h"><div class="ml_">Humidity</div><div class="mv" id="v2h">—<span class="mu">%</span></div></div>
        <div class="mc" id="m2g"><div class="ml_">Gas</div><div class="mv" id="v2g">—<span class="mu">ppm</span></div></div>
      </div>
      <div class="div"></div><div class="sec">ML Classification</div>
      <div class="ml-panel" id="ml2"></div>
      <div class="div"></div><div class="sec">Fault status</div>
      <div class="fl" id="fl2"><div class="fi-n">No faults — nominal</div></div>
      <div class="rrow">
        <div class="rst"><div class="rdot rok" id="rd2"></div><span id="rl2">Relay: Normal</span></div>
        <div class="brow">
          <button class="btn b-cut" onclick="relayAction('node2','cut')">Cut power</button>
          <button class="btn b-rst" onclick="relayAction('node2','restore')">Restore</button>
        </div>
      </div>
    </div>
  </div>
  <div class="g2">
    <div class="card"><div class="chdr"><span class="ctitle">Node 1 — Current &amp; Voltage</span></div>
      <div class="cbox" style="height:220px"><canvas id="cc1"></canvas></div></div>
    <div class="card"><div class="chdr"><span class="ctitle">Node 2 — Current &amp; Voltage</span></div>
      <div class="cbox" style="height:220px"><canvas id="cc2"></canvas></div></div>
  </div>
  <div class="g2">
    <div class="card"><div class="chdr"><span class="ctitle">Node 1 — Temperature &amp; Gas</span></div>
      <div class="cbox" style="height:200px"><canvas id="ct1"></canvas></div></div>
    <div class="card"><div class="chdr"><span class="ctitle">Node 2 — Temperature &amp; Gas</span></div>
      <div class="cbox" style="height:200px"><canvas id="ct2"></canvas></div></div>
  </div>
</div>

<!-- TAB: ML MODELS -->
<div class="page" id="tab-models">
  <div class="g2">
    <div class="card">
      <div class="chdr"><span class="ctitle">Gradient Boosting (XGBoost)</span>
        <span class="cbdg" id="gb-status-badge" style="background:var(--rd-bg);color:var(--rd)">Not loaded</span></div>
      <div style="font-size:.82rem;color:var(--t2);line-height:1.8;margin-bottom:14px">
        400 estimators · max_depth=6 · 17 engineered features → 11 classes</div>
      <div style="margin-bottom:10px">
        <div style="display:flex;justify-content:space-between;font-size:.75rem;color:var(--t3);margin-bottom:5px">
          <span>Accuracy</span><span id="gb-acc" style="color:var(--cy);font-weight:600">—</span></div>
        <div style="height:8px;background:var(--bg2);border-radius:4px;overflow:hidden;border:1px solid var(--bd)">
          <div id="gb-acc-bar" style="height:100%;border-radius:4px;background:var(--cy);width:0%;transition:width .6s"></div></div></div>
      <div class="g3" style="margin-bottom:0" id="gb-metrics"></div></div>
    <div class="card">
      <div class="chdr"><span class="ctitle">Random Forest</span>
        <span class="cbdg" id="rf-status-badge" style="background:var(--rd-bg);color:var(--rd)">Not loaded</span></div>
      <div style="font-size:.82rem;color:var(--t2);line-height:1.8;margin-bottom:14px">
        200 trees · max_depth=12 · Gini criterion · balanced class weights</div>
      <div style="margin-bottom:10px">
        <div style="display:flex;justify-content:space-between;font-size:.75rem;color:var(--t3);margin-bottom:5px">
          <span>Accuracy</span><span id="rf-acc" style="color:var(--pu);font-weight:600">—</span></div>
        <div style="height:8px;background:var(--bg2);border-radius:4px;overflow:hidden;border:1px solid var(--bd)">
          <div id="rf-acc-bar" style="height:100%;border-radius:4px;background:var(--pu);width:0%;transition:width .6s"></div></div></div>
      <div class="g3" style="margin-bottom:0" id="rf-metrics"></div></div>
  </div>
  <div class="g2">
    <div class="card"><div class="chdr"><span class="ctitle">Gradient Boosting — live confusion matrix</span></div>
      <div id="cm-gb"></div><div style="font-size:.75rem;color:var(--t3);margin-top:8px" id="cm-gb-stat">—</div></div>
    <div class="card"><div class="chdr"><span class="ctitle">Random Forest — live confusion matrix</span></div>
      <div id="cm-rf"></div><div style="font-size:.75rem;color:var(--t3);margin-top:8px" id="cm-rf-stat">—</div></div>
  </div>
  <div class="card">
    <div class="chdr"><span class="ctitle">11 classes &amp; fault color map</span></div>
    <div id="class-legend" style="display:flex;flex-wrap:wrap;gap:8px;margin-top:4px"></div></div>
</div>

<!-- TAB: STORAGE -->
<div class="page" id="tab-storage">
  <div class="g5" style="margin-bottom:16px">
    <div class="kpi"><div class="kv" id="db-tot">0</div><div class="kl">Total records</div></div>
    <div class="kpi"><div class="kv" id="db-flt">0</div><div class="kl">Fault records</div></div>
    <div class="kpi"><div class="kv" id="db-sp">0</div><div class="kl">Spoof events</div></div>
    <div class="kpi"><div class="kv" id="db-gba">0</div><div class="kl">GB detections</div></div>
    <div class="kpi"><div class="kv" id="db-rfa">0</div><div class="kl">RF detections</div></div>
  </div>
  <div class="card">
    <div class="fr">
      <select id="dn" onchange="dbL()"><option value="all">All nodes</option>
        <option value="node1">Node 1</option><option value="node2">Node 2</option></select>
      <input type="date" id="df" onchange="dbL()"/>
      <input type="date" id="dt" onchange="dbL()"/>
      <input type="number" id="dlim" value="100" min="10" max="5000" style="width:88px" onchange="dbL()"/>
      <button class="btn b-pri" onclick="dbL()">Refresh</button>
      <button class="btn b-pri" onclick="exportD('csv')">Export CSV</button>
      <span style="color:var(--t3);font-size:.78rem;align-self:center" id="dclbl">—</span>
    </div>
    <div class="tw"><table>
      <thead><tr><th>ID</th><th>Timestamp</th><th>Node</th><th>I(A)</th><th>V(V)</th>
        <th>P(W)</th><th>T°C</th><th>H%</th><th>Gas</th>
        <th>GB class</th><th>GB conf</th><th>RF class</th><th>RF conf</th>
        <th>Faults</th><th>Spoof</th></tr></thead>
      <tbody id="dtb"><tr><td colspan="15" style="text-align:center;color:var(--t3);padding:32px">Loading…</td></tr></tbody>
    </table></div>
    <div class="pag">
      <button class="pb" onclick="dbPg(-1)">← Prev</button>
      <span style="font-family:var(--mn);font-size:.72rem;color:var(--t3)" id="dpi">Page 1</span>
      <button class="pb" onclick="dbPg(1)">Next →</button>
    </div>
  </div>
</div>

<!-- TAB: ALERTS -->
<div class="page" id="tab-alerts">
  <div class="al-f">
    <div class="af active" onclick="setAF('all',this)">All</div>
    <div class="af" onclick="setAF('critical',this)">Critical</div>
    <div class="af" onclick="setAF('warning',this)">Warning</div>
    <div class="af" onclick="setAF('attack',this)">Attacks</div>
    <div style="flex:1"></div>
    <button class="btn b-pri" onclick="clrAl()">Clear display</button>
  </div>
  <div class="al-list" id="alist"><div class="noal">No alerts recorded</div></div>
</div>

<script>
const FAULT_LABELS=""" + _FAULT_LABELS_JS + r""";
const FAULT_COLORS=""" + _FAULT_COLORS_JS + r""";
const ATTACK_IDS=[6,7,8,9,10];
const FAULT_IDS=[1,2,3,4,5];
const socket=io();
let tmsg=0,alFil='all',alCnt=0,dbPgN=0,dbTot=0;
const fv=(v,d=2)=>(isNaN(+v)||v===null||v===undefined)?'—':(+v).toFixed(d);
const fUp=s=>{const h=Math.floor(s/3600),m=Math.floor((s%3600)/60),ss=s%60;
  return[h,m,ss].map(x=>String(x).padStart(2,'0')).join(':')};
function swTab(n,b){
  document.querySelectorAll('.page').forEach(p=>p.classList.remove('active'));
  document.querySelectorAll('.tbtn').forEach(x=>x.classList.remove('active'));
  document.getElementById('tab-'+n).classList.add('active');b.classList.add('active');
  if(n==='models')loadModels();if(n==='storage')dbL();
}
const CO={responsive:true,maintainAspectRatio:false,animation:{duration:0},
  plugins:{legend:{display:true,labels:{color:'#5a637a',font:{size:12},boxWidth:12,padding:14}}},
  scales:{x:{ticks:{color:'#9199ae',font:{size:11},maxTicksLimit:8},grid:{color:'rgba(0,0,0,.05)'}},
    y:{ticks:{color:'#9199ae',font:{size:11}},grid:{color:'rgba(0,0,0,.06)'},position:'left'},
    y2:{ticks:{color:'#9199ae',font:{size:11}},grid:{display:false},position:'right'}}};
function mkLC(id,ds){
  const o=JSON.parse(JSON.stringify(CO));o.scales.y2.display=ds.length>1;
  return new Chart(document.getElementById(id).getContext('2d'),{
    type:'line',data:{labels:[],datasets:ds.map(d=>({label:d.l,data:[],
      borderColor:d.c,borderWidth:2,pointRadius:0,tension:.35,fill:false,yAxisID:d.ax||'y'}))},options:o});
}
const MAXP=60;
const ch={c1:mkLC('cc1',[{l:'I(A)',c:'#0969da'},{l:'V(V)',c:'#6639ba',ax:'y2'}]),
          c2:mkLC('cc2',[{l:'I(A)',c:'#0969da'},{l:'V(V)',c:'#6639ba',ax:'y2'}]),
          t1:mkLC('ct1',[{l:'T°C',c:'#bc4c00'},{l:'Gas',c:'#0a7d6e',ax:'y2'}]),
          t2:mkLC('ct2',[{l:'T°C',c:'#bc4c00'},{l:'Gas',c:'#0a7d6e',ax:'y2'}])};
function pushC(c2,ts,vs){
  c2.data.labels.push(ts);vs.forEach((v,i)=>c2.data.datasets[i].data.push(v));
  if(c2.data.labels.length>MAXP){c2.data.labels.shift();c2.data.datasets.forEach(d=>d.data.shift());}
  c2.update('none');
}
function renderMLPanel(pid,ml){
  const el=document.getElementById(pid);if(!el)return;
  el.innerHTML=['gradient_boosting','random_forest'].map(key=>{
    const r=ml[key]||{};if(!Object.keys(r).length)return'';
    const short=key==='gradient_boosting'?'GB':'RF';
    const cls=r.label||0;const name=r.class_name||FAULT_LABELS[cls]||'Normal';
    const conf=r.confidence||0;const isA=ATTACK_IDS.includes(cls);const isF=FAULT_IDS.includes(cls);
    const col=isA?'var(--pu)':isF?'var(--rd)':'var(--gn)';
    const bg=isA?'var(--pu-bg)':isF?'var(--rd-bg)':'var(--gn-bg)';
    const bd=isA?'#c4b5fd':isF?'#f9a8a8':'#86efac';
    const cc=isA?'attack':isF?'fault':'';
    return`<div class="ml-card ${cc}" style="border-color:${bd};background:${bg}">
      <div class="ml-name">${short} — ${key.replace(/_/g,' ')}</div>
      <div class="ml-cls" style="color:${col}">${name}</div>
      <div class="ml-conf">Confidence: ${conf.toFixed(1)}%</div>
      <div class="ml-bar"><div class="ml-barfill" style="width:${conf}%;background:${col}"></div></div>
    </div>`;}).join('');
}
function updNode(node,data,ml,faults,relay,spoof){
  const n=node==='node1'?'1':'2';
  const ts=(data._ts||'').slice(11,19);
  [['m'+n+'c','v'+n+'c','current',3,'A',3,0],
   ['m'+n+'v','v'+n+'v','voltage',2,'V',20,3],
   ['m'+n+'p','v'+n+'p','power',2,'W',30,0],
   ['m'+n+'t','v'+n+'t','temperature',1,'°C',70,0],
   ['m'+n+'h','v'+n+'h','humidity',0,'%',95,10],
   ['m'+n+'g','v'+n+'g','gas',0,'ppm',1000,0],
  ].forEach(([cid,vid,key,dec,unit,hi,lo])=>{
    const cel=document.getElementById(cid);const vel=document.getElementById(vid);
    if(!cel||!vel)return;
    const v=parseFloat(data[key]);
    const ia=!isNaN(v)&&hi>0&&v>hi;const iw=!isNaN(v)&&lo>0&&v>0&&v<lo;
    cel.className='mc'+(ia?' al':iw?' wn':'');
    vel.innerHTML=isNaN(v)?`—<span class='mu'>${unit}</span>`:`${v.toFixed(dec)}<span class='mu'>${unit}</span>`;
  });
  pushC(ch['c'+n],ts,[parseFloat(data.current)||0,parseFloat(data.voltage)||0]);
  pushC(ch['t'+n],ts,[parseFloat(data.temperature)||0,parseFloat(data.gas)||0]);
  const bdg=document.getElementById('n'+n+'b');
  const gbr=ml.gradient_boosting||{};const rfr=ml.random_forest||{};
  const anyCls=(gbr.label||0)>0?(gbr.label||0):(rfr.label||0);
  if(spoof){bdg.textContent='Spoof';bdg.className='cbdg cb-sp';}
  else if(anyCls>0){
    bdg.textContent=FAULT_LABELS[anyCls]||'Fault';
    bdg.className='cbdg '+(ATTACK_IDS.includes(anyCls)?'cb-atk':FAULT_IDS.includes(anyCls)?'cb-err':'cb-warn');
  }else{bdg.textContent='OK';bdg.className='cbdg cb-ok';}
  renderMLPanel('ml'+n,ml);
  const fl=document.getElementById('fl'+n);
  if(!faults||!faults.length){fl.innerHTML='<div class="fi-n">No faults — nominal</div>';}
  else{const seen=new Set();fl.innerHTML=faults.filter(f=>{if(seen.has(f.type))return false;seen.add(f.type);return true;}).map(f=>{
    const isA=(f.type||'').match(/ATTACK|SPOOF|DRIFT|PULSE|PHYSICS|REPLAY/);
    const c2=f.severity==='critical'?(isA?'fi-a':'fi-c'):f.severity==='warning'?'fi-w':'fi-h';
    return`<div class="fi ${c2}">${f.type.replace(/_/g,' ')}: ${f.value} ${f.unit}</div>`;}).join('');}
  const cut=relay==='cut';
  document.getElementById('rd'+n).className='rdot '+(cut?'rcu':'rok');
  document.getElementById('rl'+n).textContent=cut?'Relay: CUT':'Relay: Normal';
  tmsg++;document.getElementById('mcnt').textContent=tmsg+' msgs';
}
function loadModels(){
  fetch('/api/model/accuracy').then(r=>r.json()).then(d=>{
    const a=d.accuracy||{};const m=d.metrics||{};
    const gbAcc=a.gradient_boosting||0;
    document.getElementById('gb-acc').textContent=gbAcc>0?(gbAcc*100).toFixed(1)+'%':'Not trained';
    document.getElementById('gb-acc-bar').style.width=(gbAcc*100)+'%';
    document.getElementById('gb-status-badge').textContent=d.gb_loaded?'Loaded':'Not loaded';
    document.getElementById('gb-status-badge').style.cssText=d.gb_loaded?
      'background:var(--gn-bg);color:var(--gn)':'background:var(--rd-bg);color:var(--rd)';
    document.getElementById('gbs').textContent=d.gb_loaded?'✓':'✗';
    document.getElementById('gbs').style.color=d.gb_loaded?'var(--gn)':'var(--rd)';
    const gm=m.gradient_boosting||{};
    document.getElementById('gb-metrics').innerHTML=
      ['precision','recall','f1'].map(k=>`<div class="kpi" style="padding:12px">
        <div class="kv" style="font-size:1.2rem">${gm[k]!=null?(gm[k]*100).toFixed(1)+'%':'—'}</div>
        <div class="kl">${k}</div></div>`).join('');
    const rfAcc=a.random_forest||0;
    document.getElementById('rf-acc').textContent=rfAcc>0?(rfAcc*100).toFixed(1)+'%':'Not trained';
    document.getElementById('rf-acc-bar').style.width=(rfAcc*100)+'%';
    document.getElementById('rf-status-badge').textContent=d.rf_loaded?'Loaded':'Not loaded';
    document.getElementById('rf-status-badge').style.cssText=d.rf_loaded?
      'background:var(--gn-bg);color:var(--gn)':'background:var(--rd-bg);color:var(--rd)';
    document.getElementById('rfs').textContent=d.rf_loaded?'✓':'✗';
    document.getElementById('rfs').style.color=d.rf_loaded?'var(--gn)':'var(--rd)';
    const rm=m.random_forest||{};
    document.getElementById('rf-metrics').innerHTML=
      ['precision','recall','f1'].map(k=>`<div class="kpi" style="padding:12px">
        <div class="kv" style="font-size:1.2rem">${rm[k]!=null?(rm[k]*100).toFixed(1)+'%':'—'}</div>
        <div class="kl">${k}</div></div>`).join('');
    document.getElementById('class-legend').innerHTML=Object.entries(FAULT_LABELS).map(([k,v])=>
      `<span style="display:inline-flex;align-items:center;gap:5px;padding:4px 10px;border-radius:5px;
        font-size:.75rem;font-weight:500;background:${FAULT_COLORS[k]}22;
        color:${FAULT_COLORS[k]};border:1px solid ${FAULT_COLORS[k]}55">
        <span style="width:8px;height:8px;border-radius:50%;background:${FAULT_COLORS[k]};display:inline-block"></span>
        ${k}: ${v}</span>`).join('');
  });
  fetch('/api/state').then(r=>r.json()).then(d=>{
    const cm=d.confusion||{};
    ['gradient_boosting','random_forest'].forEach(mn=>{
      const pfx=mn==='gradient_boosting'?'gb':'rf';
      const m2=cm[mn]||{tp:0,tn:0,fp:0,fn:0};
      document.getElementById('cm-'+pfx).innerHTML=`<div class="cm-grid">
        <div class="cm-cell cm-tp"><div class="cm-l">True Positive</div><div class="cm-v">${m2.tp}</div></div>
        <div class="cm-cell cm-fp"><div class="cm-l">False Positive</div><div class="cm-v">${m2.fp}</div></div>
        <div class="cm-cell cm-fn"><div class="cm-l">False Negative</div><div class="cm-v">${m2.fn}</div></div>
        <div class="cm-cell cm-tn"><div class="cm-l">True Negative</div><div class="cm-v">${m2.tn}</div></div></div>`;
      const tot=m2.tp+m2.tn+m2.fp+m2.fn;
      const acc=tot>0?((m2.tp+m2.tn)/tot*100).toFixed(1)+'%':'—';
      document.getElementById('cm-'+pfx+'-stat').textContent=
        `Live accuracy: ${acc}  TP:${m2.tp} TN:${m2.tn} FP:${m2.fp} FN:${m2.fn}`;
    });
  });
}
function dbL(){
  const node=document.getElementById('dn').value;
  const from=document.getElementById('df').value;const to=document.getElementById('dt').value;
  const lim=document.getElementById('dlim').value||100;
  let url=`/api/history/${node}?limit=${lim}&offset=${dbPgN*parseInt(lim)}`;
  if(from)url+=`&from=${from}`;if(to)url+=`&to=${to}`;
  fetch(url).then(r=>r.json()).then(d=>{
    dbTot=d.total;const rows=d.rows||[];
    document.getElementById('dclbl').textContent=`${rows.length} of ${dbTot.toLocaleString()} records`;
    document.getElementById('dpi').textContent=`Page ${dbPgN+1} of ${Math.ceil(dbTot/parseInt(lim))||1}`;
    document.getElementById('dtb').innerHTML=rows.length===0?
      `<tr><td colspan="15" style="text-align:center;color:var(--t3);padding:32px">No records</td></tr>`:
      rows.map(r=>{
        const gc=r.gb_class,rc2=r.rf_class;
        const gN=FAULT_LABELS[gc]||'—';const rN=FAULT_LABELS[rc2]||'—';
        const gcol=gc>0?(ATTACK_IDS.includes(gc)?'var(--pu)':'var(--rd)'):'var(--gn)';
        const rcol=rc2>0?(ATTACK_IDS.includes(rc2)?'var(--pu)':'var(--rd)'):'var(--gn)';
        return`<tr>
          <td style="color:var(--t3)">${r.id}</td>
          <td style="font-family:var(--mn);font-size:.75rem">${(r.ts||'').slice(0,19).replace('T',' ')}</td>
          <td><span class="np ${r.node==='node1'?'np1':'np2'}">${r.node}</span></td>
          <td>${fv(r.current,3)}</td><td>${fv(r.voltage,2)}</td><td>${fv(r.power,2)}</td>
          <td>${fv(r.temperature,1)}</td><td>${fv(r.humidity,0)}</td><td>${fv(r.gas,0)}</td>
          <td><span class="cls-tag" style="color:${gcol};border-color:${gcol}40">${gc>=0?gN:'—'}</span></td>
          <td style="font-family:var(--mn)">${fv(r.gb_conf,1)}%</td>
          <td><span class="cls-tag" style="color:${rcol};border-color:${rcol}40">${rc2>=0?rN:'—'}</span></td>
          <td style="font-family:var(--mn)">${fv(r.rf_conf,1)}%</td>
          <td class="${r.n_faults>0?'cbad':''}">${r.n_faults||0}</td>
          <td>${r.spoof?'<span style="color:var(--pu);font-weight:700">Yes</span>':'—'}</td>
        </tr>`;}).join('');
  });
}
function dbPg(d2){const lim=parseInt(document.getElementById('dlim').value)||100;
  dbPgN=Math.max(0,Math.min(dbPgN+d2,Math.ceil(dbTot/lim)-1));dbL();}
function exportD(fmt){const n=document.getElementById('dn').value,lim=document.getElementById('dlim').value||5000;
  window.location.href=`/api/export/${fmt}?node=${n}&limit=${lim}`;}
const _seen=new Set();
function pushAlert(a){
  const key=(a.node||'')+(a.type||'')+(a.ts||'').slice(0,19);
  if(_seen.has(key))return;_seen.add(key);
  alCnt++;document.getElementById('acnt').textContent=alCnt;
  const sev=a.severity||'medium';
  const isAtk=(a.type||'').match(/ATTACK|SPOOF|DRIFT|PULSE|PHYSICS|REPLAY|INJECTION/);
  const cls=isAtk?'ai-a':sev==='critical'?'ai-c':sev==='warning'?'ai-w':'ai-h';
  const show=(alFil==='all')||(alFil==='critical'&&sev==='critical')||
    (alFil==='warning'&&sev==='warning')||(alFil==='attack'&&isAtk);
  const el=document.createElement('div');el.className=`ai ${cls}`;
  el.dataset.sev=sev;el.dataset.atk=isAtk?'1':'0';
  el.innerHTML=`<div class="an">${(a.node||'').toUpperCase()}</div>
    <div class="at">${(a.type||'').replace(/_/g,' ')}</div>
    <div class="av">${a.value||''} ${a.unit||''}</div>
    <div class="ats">${(a.ts||'').slice(11,19)}</div>`;
  if(!show)el.style.display='none';
  const lst=document.getElementById('alist');
  const emp=lst.querySelector('.noal');if(emp)emp.remove();
  lst.insertBefore(el,lst.firstChild);
  while(lst.children.length>300)lst.lastChild.remove();
}
function setAF(f2,b){alFil=f2;
  document.querySelectorAll('.af').forEach(x=>x.classList.remove('active'));b.classList.add('active');
  document.querySelectorAll('.ai').forEach(el=>{
    const s=el.dataset.sev,at=el.dataset.atk==='1';
    el.style.display=((f2==='all')||(f2==='critical'&&s==='critical')||
      (f2==='warning'&&s==='warning')||(f2==='attack'&&at))?'':'none';});}
function clrAl(){document.getElementById('alist').innerHTML='<div class="noal">Display cleared</div>';
  _seen.clear();alCnt=0;document.getElementById('acnt').textContent='0';}
socket.on('connect',()=>{document.getElementById('cpill').className='pill ok';document.getElementById('clbl').textContent='Connected';});
socket.on('disconnect',()=>{document.getElementById('cpill').className='pill err';document.getElementById('clbl').textContent='Disconnected';});
socket.on('sensor_update',d=>{
  updNode(d.node,d.data,d.ml,d.faults,d.relay,d.spoof);
  if(d.faults&&d.faults.length)d.faults.forEach(f=>pushAlert({node:d.node,ts:d.data._ts,...f}));
});
socket.on('relay_update',d=>{
  const n=d.node==='node1'?'1':'2';const cut=d.relay==='cut';
  document.getElementById('rd'+n).className='rdot '+(cut?'rcu':'rok');
  document.getElementById('rl'+n).textContent=cut?'Relay: CUT':'Relay: Normal';
});
socket.on('alert',d=>pushAlert(d));
socket.on('node_restored',d=>{
  const n=d.node==='node1'?'1':'2';
  document.getElementById('rd'+n).className='rdot rok';
  document.getElementById('rl'+n).textContent='Relay: Normal';
  document.getElementById('fl'+n).innerHTML='<div class="fi-n">Restored — nominal</div>';
  document.getElementById('n'+n+'b').textContent='OK';
  document.getElementById('n'+n+'b').className='cbdg cb-ok';
  document.getElementById('ml'+n).innerHTML='';
});
const relayAction=(node,action)=>fetch('/api/relay',{method:'POST',
  headers:{'Content-Type':'application/json'},body:JSON.stringify({node,action})})
  .then(r=>r.json()).then(d=>{if(d.ok){
    const n=node==='node1'?'1':'2';const cut=action==='cut';
    document.getElementById('rd'+n).className='rdot '+(cut?'rcu':'rok');
    document.getElementById('rl'+n).textContent=cut?'Relay: CUT':'Relay: Normal';}});
function refreshStats(){
  fetch('/api/stats').then(r=>r.json()).then(s=>{
    document.getElementById('k-tot').textContent=(s.total_readings||0).toLocaleString();
    document.getElementById('k-flt').textContent=(s.total_faults||0).toLocaleString();
    document.getElementById('k-sp').textContent=(s.spoof_count||0)+' spoof events';
    document.getElementById('k-gba').textContent=(s.gb_anomalies||0).toLocaleString();
    document.getElementById('k-rfa').textContent=(s.rf_anomalies||0).toLocaleString();
    document.getElementById('db-tot').textContent=(s.total_readings||0).toLocaleString();
    document.getElementById('db-flt').textContent=(s.total_faults||0).toLocaleString();
    document.getElementById('db-sp').textContent=(s.spoof_count||0).toLocaleString();
    document.getElementById('db-gba').textContent=(s.gb_anomalies||0).toLocaleString();
    document.getElementById('db-rfa').textContent=(s.rf_anomalies||0).toLocaleString();
  });
  fetch('/api/state').then(r=>r.json()).then(s=>{
    document.getElementById('ucnt').textContent='Up: '+fUp(s.uptime||0);
    document.getElementById('k-up').textContent=fUp(s.uptime||0);
    if(s.node1?.latest&&Object.keys(s.node1.latest).length)
      updNode('node1',s.node1.latest,s.node1.ml,s.node1.faults,s.node1.relay,s.node1.spoof_active);
    if(s.node2?.latest&&Object.keys(s.node2.latest).length)
      updNode('node2',s.node2.latest,s.node2.ml,s.node2.faults,s.node2.relay,s.node2.spoof_active);
    (s.alerts||[]).slice(0,40).forEach(a=>pushAlert(a));
  });
  fetch('/api/model/accuracy').then(r=>r.json()).then(d=>{
    document.getElementById('gbs').textContent=d.gb_loaded?'✓ Loaded':'✗ Not loaded';
    document.getElementById('gbs').style.color=d.gb_loaded?'var(--gn)':'var(--rd)';
    document.getElementById('rfs').textContent=d.rf_loaded?'✓ Loaded':'✗ Not loaded';
    document.getElementById('rfs').style.color=d.rf_loaded?'var(--gn)':'var(--rd)';
  });
}
['node1','node2'].forEach((node,ni)=>fetch('/api/stream_history/'+node).then(r=>r.json()).then(pts=>{
  pts.forEach(p=>{
    if(ni===0){pushC(ch.c1,p.ts,[p.current,p.voltage]);pushC(ch.t1,p.ts,[p.temperature,p.gas]);}
    else{pushC(ch.c2,p.ts,[p.current,p.voltage]);pushC(ch.t2,p.ts,[p.temperature,p.gas]);}
  });
}));
refreshStats();
setInterval(refreshStats,10000);
</script>
</body></html>"""

# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("="*65)
    print("  ESP32 Edge AI Monitor — RF + Gradient Boosting")
    print(f"  Classes : 11 (Normal + 5 faults + 5 attacks)")
    print(f"  Features: 17 engineered")
    print(f"  DB      : {DB_PATH}")
    print(f"  MQTT    : {MQTT_BROKER}:{MQTT_PORT}")
    print(f"  Web     : http://0.0.0.0:5000")
    print("="*65)
    print(f"  GB: {'loaded' if models['gradient_boosting'] else 'run train_gradient_boosting.py first'}")
    print(f"  RF: {'loaded' if models['random_forest'] else 'run train_random_forest.py first'}")
    print()
    start_mqtt()
    socketio.run(app, host="0.0.0.0", port=5000, debug=False, use_reloader=False)
