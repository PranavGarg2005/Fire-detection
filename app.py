import cv2
import time
import threading
from datetime import datetime
from flask import Flask, Response, render_template_string, jsonify
from ultralytics import YOLO
from twilio.rest import Client
from dotenv import load_dotenv
import os

load_dotenv()

TWILIO_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_FROM = os.getenv("TWILIO_FROM_NUMBER")
ALERT_TO = os.getenv("ALERT_TO_NUMBER")

model = YOLO("fire_model.pt")
FIRE_CLASS_NAMES = ["fire", "smoke", "other"]
SMS_COOLDOWN = 60

state = {
    "fire_detected": False,
    "last_detection": None,
    "confidence": 0,
    "zone": None,
    "alerts": [],
    "last_sms_time": 0
}

app = Flask(__name__)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
output_frame = None
lock = threading.Lock()

def get_zone(cx, cy, w, h):
    col = "Left" if cx < w / 2 else "Right"
    row = "Top" if cy < h / 2 else "Bottom"
    return f"{row}-{col}"

def send_sms(zone, confidence):
    now = time.time()
    if now - state["last_sms_time"] < SMS_COOLDOWN:
        return
    state["last_sms_time"] = now
    try:
        client = Client(TWILIO_SID, TWILIO_TOKEN)
        msg = (f"🔥 FIRE DETECTED\nZone: {zone}\n"
               f"Confidence: {confidence:.0%}\n"
               f"Time: {datetime.now().strftime('%H:%M:%S')}")
        client.messages.create(body=msg, from_=TWILIO_FROM, to=ALERT_TO)
        print(f"[SMS SENT] {msg}")
    except Exception as e:
        print(f"[SMS ERROR] {e}")

def detect_loop():
    global output_frame
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        results = model(frame, verbose=False)
        fire_detected = False
        best_conf = 0
        best_zone = None

        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                label = model.names[cls_id].lower()
                confidence = float(box.conf[0])

                print(f"Detected: {label} | Confidence: {confidence:.2f}")

                if any(n in label for n in FIRE_CLASS_NAMES) and confidence > 0.3:
                    fire_detected = True
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx, cy = (x1+x2)//2, (y1+y2)//2
                    zone = get_zone(cx, cy, frame.shape[1], frame.shape[0])

                    if confidence > best_conf:
                        best_conf = confidence
                        best_zone = zone

                    color = (0, 0, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{label.upper()} {confidence:.0%}",
                                (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, color, 2)

        state["fire_detected"] = fire_detected
        state["confidence"] = best_conf
        state["zone"] = best_zone

        if fire_detected:
            now_str = datetime.now().strftime("%H:%M:%S")
            state["last_detection"] = now_str
            state["alerts"].insert(0, {
                "time": now_str,
                "zone": best_zone,
                "confidence": f"{best_conf:.0%}"
            })
            state["alerts"] = state["alerts"][:20]
            send_sms(best_zone, best_conf)

        _, buffer = cv2.imencode(".jpg", frame)
        with lock:
            output_frame = buffer.tobytes()

def generate():
    while True:
        with lock:
            if output_frame is None:
                continue
            frame = output_frame
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

@app.route("/video")
def video():
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/status")
def status():
    return jsonify(state)

@app.route("/")
def index():
    return render_template_string(HTML)

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Fire Detection System</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: #0f0f0f; color: #fff; font-family: 'Segoe UI', sans-serif; min-height: 100vh; }

  header {
    display: flex; align-items: center; gap: 12px;
    padding: 16px 24px;
    background: #1a1a1a;
    border-bottom: 1px solid #2a2a2a;
  }
  header h1 { font-size: 18px; font-weight: 600; }
  .dot { width: 10px; height: 10px; border-radius: 50%; background: #22c55e; animation: pulse 1.5s infinite; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }

  .grid {
    display: grid;
    grid-template-columns: 1fr 320px;
    grid-template-rows: auto auto;
    gap: 16px;
    padding: 16px 24px;
    max-width: 1200px;
    margin: 0 auto;
  }

  .card {
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-radius: 12px;
    padding: 16px;
  }
  .card-title {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #666;
    margin-bottom: 12px;
  }

  /* Camera feed */
  .feed-card { grid-row: 1 / 3; }
  .feed-card img {
    width: 100%; border-radius: 8px;
    border: 2px solid #2a2a2a;
    display: block;
  }
  .feed-card.alert img { border-color: #ef4444; box-shadow: 0 0 20px rgba(239,68,68,0.3); }

  /* Status card */
  .status-banner {
    display: flex; align-items: center; gap: 12px;
    padding: 14px 16px;
    border-radius: 8px;
    background: #111;
    border: 1px solid #2a2a2a;
    margin-bottom: 12px;
    transition: all 0.3s;
  }
  .status-banner.fire { background: rgba(239,68,68,0.1); border-color: #ef4444; }
  .status-icon { font-size: 24px; }
  .status-label { font-size: 15px; font-weight: 600; }
  .status-sub { font-size: 12px; color: #888; margin-top: 2px; }

  /* Stats */
  .stats { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 12px; }
  .stat { background: #111; border-radius: 8px; padding: 12px; border: 1px solid #2a2a2a; }
  .stat-label { font-size: 11px; color: #666; margin-bottom: 4px; }
  .stat-value { font-size: 20px; font-weight: 600; color: #fff; }

  /* Zone map */
  .zone-grid {
    display: grid; grid-template-columns: 1fr 1fr;
    grid-template-rows: 1fr 1fr;
    gap: 6px; height: 120px;
    margin-top: 8px;
  }
  .zone-cell {
    border-radius: 6px; background: #111;
    border: 1px solid #2a2a2a;
    display: flex; align-items: center; justify-content: center;
    font-size: 11px; color: #555; transition: all 0.3s;
  }
  .zone-cell.active { background: rgba(239,68,68,0.2); border-color: #ef4444; color: #ef4444; font-weight: 600; }

  /* Alert log */
  .alert-log { max-height: 180px; overflow-y: auto; }
  .alert-item {
    display: flex; align-items: center; gap: 10px;
    padding: 8px 0; border-bottom: 1px solid #1f1f1f;
    font-size: 13px;
  }
  .alert-time { color: #666; font-size: 11px; min-width: 60px; font-family: monospace; }
  .alert-zone { color: #ef4444; font-weight: 500; }
  .alert-conf { margin-left: auto; color: #888; font-size: 11px; }
  .no-alerts { color: #444; font-size: 13px; text-align: center; padding: 20px 0; }

  .conf-bar { height: 4px; background: #2a2a2a; border-radius: 2px; margin-top: 8px; overflow: hidden; }
  .conf-fill { height: 100%; background: #ef4444; border-radius: 2px; transition: width 0.5s; width: 0%; }
</style>
</head>
<body>

<header>
  <div class="dot" id="live-dot"></div>
  <h1>🔥 Fire Detection System</h1>
  <span style="margin-left:auto;font-size:12px;color:#555;" id="clock"></span>
</header>

<div class="grid">

  <!-- Camera Feed -->
  <div class="card feed-card" id="feed-card">
    <div class="card-title">Live Camera Feed</div>
    <img src="/video" alt="Live feed" />
  </div>

  <!-- Status -->
  <div class="card">
    <div class="card-title">Detection Status</div>
    <div class="status-banner" id="status-banner">
      <span class="status-icon" id="status-icon">✅</span>
      <div>
        <div class="status-label" id="status-label">Monitoring</div>
        <div class="status-sub" id="status-sub">No fire detected</div>
      </div>
    </div>
    <div class="conf-bar"><div class="conf-fill" id="conf-fill"></div></div>
    <div class="stats">
      <div class="stat">
        <div class="stat-label">Confidence</div>
        <div class="stat-value" id="conf-val">—</div>
      </div>
      <div class="stat">
        <div class="stat-label">Last Alert</div>
        <div class="stat-value" style="font-size:14px" id="last-alert">—</div>
      </div>
    </div>
  </div>

  <!-- Zone Map -->
  <div class="card">
    <div class="card-title">Detection Zone</div>
    <div class="zone-grid">
      <div class="zone-cell" id="zone-top-left">Top-Left</div>
      <div class="zone-cell" id="zone-top-right">Top-Right</div>
      <div class="zone-cell" id="zone-bottom-left">Bottom-Left</div>
      <div class="zone-cell" id="zone-bottom-right">Bottom-Right</div>
    </div>
  </div>

  <!-- Alert Log -->
  <div class="card">
    <div class="card-title">Alert History</div>
    <div class="alert-log" id="alert-log">
      <div class="no-alerts">No alerts yet</div>
    </div>
  </div>

</div>

<script>
  const zones = ["top-left","top-right","bottom-left","bottom-right"];

  function tick() {
    document.getElementById("clock").textContent = new Date().toLocaleTimeString();
  }
  setInterval(tick, 1000); tick();

  async function update() {
    try {
      const res = await fetch("/status");
      const d = await res.json();

      const fire = d.fire_detected;
      const banner = document.getElementById("status-banner");
      const feedCard = document.getElementById("feed-card");

      banner.className = "status-banner" + (fire ? " fire" : "");
      feedCard.className = "card feed-card" + (fire ? " alert" : "");
      document.getElementById("status-icon").textContent = fire ? "🔥" : "✅";
      document.getElementById("status-label").textContent = fire ? "FIRE DETECTED!" : "Monitoring";
      document.getElementById("status-sub").textContent = fire
        ? `Zone: ${d.zone} · ${(d.confidence*100).toFixed(0)}% confidence`
        : "No fire detected";

      const confPct = fire ? Math.round(d.confidence * 100) : 0;
      document.getElementById("conf-val").textContent = fire ? confPct + "%" : "—";
      document.getElementById("conf-fill").style.width = confPct + "%";
      document.getElementById("last-alert").textContent = d.last_detection || "—";

      zones.forEach(z => {
        const el = document.getElementById("zone-" + z.toLowerCase().replace("-","-"));
        el.className = "zone-cell" + (fire && d.zone && d.zone.toLowerCase() === z ? " active" : "");
      });

      const log = document.getElementById("alert-log");
      if (d.alerts && d.alerts.length > 0) {
        log.innerHTML = d.alerts.map(a => `
          <div class="alert-item">
            <span class="alert-time">${a.time}</span>
            <span class="alert-zone">🔥 ${a.zone}</span>
            <span class="alert-conf">${a.confidence}</span>
          </div>`).join("");
      }
    } catch(e) {}
  }

  setInterval(update, 800);
  update();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    t = threading.Thread(target=detect_loop, daemon=True)
    t.start()
    print("Dashboard running at http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)