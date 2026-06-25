import cv2
import time
from datetime import datetime
from ultralytics import YOLO
from twilio.rest import Client
from dotenv import load_dotenv
import os
import threading
import winsound

load_dotenv()

TWILIO_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_FROM = os.getenv("TWILIO_FROM_NUMBER")
ALERT_TO = os.getenv("ALERT_TO_NUMBER")

model = YOLO("fire_model.pt")
FIRE_CLASS_NAMES = ["fire", "smoke", "other"]

SMS_COOLDOWN = 30
last_sms_time = 0

def get_zone(x_center, y_center, frame_w, frame_h):
    col = "left" if x_center < frame_w / 2 else "right"
    row = "top" if y_center < frame_h / 2 else "bottom"
    return f"{row}-{col}"

def send_sms(zone, confidence):
    global last_sms_time
    now = time.time()
    if now - last_sms_time < SMS_COOLDOWN:
        return
    last_sms_time = now
    client = Client(TWILIO_SID, TWILIO_TOKEN)
    msg = (
        f"FIRE DETECTED\n"
        f"Zone: {zone}\n"
        f"Confidence: {confidence:.0%}\n"
        f"Time: {datetime.now().strftime('%H:%M:%S')}"
    )
    client.messages.create(body=msg, from_=TWILIO_FROM, to=ALERT_TO)
    print(f"[SMS SENT] {msg}")

def play_alarm() :
    for _ in range(5):
        winsound.Beep(1000,500)
        time.sleep(0.2)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
print("Fire detection running... Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)
    fire_detected = False

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id].lower()
            confidence = float(box.conf[0])

            if any(name in label for name in FIRE_CLASS_NAMES) and confidence > 0.3:
                fire_detected = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                zone = get_zone(cx, cy, frame.shape[1], frame.shape[0])

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"FIRE {confidence:.0%} | {zone}",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 0, 255), 2)
                send_sms(zone, confidence)

    status = "FIRE DETECTED!" if fire_detected else "Monitoring..."
    color = (0, 0, 255) if fire_detected else (0, 255, 0)
    cv2.putText(frame, status, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.imshow("Fire Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()