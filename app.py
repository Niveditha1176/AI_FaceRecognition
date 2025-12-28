import cv2
import numpy as np
import json
import time
import os

CONFIDENCE_THRESHOLD = 60

# --- Load trained model ---
if not os.path.exists("trainer.yml"):
    print("❌ trainer.yml not found. Run train_model.py after capturing faces.")
    raise SystemExit(1)
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

# --- Load labels (expecting a mapping of id -> name) ---
if not os.path.exists("labels.json"):
    print("❌ labels.json not found. Run train_model.py after capturing faces.")
    raise SystemExit(1)
with open("labels.json", "r") as f:
    raw_labels = json.load(f)
    # ensure keys are ints
    labels = {int(k): v for k, v in raw_labels.items()}

# --- Initialize Haar Cascade ---
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# --- Start Video Capture ---
cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # CAP_DSHOW fixes some Windows webcam issues

# --- Delay Line ---
time.sleep(2)
if not cam.isOpened():
    print("❌ Camera not detected! Try changing the index (0 → 1 or 2).")
    exit()

print("🔍 Face recognition started... Press 'q' to quit.")

while True:
    ret, frame = cam.read()
    if not ret:
        print("⚠️ Failed to read frame from camera.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        id_, conf = recognizer.predict(roi_gray)

        if conf < CONFIDENCE_THRESHOLD:
            name = labels.get(id_, "Unknown")
            color = (0, 255, 0)
            label_text = f"{name} ({round(conf, 2)})"
            print(f"✅ Recognized: {name} (conf={round(conf,2)})")
        else:
            name = "Unknown"
            color = (0, 0, 255)
            label_text = "Unknown"
            print(f"❌ Unknown (conf={round(conf,2)})")

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # (No hardware) show label on the frame only

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
print("👋 Program closed.")
