import cv2
import os

dataset_path = 'dataset'
person_name = input("Enter name for this person: ").strip()
if not person_name:
    print("Name cannot be empty.")
    raise SystemExit(1)

try:
    requested = input("Number of images to capture (default 50): ").strip()
    TARGET = int(requested) if requested else 50
    if TARGET < 10 or TARGET > 2000:
        print("Please choose a number between 10 and 2000.")
        raise SystemExit(1)
except ValueError:
    print("Invalid number.")
    raise SystemExit(1)

person_folder = os.path.join(dataset_path, person_name)
os.makedirs(person_folder, exist_ok=True)

face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Use CAP_DSHOW on Windows for more reliable capture
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
count = 0
print(f"📸 Capturing up to {TARGET} images for '{person_name}'. Press Enter to finish early, 'q' to cancel.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Camera frame not available")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        if count >= TARGET:
            break
        count += 1
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (200, 200))
        filename = os.path.join(person_folder, f"{count:03d}.jpg")
        cv2.imwrite(filename, face)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, f"Image: {count}/{TARGET}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow('Capturing Faces', frame)
    key = cv2.waitKey(1)
    # Enter to finish, 'q' to cancel
    if key == 13 or count >= TARGET:
        break
    if key & 0xFF == ord('q'):
        print("Cancelled by user.")
        break

cap.release()
cv2.destroyAllWindows()
print(f"\n✅ {count} images saved to {person_folder}")
