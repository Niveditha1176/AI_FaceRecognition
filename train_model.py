import cv2
import os
import numpy as np
import json

dataset_path = 'dataset'
recognizer = cv2.face.LBPHFaceRecognizer_create()

faces, ids, label_dict = [], [], {}

print(">> Starting training...")

if not os.path.exists(dataset_path):
    print(f">> No dataset found at '{dataset_path}'. Run capture_faces.py first.")
    raise SystemExit(1)

person_dirs = [d for d in sorted(os.listdir(dataset_path)) if os.path.isdir(os.path.join(dataset_path, d))]
if not person_dirs:
    print(">> No person folders in dataset. Add faces with capture_faces.py")
    raise SystemExit(1)

for idx, dir_name in enumerate(person_dirs):
    person_dir = os.path.join(dataset_path, dir_name)
    print(f">> Processing '{dir_name}' (label {idx})")
    files = sorted(os.listdir(person_dir))
    found = 0
    for file in files:
        img_path = os.path.join(person_dir, file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        faces.append(img)
        ids.append(idx)
        found += 1
    if found == 0:
        print(f">> No valid images in {person_dir}, skipping.")
        continue
    label_dict[idx] = dir_name

if not faces:
    print(">> No training images found. Aborting.")
    raise SystemExit(1)

recognizer.train(faces, np.array(ids))
recognizer.save("trainer.yml")

with open("labels.json", "w") as f:
    json.dump(label_dict, f)

print("\n✅ Training complete!")
print(">> Model saved as 'trainer.yml'")
print(">> Labels saved as 'labels.json'")
