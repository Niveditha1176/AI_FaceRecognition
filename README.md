# FaceRecognition

A lightweight, software-only face recognition system using OpenCV's Local Binary Patterns Histograms (LBPH). This project allows you to capture face data, train a local model, and run real-time recognition via a desktop webcam.

Note: This repository was adapted from an Arduino-integrated project. All hardware dependencies have been removed for a pure software implementation.

---

## 📂Repository Structure

```
Face recognition
├── app.py                # Real-time recognition (uses trainer.yml & labels.json)
├── capture_faces.py      # Captures labeled face images to dataset/
├── train_model.py        # Trains LBPH model; generates trainer.yml & labels.json
├── camera_test.py        # Diagnostic tool for webcam access
├── dataset/              # Created by capture_faces.py (contains person subfolders)
├── trainer.yml           # Trained model file 
└── labels.json           # ID-to-Name mapping (created during training)
```
---

## 🛠️ Requirements & Installation

**Python:** 3.8+ (3.10+ recommended)

Environment: Virtual environment recommended

1. Setup Environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

2. Install Dependencies

```bash
pip install --upgrade pip
pip install opencv-python opencv-contrib-python numpy
```

Note: `opencv-contrib-python` is strictly required for the LBPH modules.

---

## 🚀 Basic Usage

Follow these three steps in order to get the system running:

### Step 1: Capture Labeled Faces

Run the capture script to build your dataset.

```bash
python capture_faces.py
```

- **Input:** Enter the name of the person when prompted.
- **Samples:** Default is 50. For better accuracy, aim for 200–500 images.
- **Controls:** Press `Enter` to finish early or `q` to cancel.

### Step 2: Train the Model

Process the images in the `dataset/` folder into a machine-readable model.

```bash
python train_model.py
```

This assigns numeric IDs to folders and generates `trainer.yml` and `labels.json`.

### Step 3: Run Recognition

Start the live desktop demo.

```bash
python app.py
```

- The script uses your webcam to identify faces in real-time.
- **Controls:** Press `q` to exit the window.
  
---

## 📈 Improving Accuracy

- **Sample Size:** Capture 500+ images per person.
- **Variety:** Capture faces under different lighting, different angles, and with/without glasses.
- **Thresholding:** Adjust the `CONFIDENCE_THRESHOLD` in `app.py`. A lower number makes the system "stricter" about who it recognizes.
 
---
## 🔧 Troubleshooting

| Issue | Solution |
|---|---|
| `trainer.yml not found` | Run `train_model.py` first. Ensure `dataset/` is not empty. |
| Camera not opening | Change the index in `cv2.VideoCapture(0)` to `1` or `2`. |
| `AttributeError: 'cv2' has no 'face'` | Install `opencv-contrib-python`. |
| Low Accuracy | Increase dataset size, improve lighting, and ensure faces are cropped/resized properly. |

---
## 🔒 Privacy & Safety

- **Local Storage:** All face data and models are stored locally on your machine.
- **Consent:** Do not capture or store face data of individuals without their explicit consent.
- **Security:** If using in a production environment, ensure `trainer.yml` and `labels.json` are stored securely.
---

## 📜 License

This project is licensed under the MIT License.

---
## 📸 Screenshots of Output
- **facial recognition of existing datasets:**

<img width="3061" height="1813" alt="image" src="https://github.com/user-attachments/assets/82e338b7-e525-4e92-8da1-5dd4d7747436" />

<img width="1872" height="1060" alt="image" src="https://github.com/user-attachments/assets/f6813def-58d9-43b1-92d0-b4991af02526" />

- **setting up a new facial recognition :**

<img width="1697" height="588" alt="image" src="https://github.com/user-attachments/assets/5f97a58e-07ba-4911-9299-db5e92ce3e9f" />



