Automated Gait-Based Human Identification System

This project identifies individuals based on their walking patterns using pose estimation and machine learning models. It processes webcam video input in real-time and detects human gait structure using a pre-trained deep learning model.

---

Project Structure

```
gait-identification/
│
├── model/
│   └── keras/
│       └── model.h5               # Pretrained pose estimation model
│
├── config                        # Configuration file (for config_reader.py)
├── utils/                        # Contains util.py and color mappings
├── processing.py                 # Contains extract_parts() and draw()
├── config_reader.py              # Loads the config and model parameters
├── main.py                       # (Rename your main script if needed)
├── requirements.txt
└── README.md
```

---

Download the Pretrained Keras Model through the below link:-
https://www.dropbox.com/s/llpxd14is7gyj0z/model.h5

System Requirements

- Python 3.7+
- Webcam-enabled system (USB or built-in)
- OpenCV support

---

Required Python Libraries

Create a `requirements.txt` with:

```
opencv-python
numpy
scipy
configobj
```

Then install:

```bash
pip install -r requirements.txt
```

---

How to Run

1. Clone the repository:
```bash
git clone https://github.com/your-username/gait-identification.git
cd gait-identification
```

2. Ensure the model is placed here:
```
model/keras/model.h5
```

3. Run the main script:
```bash
python df03a41a-706c-4edf-a1a9-abb84f32feb3.py --device 0 --model model/keras/model.h5
```

4. Webcam will open. You will see:
```
✅ Human detected
❌ No human detected
```

5. Press `q` to quit the live detection window.

---

What the Code Does

- Loads a pre-trained pose estimation model
- Reads webcam input frame-by-frame
- Detects human pose using pose estimation
- Tracks and displays humans with labeled keypoints
- Displays and optionally saves the processed output

---

Optional Parameters

```bash
--frame_ratio 7              # Analyze every 7th frame
--process_speed 2            # Speed vs accuracy tradeoff
--out_name output_filename   # Save output video
--mirror False               # Don't mirror webcam input
```

---

Author

**Nidheesh Kumar Nissankula**  
B.Tech in CSE – Specialization in AI & Robotics  
[LinkedIn](https://www.linkedin.com/in/nidheesh-kumar-nissankula-58a4972a8)
