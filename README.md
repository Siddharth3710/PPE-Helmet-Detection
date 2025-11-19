🛡️ PPE Helmet Detection System
Real-time Helmet / No-Helmet + White/Yellow Helmet Classification
A complete Computer Vision Case Study for Vinfotech

🚀 Overview
This repository contains an end-to-end PPE Helmet Detection System that runs on real CCTV-style industrial videos.
The system performs:


Person Detection (Faster R-CNN – COCO pretrained)


Head Region Extraction (top-ROI cropping)


Helmet vs No-Helmet Classification (custom-trained ResNet-18)


Helmet Color Classification (White / Yellow)
using a multi-region color analysis pipeline (HSV + LAB + RGB)


Real-time visualization with bounding boxes and labels


CPU-optimized inference with progress logs, FPS counter, and ETA


Pause/Resume/Skip Frame controls


MP4 output generation + automatic head crop saving


This project is designed under the constraints of a CPU-only laptop, ensuring efficiency without requiring a GPU.

🧠 A. Problem Understanding
In industrial environments (construction, factories, warehouses), worker safety is often compromised when people ignore wearing PPE helmets. The goal is to create a robust CV system capable of:


Detecting workers in CCTV video (18–25m distance)


Determining whether each person is wearing a helmet


Classifying helmet color (white / yellow) despite challenging lighting


Real-World Challenges Addressed


CCTV Distance (18–25m):
Low-resolution head regions → requires careful cropping and robust color features.


White Helmets Blending Into Background:
Bright walls/ceilings/windows make white helmets nearly invisible → solved via a hybrid CNN + color-space heuristic.


Lighting & Domain Shift:
Dataset (GDUT-HWD) vs test videos (warehouse/office) differ in lighting, angle, resolution → handled using a stable head-crop ROI and multi-region color voting.


Compute Constraints:
Entire pipeline optimized for CPU-only execution with real-time display, progress logs, and minimal overhead.



🏗️ B. Architecture
Pipeline Flow Diagram
┌──────────────────────────┐
│        Input Video        │
└──────────────┬───────────┘
               │
               ▼
┌──────────────────────────┐
│  Person Detection (FRCNN)│
└──────────────┬───────────┘
               │  Person Box (Green)
               ▼
┌──────────────────────────┐
│    Head ROI Extraction    │
│ (Top region of person box)│
└──────────────┬───────────┘
               │  Head Box (Blue)
               ▼
┌──────────────────────────┐
│ Helmet Classifier (ResNet18)
│    → helmet / no_helmet   │
└──────────────┬───────────┘
               │
               ▼
┌──────────────────────────┐
│  Advanced Color Detection │
│ HSV + LAB + RGB + Region │
│     → white / yellow      │
└──────────────┬───────────┘
               │
               ▼
┌──────────────────────────┐
│ Final Decision Fusion     │
│  - Color > CNN override   │
│  - Confidence shown       │
└──────────────┬───────────┘
               │
               ▼
┌──────────────────────────┐
│ Visualization & MP4 Save │
│ BBoxes + Labels + RealTime│
└──────────────────────────┘


⚙️ C. Key Implementation Choices
1️⃣ Detection Model — Faster R-CNN ResNet50-FPN


Strong baseline detector trained on COCO


Excellent person detection performance


Works well even when people are small or partially visible


CPU-friendly when used frame-by-frame


2️⃣ Head Cropping Strategy (Blue Box)
Instead of using full person ROI:


Only the top region of the person box is extracted


Head height is dynamically scaled using person width


Helps isolate helmet even when the body is noisy


Blue rectangular box drawn for clarity


Ensures classifier receives consistent helmet-only input


3️⃣ Helmet Classification Method


Trained a binary ResNet-18 on GDUT-HWD head crops


Lightweight, fast, and accurate for CPU inference


Uses ImageNet normalization


Produces probability scores (softmax confidence)


4️⃣ Advanced Color Detection (White / Yellow)
A simple HSV threshold is not enough for CCTV.
So I implemented a multi-region, multi-color-space voting system:


Uses:


HSV (Hue/Saturation/Brightness)


LAB (Lightness, chromatic channels for white detection)


RGB balance


Pixel brightness distribution




Checks 3 separate head sub-regions


Uses a vote-based scoring system for stability


Color overrides the CNN decision (stronger evidence)



📦 D. Dataset & Training
Dataset Used


GDUT-HWD Helmet Detection Dataset (via Roboflow)


Downloaded in COCO JSON format


Custom preprocessing to:


Extract helmet and no-helmet head crops


Balance the dataset (initial dataset heavily imbalanced)




Training


Model: ResNet-18


Loss: Cross Entropy


Optimizer: Adam


Epochs: ~10


Achieved ~80% validation accuracy (small dataset + imbalance)



🖥️ E. Running Inference
Requirements
pip install torch torchvision opencv-python numpy pillow

Run the script
python helmet_detector_final.py

Optional real-time display controls


q → quit


p → pause/resume


SPACE → skip frame


Output video saved as:
w1.mp4

Head crops saved to:
head_crops/


📝 F. Results


Accurate detection of yellow helmets


Improved detection of white helmets even in bright warehouses


Stabilized confidence scores


Clear bounding boxes:


Green → person


Blue → head ROI




Label examples:
white_helmet (0.87)
yellow_helmet (0.92)
helmet (0.65)
no_helmet (0.78)



Include sample frames in GitHub for better presentation.

🧩 G. Repository Structure
📂 PPE-Helmet-Detection
│── helmet_detector_final.py
│── resnet18_helmet_binary.pth
│── requirements.txt
│── README.md
│── head_crops/
│── sample_videos/
└── outputs/


🔮 H. Future Improvements

Train multi-class classifier (helmet color included)
Use YOLOv8 for faster detection if GPU becomes available
Temporal smoothing across video frames
Better white helmet dataset expansion



❤️ Thanks
This project was prepared as part of the Vinfotech Computer Vision Case Study
and demonstrates practical application of deep learning for industrial safety monitoring.

