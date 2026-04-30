# 🚗 License Plate Detection

> **Automatic license plate detection and recognition using YOLOv4 + SRN OCR + DeepSORT tracking.**

---

## 📖 Overview

This project implements an end-to-end **Automatic License Plate Recognition (ALPR)** pipeline:

1. **Detection** — YOLOv4 trained on 1,500 images, achieving **90% mAP** after 3,000 epochs
2. **Recognition** — SRN OCR selected after benchmarking 3 OCR tools (PP-OCR, CRNN, SRN)
3. **Tracking** — DeepSORT for persistent plate tracking across video frames

**Use cases:** traffic enforcement, parking management, toll collection.

---

## 🏗️ Pipeline

```
Image / Video Input
        │
        ▼
YOLOv4 Plate Detector
(bounding box → plate region)
crop() → resize_bbox()
        │
        ▼
SRN OCR
(character recognition on cropped plate)
        │
        ▼
DeepSORT Tracker
(avoid re-reading the same plate every frame)
        │
        ▼
Annotated Output + Plate Text
```

---

## 📊 OCR Benchmark

Three OCR tools were evaluated using `score_calc()` (lower = better):

| OCR Tool | Score |
|----------|-------|
| CRNN     | 20.68 |
| PP-OCR   | 3.162 |
| **SRN**  | **0.774** ✅ |

**SRN was selected** as it achieved the lowest error score.

---

## 📊 Detection Training

| Parameter | Value |
|-----------|-------|
| Architecture | YOLOv4 (CSPDarknet53) |
| Training images | 1,500 |
| Validation images | 300 |
| Epochs | 3,000 |
| Training time | ~5.3 hours |
| mAP | **90%** |

---

## 📁 Project Structure

```
license-plate-detection/
│
├── utils/
│   └── plate_utils.py          # crop, resize_bbox, yolo_det, score_calc
│
├── data/
│   └── obj/                    # Training images (YOLO format, not included)
│
├── weights/                    # YOLOv4 weights (not included, see Setup)
│   ├── yolov4-custom.weights
│   └── yolov4-custom.cfg
│
├── output/                     # Output images and videos
│
├── detect_image.py             # Test on single image
├── detect_video.py             # Test on video with DeepSORT tracking
└── README.md
```

---

## ⚙️ Setup

```bash
git clone https://github.com/busesaricayir/license-plate-detection.git
cd license-plate-detection
pip install -r requirements.txt

# Clone DeepSORT
git clone https://github.com/nwojke/deep_sort.git
```

---

## 🚀 Usage

```bash
# Test on image
python detect_image.py --image path/to/car.jpg

# Test on video with tracking
python detect_video.py --input path/to/video.mp4 --output output/result.mp4
```

---

## 📦 Requirements

```
opencv-python
numpy
torch
paddlepaddle
paddleocr
```

---

## 📄 References

- Bochkovskiy et al. (2020). *YOLOv4: Optimal speed and accuracy of object detection.* arXiv:2004.10934
- Du, S. et al. (2013). *Automatic License Plate Recognition (ALPR): A State-of-the-Art Review.* IEEE TCSVT.

---

<div align="center"></div>
