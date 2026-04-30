"""
detect_video.py — License plate detection + DeepSORT tracking on video.

DeepSORT ensures the same plate is not re-read every frame — once a plate
is detected and a tracker is assigned, OCR runs only on the first detection.
Subsequent frames reuse the saved plate text until the tracker is lost.

Usage:
    python detect_video.py --input video.mp4 --output output/result.mp4

Forte Bilgi İletişim Teknolojileri A.Ş. Internship · August 2023
"""

import argparse
import os
import time

import cv2
import numpy as np
from paddleocr import PaddleOCR

from utils.plate_utils import yolo_det, get_output_layer_names, crop, resize_bbox

# ── Config ────────────────────────────────────────────────────────────────────
WEIGHTS     = "weights/yolov4-custom.weights"
CONFIG      = "weights/yolov4-custom.cfg"
CONF_THRESH  = 0.5
NMS_THRESH   = 0.4
OCR_THRESHOLD = 0.5
FONT = cv2.FONT_HERSHEY_SIMPLEX


def get_best_ocr(plate_crop, ocr, threshold=OCR_THRESHOLD):
    """
    Run SRN OCR on a plate crop and return the highest-confidence text.

    Returns:
        Best plate string or empty string if nothing detected.
    """
    result = ocr.ocr(plate_crop, cls=True)
    if not result or not result[0]:
        return ""
    candidates = [(line[1][0], line[1][1]) for line in result[0]]
    if not candidates:
        return ""
    best_text, best_conf = max(candidates, key=lambda x: x[1])
    return best_text if best_conf > threshold else ""


def load_model():
    net = cv2.dnn.readNetFromDarknet(CONFIG, WEIGHTS)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    return net


def run(args):
    os.makedirs("output", exist_ok=True)

    print("Loading YOLOv4...")
    net          = load_model()
    output_names = get_output_layer_names(net)

    print("Loading SRN OCR (PaddleOCR)...")
    ocr = PaddleOCR(use_angle_cls=True, lang="en", rec_algorithm="SRN", show_log=False)

    # ── Load DeepSORT ─────────────────────────────────────────────────────────
    try:
        from deep_sort.deep_sort import DeepSort
        tracker = DeepSort(model_path="deep_sort/deep/checkpoint/ckpt.t7")
        use_tracker = True
        print("DeepSORT loaded.")
    except ImportError:
        print("DeepSORT not found — running without tracking (clone from https://github.com/nwojke/deep_sort)")
        use_tracker = False
        tracker = None

    # ── Open video ────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {args.input}")

    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25

    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (fw, fh))

    # {track_id: plate_text} — cache so we don't re-run OCR every frame
    track_plate_cache = {}

    print("Processing...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0     = time.time()
        bboxes = yolo_det(net, output_names, frame, CONF_THRESH, NMS_THRESH)

        if use_tracker and bboxes:
            # DeepSORT expects [x, y, w, h] format + confidence list
            ds_bboxes = []
            for (x1, y1, x2, y2) in bboxes:
                ds_bboxes.append([x1, y1, x2 - x1, y2 - y1])
            confs = [1.0] * len(ds_bboxes)
            tracks = tracker.update(np.array(ds_bboxes), confs, frame)

            for track in tracks:
                x1, y1, x2, y2, track_id = map(int, track)
                plate_crop = crop(frame, resize_bbox((x1, y1, x2, y2), frame.shape))

                # OCR only on first sighting of this track_id
                if track_id not in track_plate_cache and plate_crop.size > 0:
                    track_plate_cache[track_id] = get_best_ocr(plate_crop, ocr)

                plate_text = track_plate_cache.get(track_id, "")
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID:{track_id} {plate_text}",
                            (x1, y1 - 8), FONT, 0.7, (0, 0, 255), 2)
        else:
            # No tracker — run OCR on every detection
            for bbox in bboxes:
                scaled     = resize_bbox(bbox, frame.shape)
                plate_crop = crop(frame, scaled)
                plate_text = get_best_ocr(plate_crop, ocr) if plate_crop.size > 0 else ""
                x1, y1, x2, y2 = scaled
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, plate_text, (x1, y1 - 8), FONT, 0.7, (0, 0, 255), 2)

        fps_label = f"FPS: {1 / (time.time() - t0):.1f}"
        cv2.putText(frame, fps_label, (10, 30), FONT, 0.8, (255, 255, 0), 2)

        writer.write(frame)
        cv2.imshow("Plate Detection + Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"Output saved → {args.output}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input",  required=True)
    p.add_argument("--output", default="output/result.mp4")
    run(p.parse_args())
