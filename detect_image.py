"""
detect_image.py — Detect and read license plates from a single image.

Usage:
    python detect_image.py --image car.jpg

Forte Bilgi İletişim Teknolojileri A.Ş. Internship · August 2023
"""

import argparse
import cv2
import os
from paddleocr import PaddleOCR

from utils.plate_utils import yolo_det, get_output_layer_names, crop, resize_bbox

# ── Config ────────────────────────────────────────────────────────────────────
WEIGHTS    = "weights/yolov4-custom.weights"
CONFIG     = "weights/yolov4-custom.cfg"
CONF_THRESH = 0.5
NMS_THRESH  = 0.4
BATCH_SIZE  = 4
THRESHOLD   = 0.5


def load_model():
    net = cv2.dnn.readNetFromDarknet(CONFIG, WEIGHTS)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    return net


def test_img(image_path, ocr, net, output_names):
    """
    Detect plate in image, crop it, apply SRN OCR, annotate and return result.

    Returns:
        annotated image, detected plate text
    """
    frame = cv2.imread(image_path)
    if frame is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    bboxes = yolo_det(net, output_names, frame, CONF_THRESH, NMS_THRESH)

    plate_text = "Not detected"

    for bbox in bboxes:
        # Rescale bbox to original image size
        scaled = resize_bbox(bbox, frame.shape)
        plate_crop = crop(frame, scaled)

        if plate_crop.size == 0:
            continue

        # Run SRN OCR
        result = ocr.ocr(plate_crop, cls=True)
        if result and result[0]:
            texts = [line[1][0] for line in result[0] if line[1][1] > THRESHOLD]
            plate_text = " ".join(texts)

        # Draw bounding box and text
        xmin, ymin, xmax, ymax = scaled
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(frame, plate_text, (xmin, ymin - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    return frame, plate_text


def run(args):
    os.makedirs("output", exist_ok=True)

    print("Loading YOLOv4...")
    net          = load_model()
    output_names = get_output_layer_names(net)

    print("Loading SRN OCR...")
    ocr = PaddleOCR(use_angle_cls=True, lang="en", rec_algorithm="SRN", show_log=False)

    annotated, plate_text = test_img(args.image, ocr, net, output_names)

    out_path = os.path.join("output", os.path.basename(args.image))
    cv2.imwrite(out_path, annotated)
    print(f"Detected plate: {plate_text}")
    print(f"Result saved  → {out_path}")

    cv2.imshow("Plate Detection", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True, help="Path to input image")
    run(p.parse_args())
