"""
plate_utils.py — Core utility functions for license plate detection.

Forte Bilgi İletişim Teknolojileri A.Ş. Internship · August 2023
"""

import cv2
import numpy as np


# ── Detection helpers ─────────────────────────────────────────────────────────

def resize_bbox(bbox, orig_shape, model_size=(416, 416)):
    """
    Rescale bounding box coordinates from model output resolution
    back to the original image dimensions.

    The detector internally resizes images to model_size before inference;
    this function inverts that scaling so bboxes align with the original image.

    Args:
        bbox        : (xmin, ymin, xmax, ymax) in model coordinate space.
        orig_shape  : (height, width) of the original image.
        model_size  : (width, height) the detector resizes inputs to.

    Returns:
        Rescaled (xmin, ymin, xmax, ymax) as integers.
    """
    orig_h, orig_w = orig_shape[:2]
    model_w, model_h = model_size

    scale_x = orig_w / model_w
    scale_y = orig_h / model_h

    xmin = int(bbox[0] * scale_x)
    ymin = int(bbox[1] * scale_y)
    xmax = int(bbox[2] * scale_x)
    ymax = int(bbox[3] * scale_y)

    # Clamp to image bounds
    xmin = max(0, xmin); ymin = max(0, ymin)
    xmax = min(orig_w, xmax); ymax = min(orig_h, ymax)

    return xmin, ymin, xmax, ymax


def crop(image, bbox):
    """
    Crop the detected plate region from the image.

    Args:
        image : Original BGR image (numpy array).
        bbox  : (xmin, ymin, xmax, ymax) in original image coordinates.

    Returns:
        Cropped plate image (numpy array).
    """
    xmin, ymin, xmax, ymax = bbox
    return image[ymin:ymax, xmin:xmax]


def yolo_det(net, output_names, frame, conf_thresh=0.5, nms_thresh=0.4,
             model_size=(416, 416)):
    """
    Run YOLOv4 inference on a single frame and return plate bounding boxes.

    Args:
        net          : cv2.dnn network loaded with YOLOv4 weights.
        output_names : Output layer names from get_output_layer_names().
        frame        : BGR image (numpy array).
        conf_thresh  : Minimum confidence to keep a detection.
        nms_thresh   : IoU threshold for Non-Maximum Suppression.
        model_size   : (width, height) the network expects.

    Returns:
        List of (xmin, ymin, xmax, ymax) tuples in original image coordinates.
    """
    h, w = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(
        frame, 1 / 255.0, model_size, swapRB=True, crop=False
    )
    net.setInput(blob)
    layer_outputs = net.forward(output_names)

    boxes, confidences = [], []

    for output in layer_outputs:
        for detection in output:
            scores     = detection[5:]
            confidence = float(np.max(scores))
            if confidence > conf_thresh:
                cx = int(detection[0] * w)
                cy = int(detection[1] * h)
                bw = int(detection[2] * w)
                bh = int(detection[3] * h)
                xmin = max(0, cx - bw // 2)
                ymin = max(0, cy - bh // 2)
                boxes.append([xmin, ymin, bw, bh])
                confidences.append(confidence)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_thresh, nms_thresh)
    result  = []
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, bw, bh = boxes[i]
            result.append((x, y, x + bw, y + bh))

    return result


def get_output_layer_names(net):
    """Return names of YOLO output layers."""
    layer_names = net.getLayerNames()
    return [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]


# ── OCR benchmark helper ──────────────────────────────────────────────────────

def score_calc(predictions, ground_truths):
    """
    Calculate OCR benchmark score (lower = better).

    Computes character-level edit distance (Levenshtein) normalized by
    the ground truth length, then averages across all samples.

    Args:
        predictions  : List of predicted plate strings.
        ground_truths: List of ground truth plate strings.

    Returns:
        Average normalized edit distance (float).
    """
    assert len(predictions) == len(ground_truths), "Lists must have equal length."

    scores = []
    for pred, gt in zip(predictions, ground_truths):
        dist  = _levenshtein(pred.upper(), gt.upper())
        norm  = dist / max(len(gt), 1)
        scores.append(norm)

    avg = sum(scores) / len(scores)
    return round(avg, 3)


def _levenshtein(s1, s2):
    """Standard dynamic-programming Levenshtein distance."""
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1): dp[i][0] = i
    for j in range(n + 1): dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + cost)
    return dp[m][n]
