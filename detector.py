import argparse
import json
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Tuple

import cv2
import numpy as np

Point = Tuple[int, int]


def preprocess_for_detection(frame: np.ndarray, target_gray_mean: float) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    current_mean = float(np.mean(gray))
    if current_mean > 1.0:
        alpha = target_gray_mean / current_mean
        alpha = max(0.6, min(1.6, alpha))
        gray = cv2.convertScaleAbs(gray, alpha=alpha, beta=0)
    blur = cv2.GaussianBlur(gray, (3, 3), 1)
    threshold = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        25,
        16,
    )
    median = cv2.medianBlur(threshold, 5)
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(median, kernel, iterations=1)
    return dilated


def build_masks(shape: Tuple[int, int], slots: List[Dict[str, object]]) -> Dict[str, np.ndarray]:
    masks: Dict[str, np.ndarray] = {}
    for slot in slots:
        label = str(slot["id"])
        polygon = np.array(slot["polygon"], dtype=np.int32)
        mask = np.zeros(shape, dtype=np.uint8)
        cv2.fillPoly(mask, [polygon], 255)
        masks[label] = mask
    return masks


def main() -> None:
    parser = argparse.ArgumentParser(description="Parking slot car detector")
    parser.add_argument("--config", default="parking_calibration.json", help="Calibration JSON file")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default: 0)")
    parser.add_argument("--window", default="Parking Detector", help="Display window name")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_file():
        print(f"Config not found: {config_path}")
        return

    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)

    empty_frame_path = Path(config.get("empty_frame", ""))
    if not empty_frame_path.is_file():
        print(f"Empty reference frame not found: {empty_frame_path}")
        return

    empty_frame = cv2.imread(str(empty_frame_path))
    if empty_frame is None:
        print(f"Failed to read empty reference frame: {empty_frame_path}")
        return

    slots = config.get("slots", [])
    if not slots:
        print("No slots found in calibration config.")
        return

    detection_cfg = config.get("detection", {})
    nonzero_ratio_threshold = float(
        detection_cfg.get(
            "nonzero_ratio_threshold",
            detection_cfg.get("change_ratio_threshold", 0.175),
        )
    )
    nonzero_delta_threshold = float(detection_cfg.get("nonzero_delta_threshold", 0.06))

    empty_gray_mean = float(np.mean(cv2.cvtColor(empty_frame, cv2.COLOR_BGR2GRAY)))
    empty_processed = preprocess_for_detection(empty_frame, empty_gray_mean)

    masks = build_masks(empty_frame.shape[:2], slots)
    slot_baseline_ratios: Dict[str, float] = {}
    for slot in slots:
        label = str(slot["id"])
        mask = masks[label]
        area = int(cv2.countNonZero(mask))
        if area == 0:
            slot_baseline_ratios[label] = 0.0
            continue
        slot_empty_binary = cv2.bitwise_and(empty_processed, mask)
        slot_baseline_ratios[label] = cv2.countNonZero(slot_empty_binary) / float(area)

    slot_histories: Dict[str, Deque[bool]] = {
        str(slot["id"]): deque(maxlen=5) for slot in slots
    }

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Could not open webcam index: {args.camera}")
        return

    cv2.namedWindow(args.window, cv2.WINDOW_NORMAL)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame.shape[:2] != empty_frame.shape[:2]:
            frame = cv2.resize(frame, (empty_frame.shape[1], empty_frame.shape[0]))

        processed = preprocess_for_detection(frame, empty_gray_mean)

        open_slots: List[str] = []
        occupied_slots: List[str] = []

        for slot in slots:
            label = str(slot["id"])
            polygon = np.array(slot["polygon"], dtype=np.int32)
            mask = masks[label]

            area = int(cv2.countNonZero(mask))
            if area == 0:
                continue

            slot_binary = cv2.bitwise_and(processed, mask)
            nonzero_ratio = cv2.countNonZero(slot_binary) / float(area)
            baseline_ratio = slot_baseline_ratios.get(label, 0.0)
            occupied_raw = (
                nonzero_ratio > nonzero_ratio_threshold
                and (nonzero_ratio - baseline_ratio) > nonzero_delta_threshold
            )

            slot_histories[label].append(occupied_raw)
            occupied = sum(slot_histories[label]) >= 3

            if occupied:
                occupied_slots.append(label)
                color = (0, 0, 255)
            else:
                open_slots.append(label)
                color = (0, 200, 0)

            cv2.polylines(frame, [polygon], isClosed=True, color=color, thickness=2)
            cx, cy = np.mean(polygon, axis=0).astype(int)
            cv2.putText(frame, label, (cx - 12, cy + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

        open_text = ", ".join(open_slots) if open_slots else "-"
        cv2.rectangle(frame, (10, 10), (frame.shape[1] - 10, 95), (0, 0, 0), -1)
        cv2.putText(
            frame,
            f"Open slots: {len(open_slots)}/{len(slots)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"IDs: {open_text}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow(args.window, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
