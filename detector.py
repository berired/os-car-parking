import argparse
import json
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Tuple

import cv2
import numpy as np

Point = Tuple[int, int]


def preprocess_gray(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gray = cv2.equalizeHist(gray)
    return gray


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
    diff_threshold = int(detection_cfg.get("diff_threshold", 25))
    change_ratio_threshold = float(detection_cfg.get("change_ratio_threshold", 0.12))
    edge_ratio_threshold = float(detection_cfg.get("edge_ratio_threshold", 0.035))
    object_area_ratio_threshold = float(detection_cfg.get("object_area_ratio_threshold", 0.08))

    empty_gray = preprocess_gray(empty_frame)
    empty_edges = cv2.Canny(empty_gray, 50, 150)

    masks = build_masks(empty_gray.shape, slots)
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

        gray = preprocess_gray(frame)
        edges = cv2.Canny(gray, 50, 150)
        diff = cv2.absdiff(gray, empty_gray)

        open_slots: List[str] = []
        occupied_slots: List[str] = []

        for slot in slots:
            label = str(slot["id"])
            polygon = np.array(slot["polygon"], dtype=np.int32)
            mask = masks[label]

            area = int(cv2.countNonZero(mask))
            if area == 0:
                continue

            # Color-invariant occupancy cues: intensity change, new edge content, and contiguous changed object size.
            changed = cv2.bitwise_and((diff > diff_threshold).astype(np.uint8) * 255, mask)
            changed = cv2.morphologyEx(changed, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            changed = cv2.morphologyEx(changed, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            changed_ratio = cv2.countNonZero(changed) / float(area)

            edge_new = cv2.bitwise_and(cv2.bitwise_and(edges, cv2.bitwise_not(empty_edges)), mask)
            edge_ratio = cv2.countNonZero(edge_new) / float(area)

            contours, _ = cv2.findContours(changed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            max_object_area = max((cv2.contourArea(c) for c in contours), default=0.0)
            object_area_ratio = max_object_area / float(area)

            occupied_raw = (
                changed_ratio > change_ratio_threshold
                or edge_ratio > edge_ratio_threshold
                or object_area_ratio > object_area_ratio_threshold
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
