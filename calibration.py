import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

Point = Tuple[int, int]
Polygon = List[Point]


class CalibrationUI:
    def __init__(self, frame: np.ndarray, slot_labels: List[str]) -> None:
        self.frame = frame
        self.slot_labels = slot_labels
        self.slots: Dict[str, Polygon] = {label: [] for label in slot_labels}
        self.current_index = 0
        self.mode = "draw"
        self.drag_target: Optional[Tuple[str, int]] = None

    @property
    def current_label(self) -> Optional[str]:
        if self.current_index >= len(self.slot_labels):
            return None
        return self.slot_labels[self.current_index]

    def all_complete(self) -> bool:
        return all(len(points) == 4 for points in self.slots.values())

    def find_nearest_vertex(self, x: int, y: int, radius: int = 14) -> Optional[Tuple[str, int]]:
        closest: Optional[Tuple[str, int]] = None
        best_dist_sq = radius * radius

        for label, points in self.slots.items():
            for idx, point in enumerate(points):
                dx = point[0] - x
                dy = point[1] - y
                dist_sq = dx * dx + dy * dy
                if dist_sq <= best_dist_sq:
                    best_dist_sq = dist_sq
                    closest = (label, idx)
        return closest

    def on_mouse(self, event: int, x: int, y: int, flags: int, param: object) -> None:
        if self.mode == "draw":
            if event == cv2.EVENT_LBUTTONDOWN and self.current_label is not None:
                points = self.slots[self.current_label]
                if len(points) < 4:
                    points.append((x, y))
                if len(points) == 4:
                    self.current_index += 1
            return

        # Edit mode: drag any existing point.
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag_target = self.find_nearest_vertex(x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.drag_target is not None:
            label, point_idx = self.drag_target
            self.slots[label][point_idx] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drag_target = None

    def draw_overlay(self) -> np.ndarray:
        canvas = self.frame.copy()

        for label in self.slot_labels:
            points = self.slots[label]
            color = (70, 200, 70) if len(points) == 4 else (50, 180, 220)

            if points:
                for idx, point in enumerate(points):
                    cv2.circle(canvas, point, 5, color, -1)
                    cv2.putText(
                        canvas,
                        str(idx + 1),
                        (point[0] + 7, point[1] - 7),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        color,
                        1,
                        cv2.LINE_AA,
                    )

            if len(points) >= 2:
                for i in range(len(points) - 1):
                    cv2.line(canvas, points[i], points[i + 1], color, 2)

            if len(points) == 4:
                cv2.line(canvas, points[-1], points[0], color, 2)
                poly = np.array(points, dtype=np.int32)
                center = tuple(np.mean(poly, axis=0).astype(int))
                cv2.putText(
                    canvas,
                    label,
                    center,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

        mode_text = f"Mode: {self.mode.upper()}"
        progress = sum(1 for p in self.slots.values() if len(p) == 4)
        progress_text = f"Progress: {progress}/{len(self.slot_labels)} slots"
        current_text = f"Current: {self.current_label if self.current_label else 'done'}"

        cv2.putText(canvas, mode_text, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 30, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, progress_text, (15, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, current_text, (15, 79), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

        hint_lines = [
            "DRAW: Click 4 points per slot (clockwise or counterclockwise)",
            "TAB: toggle draw/edit, U: undo current slot point, R: reset current slot",
            "EDIT: drag a corner point to refine slot polygon",
            "S: save config, Q: quit",
        ]
        y = canvas.shape[0] - 85
        for line in hint_lines:
            cv2.putText(canvas, line, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (235, 235, 235), 1, cv2.LINE_AA)
            y += 22

        return canvas


def excel_col_name(index: int) -> str:
    # 0 -> a, 25 -> z, 26 -> aa
    name = ""
    value = index
    while True:
        value, rem = divmod(value, 26)
        name = chr(ord("a") + rem) + name
        if value == 0:
            break
        value -= 1
    return name


def build_slot_labels(rows: int, cols: int) -> List[str]:
    labels: List[str] = []
    for c in range(cols):
        col_name = excel_col_name(c)
        for r in range(1, rows + 1):
            labels.append(f"{col_name}{r}")
    return labels


def capture_frame(camera_index: int) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        return None

    captured_frame: Optional[np.ndarray] = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        display = frame.copy()
        cv2.putText(
            display,
            "Press C to capture empty parking frame, Q to quit",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow("Capture Empty Parking", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("c"):
            captured_frame = frame.copy()
            break
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyWindow("Capture Empty Parking")
    return captured_frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Parking slot calibration tool")
    parser.add_argument("--output", default="parking_calibration.json", help="Output calibration JSON file")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default: 0)")
    parser.add_argument(
        "--empty-frame",
        default="calibration_empty.png",
        help="Path to save the reference empty parking image",
    )
    args = parser.parse_args()

    try:
        rows = int(input("How many rows? (1,2,3...): ").strip())
        cols = int(input("How many columns? (1,2,3...): ").strip())
    except ValueError:
        print("Rows/columns must be integers.")
        return

    if rows <= 0 or cols <= 0:
        print("Rows/columns must be > 0.")
        return

    frame = capture_frame(args.camera)
    if frame is None:
        print(f"Could not open webcam index: {args.camera}")
        return

    slot_labels = build_slot_labels(rows, cols)
    print("Slot labels:", ", ".join(slot_labels))

    ui = CalibrationUI(frame, slot_labels)

    window = "Parking Calibration"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, ui.on_mouse)

    while True:
        canvas = ui.draw_overlay()
        cv2.imshow(window, canvas)

        key = cv2.waitKey(20) & 0xFF
        if key == ord("\t"):
            ui.mode = "edit" if ui.mode == "draw" else "draw"
        elif key == ord("u") and ui.current_label is not None:
            if ui.slots[ui.current_label]:
                ui.slots[ui.current_label].pop()
        elif key == ord("r") and ui.current_label is not None:
            ui.slots[ui.current_label] = []
        elif key == ord("s"):
            if not ui.all_complete():
                print("Please define all slot polygons (4 points each) before saving.")
                continue
            break
        elif key == ord("q"):
            cv2.destroyAllWindows()
            print("Calibration canceled.")
            return

    cv2.destroyAllWindows()

    output_path = Path(args.output)
    empty_frame_path = Path(args.empty_frame)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    empty_frame_path.parent.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(empty_frame_path), frame):
        print(f"Failed to write empty frame image: {empty_frame_path}")
        return

    config = {
        "version": 1,
        "rows": rows,
        "cols": cols,
        "slot_labels": slot_labels,
        "empty_frame": str(empty_frame_path),
        "detection": {
            "diff_threshold": 25,
            "change_ratio_threshold": 0.12,
            "edge_ratio_threshold": 0.035,
            "object_area_ratio_threshold": 0.08,
        },
        "slots": [
            {
                "id": label,
                "polygon": [[int(x), int(y)] for x, y in ui.slots[label]],
            }
            for label in slot_labels
        ],
    }

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)

    print(f"Saved calibration config to: {output_path}")
    print(f"Saved empty reference frame to: {empty_frame_path}")


if __name__ == "__main__":
    main()
