# Parking Space Car Detector (OpenCV)

This project detects parked toy cars (Hot Wheels scale diorama) in calibrated parking slots.

## Features

- Interactive `calibration.py` to define parking slots.
- Slot IDs use column letters + row numbers (`a1`, `a2`, ..., `b1`, etc.).
- Editable slot polygons in calibration (draw mode + drag-to-edit mode).
- `detector.py` shows:
  - Number of open slots.
  - Which slot IDs are open.
- Detection is color-robust by using grayscale equalization, motion/shape difference, and edge change instead of color labels.

## Install

```bash
pip install -r requirements.txt
```

## 1) Calibrate Slots

```bash
python calibration.py
```

The script asks for:

- Number of rows.
- Number of columns.
- Webcam feed (`--camera`, default `0`).

Controls:

- `Left Click` in draw mode: add a corner point (4 points per slot).
- `Tab`: toggle `DRAW` / `EDIT` mode.
- `EDIT` mode: drag any corner point to adjust slot shape.
- `U`: undo point for current slot.
- `R`: reset current slot.
- `S`: save calibration.
- `Q`: quit without saving.

Output files:

- `parking_calibration.json`
- `calibration_empty.png`

## 2) Run Detector

```bash
python detector.py --config parking_calibration.json --camera 0
```

Use `--camera` for webcam index (`0`, `1`, ...).

Controls:

- `Q`: quit detector.

## Notes for Accuracy

- Capture the calibration empty frame with stable lighting.
- Keep camera fixed after calibration.
- Ensure each slot polygon tightly matches the slot area.
- If needed, tune thresholds in `parking_calibration.json` under `detection`.
