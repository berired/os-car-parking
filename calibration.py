"""
calibration.py  –  Parking Space Calibration Tool
===================================================
Usage:
    python calibration.py                  # webcam (index 0)
    python calibration.py carPark.mp4      # video file (uses first frame)
    python calibration.py carParkImg.png   # static image

Controls (in the OpenCV window):
    L-CLICK          → place a zone at cursor using current default size
    L-DRAG           → draw a custom-size zone
    R-CLICK          → delete the zone under cursor
    S                → save calibration to parking_config.pkl
    C                → clear ALL zones
    U                → undo last placed zone
    Q / ESC          → quit (unsaved changes are lost)

Trackbars:
    Sensitivity      → pixel-count threshold (above = occupied)
    Default W / H    → size used for single-click placement
    Rows             → number of row labels  (a, b, c …)
    Cols             → number of column labels (1, 2, 3 …)

Zones are auto-named in row-major order (a1, a2 … b1, b2 …).
The first unplaced name is shown as "Next →" in the HUD.
"""

import cv2
import pickle
import numpy as np
import sys
import os

# ─────────────────────────────────────────────────────────────────────────────
CONFIG_FILE   = "parking_config.pkl"
WIN_NAME      = "Parking Space Calibration"

# Overlay panel geometry
PANEL_W       = 240   # right-side info strip (pixels)
HUD_H         = 110   # bottom control-hint strip (pixels)

# Colours
C_FREE        = (0,   255,  0  )
C_OCC         = (0,   0,   255 )
C_ZONE        = (255, 0,   255 )
C_DRAG        = (0,   255, 255 )
C_HOVER       = (0,   200, 255 )
C_TEXT_YEL    = (0,   255, 255 )
C_TEXT_WHT    = (220, 220, 220 )
C_PANEL_BG    = (25,  25,  25  )
C_ACCENT      = (100, 180, 255 )


# ─────────────────────────────────────────────────────────────────────────────
def load_config() -> dict:
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"[WARN] Could not load config: {e}")
    return dict(spaces={}, sensitivity=900, rows=2, cols=3,
                default_w=107, default_h=48)


def save_config(cfg: dict):
    with open(CONFIG_FILE, "wb") as f:
        pickle.dump(cfg, f)
    print(f"[SAVED] {len(cfg['spaces'])} spaces → {CONFIG_FILE}")


def gen_names(rows: int, cols: int) -> list[str]:
    """Generate ['a1','a2','b1','b2', …] for the given grid dimensions."""
    return [
        f"{chr(ord('a') + r)}{c}"
        for r in range(rows)
        for c in range(1, cols + 1)
    ]


# ─────────────────────────────────────────────────────────────────────────────
class CalibrationApp:
    def __init__(self, source):
        cfg = load_config()
        self.spaces      : dict[str, tuple] = cfg.get("spaces", {})
        self.sensitivity : int  = cfg.get("sensitivity", 900)
        self.rows        : int  = cfg.get("rows", 2)
        self.cols        : int  = cfg.get("cols", 3)
        self.def_w       : int  = cfg.get("default_w", 107)
        self.def_h       : int  = cfg.get("default_h", 48)

        self.names       : list[str] = gen_names(self.rows, self.cols)
        self.history     : list      = []   # undo stack (list of name strings)

        # Mouse state
        self.mx = self.my = 0
        self.drawing    = False
        self.drag_start = None
        self.drag_end   = None

        # Source
        self.cap        = None
        self.static     = None
        self._open_source(source)
        self._read_first_frame()

        # Window
        cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WIN_NAME, self._mouse_cb)
        self._create_trackbars()

    # ── source helpers ────────────────────────────────────────────────────────
    def _open_source(self, source):
        try:
            idx = int(source)
            self.cap = cv2.VideoCapture(idx)
            return
        except (ValueError, TypeError):
            pass
        img = cv2.imread(str(source))
        if img is not None:
            self.static = img
            return
        self.cap = cv2.VideoCapture(str(source))

    def _read_first_frame(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.static = frame.copy()   # freeze first frame for overlay

    def _get_frame(self) -> np.ndarray | None:
        if self.cap and self.cap.isOpened():
            ret, f = self.cap.read()
            if not ret:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, f = self.cap.read()
            return f if ret else self.static
        return self.static.copy() if self.static is not None else None

    def _create_trackbars(self):
        cv2.createTrackbar("Sensitivity",  WIN_NAME, self.sensitivity, 5000, self._tb_sens)
        cv2.createTrackbar("Default W",    WIN_NAME, self.def_w,       400,  self._tb_dw)
        cv2.createTrackbar("Default H",    WIN_NAME, self.def_h,       300,  self._tb_dh)
        cv2.createTrackbar("Rows",         WIN_NAME, self.rows,        15,   self._tb_rows)
        cv2.createTrackbar("Cols",         WIN_NAME, self.cols,        20,   self._tb_cols)

    def _tb_sens(self, v): self.sensitivity = max(1, v)
    def _tb_dw  (self, v): self.def_w       = max(10, v)
    def _tb_dh  (self, v): self.def_h       = max(10, v)

    def _tb_rows(self, v):
        self.rows  = max(1, v)
        self.names = gen_names(self.rows, self.cols)

    def _tb_cols(self, v):
        self.cols  = max(1, v)
        self.names = gen_names(self.rows, self.cols)

    def _mouse_cb(self, event, x, y, flags, param):
        self.mx, self.my = x, y

        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing    = True
            self.drag_start = (x, y)
            self.drag_end   = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.drag_end = (x, y)

        elif event == cv2.EVENT_LBUTTONUP and self.drawing:
            self.drawing = False
            if self.drag_start is None:
                return
            x0, y0 = self.drag_start
            x1, y1 = x, y
            dx, dy = abs(x1 - x0), abs(y1 - y0)

            if dx < 6 and dy < 6:          # treat as single click
                rx, ry = x0, y0
                rw, rh = self.def_w, self.def_h
            else:
                rx = min(x0, x1)
                ry = min(y0, y1)
                rw, rh = max(dx, 10), max(dy, 10)

            used = set(self.spaces.keys())
            for name in self.names:
                if name not in used:
                    self.spaces[name] = (rx, ry, rw, rh)
                    self.history.append(name)
                    break
            else:
                print("[INFO] All grid spaces are already placed.")

            self.drag_start = self.drag_end = None

        elif event == cv2.EVENT_RBUTTONDOWN:
            # Delete zone under cursor
            for name, (sx, sy, sw, sh) in list(self.spaces.items()):
                if sx <= x <= sx + sw and sy <= y <= sy + sh:
                    del self.spaces[name]
                    if name in self.history:
                        self.history.remove(name)
                    break

    # ── drawing helpers ───────────────────────────────────────────────────────
    def _draw_zones(self, frame: np.ndarray):
        for name, (sx, sy, sw, sh) in self.spaces.items():
            hover = sx <= self.mx <= sx + sw and sy <= self.my <= sy + sh
            color = C_HOVER if hover else C_ZONE
            thick = 3 if hover else 2
            cv2.rectangle(frame, (sx, sy), (sx + sw, sy + sh), color, thick)
            cv2.putText(frame, name.upper(),
                        (sx + 3, sy + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_TEXT_YEL, 1, cv2.LINE_AA)

        # Live drag rectangle
        if self.drawing and self.drag_start and self.drag_end:
            cv2.rectangle(frame, self.drag_start, self.drag_end, C_DRAG, 2)

    def _draw_right_panel(self, frame: np.ndarray) -> np.ndarray:
        h = frame.shape[0]
        panel = np.full((h, PANEL_W, 3), C_PANEL_BG, dtype=np.uint8)

        def txt(s, y, color=C_TEXT_WHT, scale=0.45, thick=1):
            cv2.putText(panel, s, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                        scale, color, thick, cv2.LINE_AA)

        txt("CALIBRATION", 24, C_ACCENT, 0.65, 2)
        txt(f"Grid  {self.rows}R x {self.cols}C", 50, C_TEXT_WHT)
        txt(f"Sens  {self.sensitivity}", 70, C_TEXT_WHT)
        txt(f"Size  {self.def_w} x {self.def_h}", 90, C_TEXT_WHT)

        # Divider
        cv2.line(panel, (0, 102), (PANEL_W, 102), (60, 60, 60), 1)

        # Space list
        txt("SPACES", 122, C_ACCENT, 0.5, 1)
        row_y = 145
        for name in self.names:
            if name in self.spaces:
                sx, sy, sw, sh = self.spaces[name]
                label = f"{name.upper()}  ({sx},{sy}) {sw}x{sh}"
                color = (140, 240, 140)
            else:
                label = f"{name.upper()}  --"
                color = (100, 100, 100)
            txt(label, row_y, color)
            row_y += 18
            if row_y > h - 10:
                break

        return np.hstack([frame, panel])

    def _draw_hud(self, canvas: np.ndarray) -> np.ndarray:
        ch, cw = canvas.shape[:2]
        strip  = np.full((HUD_H, cw, 3), C_PANEL_BG, dtype=np.uint8)

        placed   = len(self.spaces)
        total    = len(self.names)
        unplaced = [n for n in self.names if n not in self.spaces]
        nxt      = unplaced[0].upper() if unplaced else "ALL PLACED ✓"

        def row(s, y, color=C_TEXT_WHT):
            cv2.putText(strip, s, (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.48, color, 1, cv2.LINE_AA)

        row(f"Placed: {placed}/{total}   Next → {nxt}", 24, C_ACCENT)
        remaining = ", ".join(n.upper() for n in unplaced[:12])
        if len(unplaced) > 12:
            remaining += " …"
        row(f"Remaining: {remaining}", 46, (160, 160, 160))
        row("[L-CLICK] Place default   [L-DRAG] Draw custom   [R-CLICK] Delete", 70, C_TEXT_WHT)
        row("[S] Save   [C] Clear all   [U] Undo   [Q/ESC] Quit", 92, C_TEXT_WHT)

        return np.vstack([canvas, strip])

    # ── main loop ─────────────────────────────────────────────────────────────
    def run(self):
        print(f"[READY] {len(self.spaces)} spaces loaded. "
              f"Grid: {self.rows}R x {self.cols}C")

        while True:
            frame = self._get_frame()
            if frame is None:
                print("[ERROR] Cannot read frame.")
                break

            self._draw_zones(frame)
            canvas = self._draw_right_panel(frame)
            canvas = self._draw_hud(canvas)
            cv2.imshow(WIN_NAME, canvas)

            key = cv2.waitKey(30) & 0xFF

            if key == ord("s"):
                save_config(dict(
                    spaces=self.spaces,
                    sensitivity=self.sensitivity,
                    rows=self.rows,
                    cols=self.cols,
                    default_w=self.def_w,
                    default_h=self.def_h,
                ))

            elif key == ord("c"):
                self.spaces.clear()
                self.history.clear()
                print("[CLEARED] All spaces removed.")

            elif key == ord("u"):
                if self.history:
                    last = self.history.pop()
                    self.spaces.pop(last, None)
                    print(f"[UNDO] Removed {last.upper()}")

            elif key in (ord("q"), 27):   # Q or ESC
                break

        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    source = sys.argv[1] if len(sys.argv) > 1 else 0
    CalibrationApp(source).run()
