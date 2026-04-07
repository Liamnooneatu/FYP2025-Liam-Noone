"""
preprocess_ocr.py
-----------------
OCR for glowing white/cyan digits AND letters on coloured backgrounds (green, blue, teal, etc.)

HOW BACKGROUND REMOVAL WORKS:
  The key insight: coloured backgrounds have HIGH saturation in HSV.
  Glowing white/cyan text has LOW saturation (the glow washes out the colour).
  By masking out all pixels above a saturation threshold, the background
  disappears completely — leaving only the text, including the decimal dot.

RECOMMENDED STARTING SETTINGS:
  Sat Remove ON  = 1
  Max Saturation = 50   (strips anything more colourful than near-white)
  Min Value      = 20   (keeps only reasonably bright pixels)
  Auto-thresh    = 1    (Otsu finds the best binary threshold automatically)
  Upscale        = 1    (640x480 feed doesn't need upscaling)
  Invert         = 0    (white text on black background)

SETTINGS PERSISTENCE NOTE:
  All trackbar changes take effect IMMEDIATELY, including during Preview mode.
  The preprocessing pipeline always reads the current trackbar values before
  each frame, so what you see in Preview is exactly what OCR will receive.

FOCUS AREA / ROI:
  Press F to enter draw mode. Click and drag a rectangle over the area you
  want to read. Press Enter to confirm, Esc to cancel.
  When a ROI is active:
    - The full frame is shown with a bright cyan border around the focus area
    - The area outside is dimmed so the focus zone stands out clearly
    - Only pixels inside the box are sent to OCR and preprocessing
  Press F again at any time to redraw. Press X to clear the ROI entirely.

CONTROLS:
  P     - toggle preprocessing Preview  (capturing pauses)
  F     - enter Focus/ROI draw mode     (capturing pauses while drawing)
  X     - clear active ROI              (back to full-frame OCR)
  Q     - quit

  Inside ROI draw mode:
    Click+drag  draw rectangle
    Enter       confirm selection
    C           clear and redraw
    Esc         cancel, keep previous ROI
"""

import cv2
import numpy as np
import easyocr
import torch
import time
import threading

CAPTURE_INTERVAL  = 5
PREVIEW_FPS_LIMIT = 10

# ── EasyOCR loads in background ───────────────────────────────────────────────
print("Loading EasyOCR (background)...")
reader      = None
reader_lock = threading.Lock()

def _load_reader():
    global reader
    r = easyocr.Reader(['en'], gpu=torch.cuda.is_available(), verbose=False)
    with reader_lock:
        reader = r
    print("EasyOCR ready — capturing will begin now.\n")

threading.Thread(target=_load_reader, daemon=True).start()

# ── Settings ──────────────────────────────────────────────────────────────────
settings = {
    "show_preview":  False,
    "sat_remove":    1,
    "max_sat":       50,
    "min_val":       20,
    "upscale":       1,
    "channel":       3,
    "clahe_clip":    10,
    "auto_thresh":   1,
    "thresh_val":    140,
    "invert":        0,
    "morph_close":   3,
    "morph_open":    2,
    "ocr_alpha":     1,
}

# ── ROI / Focus state ─────────────────────────────────────────────────────────
roi_rect   = None       # confirmed (x1,y1,x2,y2) or None = full frame
draw_mode  = False      # True while user is drawing
draw_start = None       # mouse-down point
draw_cur   = None       # current drag point
draw_frame = None       # frozen frame shown while drawing

WIN  = "OCR Feed  |  P=preview  F=focus  X=clear-ROI  Q=quit"
SWIN = "Settings"

ROI_COLOUR  = (0, 220, 255)   # cyan
DRAW_COLOUR = (255, 180, 0)   # amber while dragging


# ── Mouse callback ────────────────────────────────────────────────────────────
def mouse_cb(event, x, y, flags, param):
    global draw_start, draw_cur, roi_rect, draw_mode

    if not draw_mode:
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        draw_start = (x, y)
        draw_cur   = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE and draw_start is not None:
        draw_cur = (x, y)

    elif event == cv2.EVENT_LBUTTONUP and draw_start is not None:
        draw_cur = (x, y)
        x1 = min(draw_start[0], draw_cur[0])
        y1 = min(draw_start[1], draw_cur[1])
        x2 = max(draw_start[0], draw_cur[0])
        y2 = max(draw_start[1], draw_cur[1])
        if (x2 - x1) > 10 and (y2 - y1) > 10:
            roi_rect   = (x1, y1, x2, y2)
            draw_mode  = False
            draw_start = None
            draw_cur   = None
            print(f"ROI confirmed: {roi_rect}")


def enter_draw_mode(frame):
    global draw_mode, draw_start, draw_cur, draw_frame
    draw_mode  = True
    draw_start = None
    draw_cur   = None
    draw_frame = frame.copy()
    print("\n── ROI DRAW MODE ─────────────────────────────────────────────")
    print("  Click and drag to draw your focus rectangle.")
    print("  Enter=confirm | C=clear & retry | Esc=cancel")
    print("──────────────────────────────────────────────────────────────\n")


# ── Settings window ───────────────────────────────────────────────────────────
def create_settings_window():
    cv2.namedWindow(SWIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(SWIN, 460, 600)
    def _n(_): pass

    cv2.createTrackbar("-- BG REMOVAL --",        SWIN, 0,                          0,   _n)
    cv2.createTrackbar("Sat Remove 0=off 1=on",   SWIN, settings["sat_remove"],     1,   _n)
    cv2.createTrackbar("Max Saturation  (0-255)",  SWIN, settings["max_sat"],        255, _n)
    cv2.createTrackbar("Min Value  (0-255)",       SWIN, settings["min_val"],        255, _n)

    cv2.createTrackbar("-- PIPELINE --",           SWIN, 0,                          0,   _n)
    cv2.createTrackbar("Upscale  1-4",             SWIN, settings["upscale"],        4,   _n)
    cv2.createTrackbar("Channel 0R 1G 2B 3Gray",   SWIN, settings["channel"],        3,   _n)
    cv2.createTrackbar("CLAHE clip  x0.1",         SWIN, settings["clahe_clip"],     100, _n)
    cv2.createTrackbar("Auto-thresh Otsu 0=off",   SWIN, settings["auto_thresh"],    1,   _n)
    cv2.createTrackbar("Threshold  (manual)",      SWIN, settings["thresh_val"],     255, _n)
    cv2.createTrackbar("Invert  0=off 1=on",       SWIN, settings["invert"],         1,   _n)
    cv2.createTrackbar("Morph close",              SWIN, settings["morph_close"],    10,  _n)
    cv2.createTrackbar("Morph open",               SWIN, settings["morph_open"],     10,  _n)

    cv2.createTrackbar("-- OCR MODE --",           SWIN, 0,                          0,   _n)
    cv2.createTrackbar("AlphaNum 0=nums 1=full",   SWIN, settings["ocr_alpha"],      1,   _n)

    print("─── Settings window open ────────────────────────────────────────")
    print("  Sat Remove=1, Max Saturation=50 strips coloured backgrounds.")
    print("  AlphaNum=1 → letters+numbers | AlphaNum=0 → numbers only")
    print("  ALL trackbar changes apply IMMEDIATELY — even in Preview mode.")
    print("  Press F on the main window to draw a focus ROI.")
    print("─────────────────────────────────────────────────────────────────\n")


def read_trackbars():
    def get(name): return cv2.getTrackbarPos(name, SWIN)
    settings["sat_remove"]   = get("Sat Remove 0=off 1=on")
    settings["max_sat"]      = get("Max Saturation  (0-255)")
    settings["min_val"]      = get("Min Value  (0-255)")
    settings["upscale"]      = max(1, get("Upscale  1-4"))
    settings["channel"]      = get("Channel 0R 1G 2B 3Gray")
    settings["clahe_clip"]   = get("CLAHE clip  x0.1")
    settings["auto_thresh"]  = get("Auto-thresh Otsu 0=off")
    settings["thresh_val"]   = get("Threshold  (manual)")
    settings["invert"]       = get("Invert  0=off 1=on")
    settings["morph_close"]  = max(1, get("Morph close"))
    settings["morph_open"]   = max(1, get("Morph open"))
    settings["ocr_alpha"]    = get("AlphaNum 0=nums 1=full")


# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess(image_bgr):
    s = settings

    if s["upscale"] > 1:
        img = cv2.resize(image_bgr, None,
                         fx=s["upscale"], fy=s["upscale"],
                         interpolation=cv2.INTER_CUBIC)
    else:
        img = image_bgr.copy()

    if s["sat_remove"]:
        hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        s_ch = hsv[:, :, 1]
        v_ch = hsv[:, :, 2]
        keep = ((s_ch <= s["max_sat"]) & (v_ch >= s["min_val"])).astype(np.uint8) * 255
        img  = cv2.bitwise_and(img, img, mask=keep)

    ch = s["channel"]
    if ch == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = cv2.split(img)[[2, 1, 0][ch]]

    clip = s["clahe_clip"] / 10.0
    if clip > 0:
        gray = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8)).apply(gray)

    if s["auto_thresh"]:
        _, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, thr = cv2.threshold(gray, s["thresh_val"], 255, cv2.THRESH_BINARY)

    k   = cv2.getStructuringElement(cv2.MORPH_RECT, (s["morph_close"], s["morph_close"]))
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, k)
    k2  = cv2.getStructuringElement(cv2.MORPH_RECT, (s["morph_open"],  s["morph_open"]))
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, k2)

    if s["invert"]:
        thr = cv2.bitwise_not(thr)

    return cv2.cvtColor(thr, cv2.COLOR_GRAY2BGR)


def crop_to_roi(frame):
    if roi_rect is None:
        return frame
    x1, y1, x2, y2 = roi_rect
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    return frame[y1:y2, x1:x2]


# ── ROI overlay drawing ───────────────────────────────────────────────────────
def draw_roi_overlay(display, rect, colour, dim_outside=True, thickness=2, corner_len=18):
    """Stylised corner-bracket border with optional outside dimming."""
    h, w = display.shape[:2]
    x1, y1, x2, y2 = rect
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)
    cl = corner_len

    if dim_outside:
        overlay = display.copy()
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[y1:y2, x1:x2] = 255
        outside = cv2.bitwise_not(mask)
        overlay[outside == 255] = (overlay[outside == 255] * 0.40).astype(np.uint8)
        display[:] = overlay

    # Thin full border
    cv2.rectangle(display, (x1, y1), (x2, y2), colour, 1)

    # Thick corner brackets
    for cx, cy, sx, sy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
        cv2.line(display, (cx, cy), (cx + sx * cl, cy), colour, thickness + 1)
        cv2.line(display, (cx, cy), (cx, cy + sy * cl), colour, thickness + 1)

    # Label
    label = f"FOCUS  {x2-x1}x{y2-y1}px  (F=redraw  X=clear)"
    (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    lx = x1 + 4
    ly = y1 - 6 if y1 > 22 else y2 + lh + 8
    cv2.rectangle(display, (lx - 2, ly - lh - 2), (lx + lw + 2, ly + 2), (0, 0, 0), -1)
    cv2.putText(display, label, (lx, ly),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, colour, 1, cv2.LINE_AA)


def draw_drag_overlay(display, start, cur):
    """Amber animated-corner rectangle while dragging."""
    if start is None or cur is None:
        return
    x1 = min(start[0], cur[0]);  y1 = min(start[1], cur[1])
    x2 = max(start[0], cur[0]);  y2 = max(start[1], cur[1])

    overlay = display.copy()
    mask = np.zeros(display.shape[:2], dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    outside = cv2.bitwise_not(mask)
    overlay[outside == 255] = (overlay[outside == 255] * 0.4).astype(np.uint8)
    display[:] = overlay

    cv2.rectangle(display, (x1, y1), (x2, y2), DRAW_COLOUR, 1)
    cl = 20
    for cx, cy, dx, dy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
        cv2.line(display, (cx, cy), (cx + dx * cl, cy), DRAW_COLOUR, 3)
        cv2.line(display, (cx, cy), (cx, cy + dy * cl), DRAW_COLOUR, 3)

    sz = f"{abs(x2-x1)}x{abs(y2-y1)}  —  Enter to confirm"
    cv2.putText(display, sz,
                (x1 + 4, y1 - 6 if y1 > 16 else y2 + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, DRAW_COLOUR, 1, cv2.LINE_AA)


# ── Threaded OCR ──────────────────────────────────────────────────────────────
ocr_running = False
last_result = "waiting for EasyOCR..."

def _ocr_thread(frame_snapshot, use_alpha):
    global ocr_running, last_result
    try:
        crop      = crop_to_roi(frame_snapshot)
        processed = preprocess(crop)
        with reader_lock:
            r = reader
        if r is None:
            last_result = "[EasyOCR still loading]"; return

        if use_alpha:
            results = r.readtext(processed, detail=0, paragraph=False)
            text = ' '.join(results).strip()
            if not text:
                results = r.readtext(
                    processed,
                    allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.-_ ',
                    detail=0, paragraph=False)
                text = ' '.join(results).strip()
        else:
            results = r.readtext(processed, allowlist='0123456789.-', detail=0, paragraph=False)
            text = ' '.join(results).strip()
            if not text:
                results = r.readtext(processed, detail=0, paragraph=False)
                text = ' '.join(results).strip()

        last_result = text if text else "[no text detected]"
        mode_lbl    = "alpha+num" if use_alpha else "num only"
        roi_lbl     = (f"ROI {roi_rect[2]-roi_rect[0]}x{roi_rect[3]-roi_rect[1]}px"
                       if roi_rect else "full frame")
        print(f"[{time.strftime('%H:%M:%S')}]  OCR ({mode_lbl}, {roi_lbl}): {last_result}")
    except Exception as e:
        last_result = f"[error: {e}]"
        print(f"OCR error: {e}")
    finally:
        ocr_running = False


def trigger_ocr(frame):
    global ocr_running
    if ocr_running:
        print(f"[{time.strftime('%H:%M:%S')}]  OCR skipped (still processing)"); return
    ocr_running = True
    use_alpha = bool(settings["ocr_alpha"])
    threading.Thread(target=_ocr_thread, args=(frame.copy(), use_alpha), daemon=True).start()


# ── Threaded camera ───────────────────────────────────────────────────────────
cam_frame  = None
cam_lock   = threading.Lock()
cam_active = True

def _cam_thread():
    global cam_frame, cam_active
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open camera."); cam_active = False; return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 15)
    while cam_active:
        ret, f = cap.read()
        if ret:
            with cam_lock:
                cam_frame = f
        else:
            time.sleep(0.01)
    cap.release()

threading.Thread(target=_cam_thread, daemon=True).start()

print("Waiting for camera...")
while cam_frame is None:
    time.sleep(0.05)
print("Camera ready.\n")

cv2.namedWindow(WIN)
cv2.setMouseCallback(WIN, mouse_cb)
create_settings_window()

last_capture = time.time() - CAPTURE_INTERVAL
prev_time    = time.time()

print("─── Controls ──────────────────────────────────────────────────")
print("  'p'  toggle preprocessing Preview  (capturing pauses)")
print("  'f'  draw Focus ROI               (capturing pauses while drawing)")
print("  'x'  clear active ROI             (back to full-frame OCR)")
print("  'q'  quit")
print("  Inside ROI draw mode:")
print("    Click+drag  draw rectangle")
print("    Enter       confirm selection")
print("    C           clear & redraw")
print("    Esc         cancel, keep previous ROI")
print("───────────────────────────────────────────────────────────────\n")


while True:
    now = time.time()

    show_preview = settings["show_preview"]

    # Frame-rate cap when smooth update not needed
    if (show_preview or draw_mode) and (now - prev_time) < 1.0 / PREVIEW_FPS_LIMIT:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue
    prev_time = now

    read_trackbars()

    with cam_lock:
        frame = cam_frame.copy() if cam_frame is not None else None
    if frame is None:
        continue

    h, w = frame.shape[:2]

    # ════════════════════════════════════════════════════════════════════════
    # DRAW MODE — frozen frame with drag overlay
    # ════════════════════════════════════════════════════════════════════════
    if draw_mode:
        display = draw_frame.copy()

        # Dim everything; drag overlay will re-highlight inside rect
        draw_drag_overlay(display, draw_start, draw_cur)

        # If a previous ROI exists and nothing is being dragged yet, show it
        if roi_rect and draw_start is None:
            draw_roi_overlay(display, roi_rect, ROI_COLOUR, dim_outside=False)

        # Instruction banner
        banner = ("DRAW MODE  |  Click+drag to select focus area  "
                  "|  Enter=confirm   C=clear   Esc=cancel")
        cv2.rectangle(display, (0, 0), (w, 32), (15, 15, 15), -1)
        cv2.putText(display, banner, (6, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 200, 0), 1, cv2.LINE_AA)

        cv2.imshow(WIN, display)
        key = cv2.waitKey(1) & 0xFF

        if key == 13:           # Enter — confirm
            if roi_rect is not None:
                draw_mode    = False
                draw_start   = None
                draw_cur     = None
                last_capture = time.time()
                print(f"ROI active: {roi_rect}")
            else:
                print("Nothing drawn yet — click and drag first.")

        elif key == ord('c'):   # Clear and redraw
            roi_rect   = None
            draw_start = None
            draw_cur   = None
            print("ROI cleared — draw a new one.")

        elif key == 27:         # Esc — cancel
            draw_mode    = False
            draw_start   = None
            draw_cur     = None
            last_capture = time.time()
            print("ROI draw cancelled — keeping previous ROI." if roi_rect else "ROI draw cancelled.")

        elif key == ord('q'):
            break
        continue

    # ════════════════════════════════════════════════════════════════════════
    # NORMAL / PREVIEW mode
    # ════════════════════════════════════════════════════════════════════════

    # Auto-capture
    if not show_preview and (now - last_capture) >= CAPTURE_INTERVAL:
        last_capture = now
        trigger_ocr(frame)

    # Build display
    if show_preview:
        crop = crop_to_roi(frame)
        proc = preprocess(crop)

        if roi_rect:
            # Paste processed crop back into a dark full-size canvas
            display = np.zeros_like(frame)
            x1, y1, x2, y2 = roi_rect
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            rh, rw  = y2 - y1, x2 - x1
            proc_rs = cv2.resize(proc, (rw, rh), interpolation=cv2.INTER_NEAREST)
            display[y1:y2, x1:x2] = proc_rs
            draw_roi_overlay(display, roi_rect, ROI_COLOUR, dim_outside=False)
        else:
            display = proc
            cv2.rectangle(display, (0, 0), (w-1, h-1), (0, 210, 80), 3)
    else:
        display = frame.copy()
        if roi_rect:
            draw_roi_overlay(display, roi_rect, ROI_COLOUR)

    h_d, w_d = display.shape[:2]

    # Status bar
    next_in   = max(0.0, CAPTURE_INTERVAL - (now - last_capture))
    ocr_label = "alpha+num" if settings["ocr_alpha"] else "num only"
    roi_label = (f"ROI {roi_rect[2]-roi_rect[0]}x{roi_rect[3]-roi_rect[1]}px"
                 if roi_rect else "full frame")

    if show_preview:
        ch_lbl  = ["R","G","B","Gray"][settings["channel"]]
        thr_lbl = "Otsu" if settings["auto_thresh"] else str(settings["thresh_val"])
        sat_lbl = f"sat<{settings['max_sat']}" if settings["sat_remove"] else "sat=OFF"
        bar_txt = (f"PREVIEW  [{ch_lbl}  thr={thr_lbl}  {sat_lbl}  x{settings['upscale']}"
                   f"  ocr={ocr_label}  {roi_label}]  — capturing PAUSED  [settings LIVE]")
        bar_col = (0, 110, 40)
    else:
        bar_txt = (f"LIVE  |  OCR in {next_in:.1f}s  [{ocr_label}  {roi_label}]"
                   f"  |  last: {last_result}")
        bar_col = (30, 30, 30)

    cv2.rectangle(display, (0, h_d-34), (w_d, h_d), bar_col, -1)
    cv2.putText(display, bar_txt, (8, h_d-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

    # ── Top-left badges ───────────────────────────────────────────────────────
    # Preview badge
    bc = (0,180,65) if show_preview else (50,50,50)
    tc = (0,0,0)    if show_preview else (220,220,220)
    cv2.rectangle(display, (8,8),(172,42), bc, -1)
    cv2.rectangle(display, (8,8),(172,42), (180,180,180), 1)
    cv2.putText(display, "PREVIEW: ON " if show_preview else "PREVIEW: OFF",
                (15,32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, tc, 1, cv2.LINE_AA)

    # OCR mode badge
    badge_col = (160, 70, 0) if settings["ocr_alpha"] else (0, 80, 180)
    cv2.rectangle(display, (180,8),(330,42), badge_col, -1)
    cv2.rectangle(display, (180,8),(330,42), (180,180,180), 1)
    cv2.putText(display, f"OCR: {ocr_label}",
                (187,32), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255,255,255), 1, cv2.LINE_AA)

    # ROI badge
    roi_active    = roi_rect is not None
    roi_badge_col = (0, 150, 210) if roi_active else (55, 55, 55)
    roi_badge_txt = (f"ROI {roi_rect[2]-roi_rect[0]}x{roi_rect[3]-roi_rect[1]}"
                     if roi_active else "ROI: full frame")
    cv2.rectangle(display, (338,8),(510,42), roi_badge_col, -1)
    cv2.rectangle(display, (338,8),(510,42), (180,180,180), 1)
    cv2.putText(display, roi_badge_txt,
                (345,32), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255,255,255), 1, cv2.LINE_AA)

    # Large OCR result on live feed
    if not show_preview and last_result not in (
            "waiting for EasyOCR...", "[no text detected]", "[EasyOCR still loading]"):
        cv2.putText(display, last_result, (10, h_d-55),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 128), 3, cv2.LINE_AA)

    cv2.imshow(WIN, display)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('p'):
        settings["show_preview"] = not settings["show_preview"]
        if not settings["show_preview"]:
            last_capture = time.time()
        print(f"Preview: {'ON' if settings['show_preview'] else 'OFF'}")
    elif key == ord('f'):
        with cam_lock:
            snap = cam_frame.copy() if cam_frame is not None else frame
        enter_draw_mode(snap)
    elif key == ord('x'):
        roi_rect = None
        print("ROI cleared — back to full-frame OCR.")

cam_active = False
cv2.destroyAllWindows()
