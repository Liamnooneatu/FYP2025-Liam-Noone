"""
preprocess_ocr.py
-----------------
OCR for glowing white/cyan digits AND letters on coloured backgrounds.

CONTROLS:
  P  — toggle preprocessing Preview   (capturing pauses)
  F  — draw polygon Focus ROI         (capturing pauses while drawing)
  S  — toggle CSV saving on/off       (saves to advancer.csv in same folder)
  X  — clear ROI, back to full frame
  Q  — quit

  In draw mode:
    Click     add corner point
    Enter     confirm & close polygon
    Z         undo last point
    C         clear all points
    Esc       cancel, keep previous ROI

CSV (advancer.csv):
  Created next to this script when S is first pressed.
  Columns: timestamp, ocr_result, ocr_mode, roi
  Each new OCR result appends one row. Duplicates of the previous result
  are NOT saved (so a static reading doesn't fill the file with repeats).
  Press S again to pause saving without losing the file.

P-KEY FIX:
  Key input is now always processed every loop tick regardless of the
  frame-rate throttle, so P responds on the very first press.

POLYGON FOCUS AREA:
  Press F — camera freezes, OCR pauses.
  Click to place corner points (any irregular shape).
  A live amber line follows the cursor; a green fill previews the area.
  Click near the first point (cyan snap ring) OR press Enter to confirm.
  Z=undo last point  C=clear all  Esc=cancel

SETTINGS:
  All trackbar changes apply immediately, even in Preview mode.
"""

import cv2
import numpy as np
import easyocr
import torch
import time
import threading
import math
import csv
import os

CAPTURE_INTERVAL  = 5
PREVIEW_FPS_LIMIT = 15

CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "advancer.csv")
CSV_HEADER = ["timestamp", "ocr_result", "ocr_mode", "roi"]

# EasyOCR 
print("Loading EasyOCR (background)...")
reader      = None
reader_lock = threading.Lock()

def _load_reader():
    global reader
    r = easyocr.Reader(['en'], gpu=torch.cuda.is_available(), verbose=False)
    with reader_lock:
        reader = r
    print("EasyOCR ready.\n")

threading.Thread(target=_load_reader, daemon=True).start()

# Settings 
settings = {
    "show_preview": False,
    "sat_remove":   1,
    "max_sat":      50,
    "min_val":      20,
    "upscale":      1,
    "channel":      3,
    "clahe_clip":   10,
    "auto_thresh":  1,
    "thresh_val":   140,
    "invert":       0,
    "morph_close":  3,
    "morph_open":   2,
    "ocr_alpha":    1,
}

# CSV / saving state 
csv_saving    = False      # toggled by S
last_saved    = None       # last result written — avoids duplicate rows
csv_row_count = 0          # rows written this session

def _ensure_csv():
    """Create the CSV with a header row if it does not exist yet."""
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(CSV_HEADER)
        print(f"Created CSV: {CSV_PATH}")

def save_to_csv(result, ocr_mode, roi_label):
    """Append one row. Returns True if written, False if skipped (duplicate)."""
    global last_saved, csv_row_count
    if result in ("", "[no text detected]", "[EasyOCR still loading]",
                  "waiting for EasyOCR...", None):
        return False
    if result == last_saved:
        return False            # same reading as last save — skip
    _ensure_csv()
    row = [time.strftime("%Y-%m-%d %H:%M:%S"), result, ocr_mode, roi_label]
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)
    last_saved    = result
    csv_row_count += 1
    print(f"[CSV] row {csv_row_count} saved: {result}")
    return True

#Polygon ROI state 
roi_poly   = None
draw_mode  = False
poly_pts   = []
mouse_pos  = (0, 0)
draw_frame = None

SNAP_RADIUS = 15
MIN_POINTS  = 3
PT_COLOUR   = (0, 220, 255)
PREVIEW_COL = (255, 180, 0)
FILL_ALPHA  = 0.18

WIN  = "OCR Feed  |  P=preview  F=focus  S=save-CSV  X=clear-ROI  Q=quit"
SWIN = "Settings"


# Mouse callback 
def mouse_cb(event, x, y, flags, param):
    global poly_pts, mouse_pos, roi_poly, draw_mode
    if not draw_mode:
        return
    mouse_pos = (x, y)
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(poly_pts) >= MIN_POINTS:
            fx, fy = poly_pts[0]
            if math.hypot(x - fx, y - fy) <= SNAP_RADIUS:
                _confirm_polygon(); return
        poly_pts.append((x, y))


def _confirm_polygon():
    global roi_poly, draw_mode, poly_pts
    if len(poly_pts) >= MIN_POINTS:
        roi_poly  = poly_pts.copy()
        draw_mode = False
        poly_pts  = []
        print(f"Polygon ROI confirmed: {len(roi_poly)} points")
    else:
        print(f"Need at least {MIN_POINTS} points.")


def enter_draw_mode(frame):
    global draw_mode, poly_pts, mouse_pos, draw_frame
    draw_mode  = True
    poly_pts   = []
    mouse_pos  = (0, 0)
    draw_frame = frame.copy()
    print("\n── POLYGON DRAW MODE ────────────────────────────────────────")
    print("  Click to place corner points (any shape).")
    print("  Click near 1st point (snap ring) OR Enter → confirm")
    print("  Z=undo   C=clear all   Esc=cancel")
    print("─────────────────────────────────────────────────────────────\n")


# Polygon helpers 
def poly_mask(shape, pts):
    mask = np.zeros(shape[:2], dtype=np.uint8)
    if pts and len(pts) >= MIN_POINTS:
        cv2.fillPoly(mask, [np.array(pts, dtype=np.int32)], 255)
    return mask


def crop_poly_bbox(frame, pts):
    if not pts:
        return frame
    arr  = np.array(pts, dtype=np.int32)
    x, y, bw, bh = cv2.boundingRect(arr)
    h, w = frame.shape[:2]
    x  = max(0, x);  y  = max(0, y)
    x2 = min(w, x+bw);  y2 = min(h, y+bh)
    crop = frame[y:y2, x:x2].copy()
    local_pts  = arr - np.array([x, y])
    local_mask = np.zeros(crop.shape[:2], dtype=np.uint8)
    cv2.fillPoly(local_mask, [local_pts], 255)
    crop[local_mask == 0] = 0
    return crop


def roi_label_str():
    return f"polygon {len(roi_poly)}pt" if roi_poly else "full frame"


#Draw-mode overlay 
def render_draw_overlay(base, pts, cursor, snap_radius):
    display = base.copy()
    h, w    = display.shape[:2]

    if len(pts) >= MIN_POINTS:
        preview_pts = pts + [cursor]
        mask    = poly_mask(display.shape, preview_pts)
        outside = cv2.bitwise_not(mask)
        overlay = display.copy()
        overlay[outside == 255] = (overlay[outside == 255] * 0.35).astype(np.uint8)
        display = overlay
        overlay2 = display.copy()
        cv2.fillPoly(overlay2, [np.array(preview_pts, dtype=np.int32)], (0, 180, 80))
        cv2.addWeighted(overlay2, FILL_ALPHA, display, 1-FILL_ALPHA, 0, display)

    for i in range(1, len(pts)):
        cv2.line(display, pts[i-1], pts[i], PT_COLOUR, 2, cv2.LINE_AA)

    if pts:
        cv2.line(display, pts[-1], cursor, PREVIEW_COL, 1, cv2.LINE_AA)
        if len(pts) >= MIN_POINTS:
            cv2.line(display, cursor, pts[0], PREVIEW_COL, 1, cv2.LINE_AA)

    for i, pt in enumerate(pts):
        is_first = (i == 0)
        near     = is_first and len(pts) >= MIN_POINTS and \
                   math.hypot(cursor[0]-pt[0], cursor[1]-pt[1]) <= snap_radius
        r        = 9 if is_first else 5
        col      = (0, 255, 100) if near else PT_COLOUR
        cv2.circle(display, pt, r, col, -1)
        cv2.circle(display, pt, r, (255,255,255), 1)
        if is_first:
            cv2.circle(display, pt, snap_radius, (0,255,100) if near else (100,100,100), 1)

    cv2.line(display, (cursor[0]-10, cursor[1]), (cursor[0]+10, cursor[1]), PREVIEW_COL, 1)
    cv2.line(display, (cursor[0], cursor[1]-10), (cursor[0], cursor[1]+10), PREVIEW_COL, 1)

    need = f"need {MIN_POINTS-len(pts)} more" if len(pts) < MIN_POINTS else "click START or Enter to close"
    pt_txt = f"{len(pts)} pts  —  {need}"
    cv2.rectangle(display,(0,0),(w,30),(15,15,15),-1)
    cv2.putText(display, pt_txt,(6,21),cv2.FONT_HERSHEY_SIMPLEX,0.46,PREVIEW_COL,1,cv2.LINE_AA)

    hint = "DRAW MODE  |  Click=add pt   Enter=confirm   Z=undo   C=clear   Esc=cancel"
    cv2.rectangle(display,(0,h-28),(w,h),(15,15,15),-1)
    cv2.putText(display, hint,(6,h-10),cv2.FONT_HERSHEY_SIMPLEX,0.40,(200,200,200),1,cv2.LINE_AA)
    return display


def render_roi_overlay(display, pts, dim=True):
    if not pts or len(pts) < MIN_POINTS:
        return
    h, w = display.shape[:2]
    arr  = np.array(pts, dtype=np.int32)
    if dim:
        mask    = poly_mask(display.shape, pts)
        outside = cv2.bitwise_not(mask)
        overlay = display.copy()
        overlay[outside == 255] = (overlay[outside == 255] * 0.38).astype(np.uint8)
        display[:] = overlay
    cv2.polylines(display,[arr],isClosed=True,color=PT_COLOUR,thickness=2,lineType=cv2.LINE_AA)
    for pt in pts:
        cv2.circle(display, pt, 4, PT_COLOUR, -1)
    x, y, bw, bh = cv2.boundingRect(arr)
    label = f"FOCUS  {bw}x{bh}px  ({len(pts)} pts)  |  F=redraw  X=clear"
    (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    lx = x+4; ly = y-6 if y>22 else y+bh+lh+6
    cv2.rectangle(display,(lx-2,ly-lh-2),(lx+lw+2,ly+2),(0,0,0),-1)
    cv2.putText(display, label,(lx,ly),cv2.FONT_HERSHEY_SIMPLEX,0.42,PT_COLOUR,1,cv2.LINE_AA)


# Settings window 
def create_settings_window():
    cv2.namedWindow(SWIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(SWIN, 460, 600)
    def _n(_): pass
    cv2.createTrackbar("-- BG REMOVAL --",        SWIN, 0,                         0,   _n)
    cv2.createTrackbar("Sat Remove 0=off 1=on",   SWIN, settings["sat_remove"],    1,   _n)
    cv2.createTrackbar("Max Saturation  (0-255)",  SWIN, settings["max_sat"],       255, _n)
    cv2.createTrackbar("Min Value  (0-255)",       SWIN, settings["min_val"],       255, _n)
    cv2.createTrackbar("-- PIPELINE --",           SWIN, 0,                         0,   _n)
    cv2.createTrackbar("Upscale  1-4",             SWIN, settings["upscale"],       4,   _n)
    cv2.createTrackbar("Channel 0R 1G 2B 3Gray",   SWIN, settings["channel"],       3,   _n)
    cv2.createTrackbar("CLAHE clip  x0.1",         SWIN, settings["clahe_clip"],    100, _n)
    cv2.createTrackbar("Auto-thresh Otsu 0=off",   SWIN, settings["auto_thresh"],   1,   _n)
    cv2.createTrackbar("Threshold  (manual)",      SWIN, settings["thresh_val"],    255, _n)
    cv2.createTrackbar("Invert  0=off 1=on",       SWIN, settings["invert"],        1,   _n)
    cv2.createTrackbar("Morph close",              SWIN, settings["morph_close"],   10,  _n)
    cv2.createTrackbar("Morph open",               SWIN, settings["morph_open"],    10,  _n)
    cv2.createTrackbar("-- OCR MODE --",           SWIN, 0,                         0,   _n)
    cv2.createTrackbar("AlphaNum 0=nums 1=full",   SWIN, settings["ocr_alpha"],     1,   _n)
    print("Settings window open. F=polygon ROI  S=toggle CSV saving\n")


def read_trackbars():
    def get(n): return cv2.getTrackbarPos(n, SWIN)
    settings["sat_remove"]  = get("Sat Remove 0=off 1=on")
    settings["max_sat"]     = get("Max Saturation  (0-255)")
    settings["min_val"]     = get("Min Value  (0-255)")
    settings["upscale"]     = max(1, get("Upscale  1-4"))
    settings["channel"]     = get("Channel 0R 1G 2B 3Gray")
    settings["clahe_clip"]  = get("CLAHE clip  x0.1")
    settings["auto_thresh"] = get("Auto-thresh Otsu 0=off")
    settings["thresh_val"]  = get("Threshold  (manual)")
    settings["invert"]      = get("Invert  0=off 1=on")
    settings["morph_close"] = max(1, get("Morph close"))
    settings["morph_open"]  = max(1, get("Morph open"))
    settings["ocr_alpha"]   = get("AlphaNum 0=nums 1=full")


# Preprocessing 
def preprocess(image_bgr):
    s   = settings
    img = (cv2.resize(image_bgr, None, fx=s["upscale"], fy=s["upscale"],
                      interpolation=cv2.INTER_CUBIC)
           if s["upscale"] > 1 else image_bgr.copy())
    if s["sat_remove"]:
        hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        keep = ((hsv[:,:,1] <= s["max_sat"]) & (hsv[:,:,2] >= s["min_val"])).astype(np.uint8)*255
        img  = cv2.bitwise_and(img, img, mask=keep)
    ch   = s["channel"]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if ch==3 else cv2.split(img)[[2,1,0][ch]]
    clip = s["clahe_clip"] / 10.0
    if clip > 0:
        gray = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8,8)).apply(gray)
    flag = cv2.THRESH_BINARY + (cv2.THRESH_OTSU if s["auto_thresh"] else 0)
    tv   = 0 if s["auto_thresh"] else s["thresh_val"]
    _, thr = cv2.threshold(gray, tv, 255, flag)
    k   = cv2.getStructuringElement(cv2.MORPH_RECT, (s["morph_close"],)*2)
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, k)
    k2  = cv2.getStructuringElement(cv2.MORPH_RECT, (s["morph_open"],)*2)
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, k2)
    if s["invert"]:
        thr = cv2.bitwise_not(thr)
    return cv2.cvtColor(thr, cv2.COLOR_GRAY2BGR)


# OCR thread 
ocr_running = False
last_result = "waiting for EasyOCR..."

def _ocr_thread(frame_snapshot, use_alpha, poly, do_save):
    global ocr_running, last_result
    try:
        crop = crop_poly_bbox(frame_snapshot, poly) if poly else frame_snapshot
        proc = preprocess(crop)
        with reader_lock:
            r = reader
        if r is None:
            last_result = "[EasyOCR still loading]"; return

        if use_alpha:
            results = r.readtext(proc, detail=0, paragraph=False)
            text    = ' '.join(results).strip()
            if not text:
                results = r.readtext(proc,
                    allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.-_ ',
                    detail=0, paragraph=False)
                text = ' '.join(results).strip()
        else:
            results = r.readtext(proc, allowlist='0123456789.-', detail=0, paragraph=False)
            text    = ' '.join(results).strip()
            if not text:
                results = r.readtext(proc, detail=0, paragraph=False)
                text    = ' '.join(results).strip()

        last_result = text if text else "[no text detected]"
        mode_lbl    = "alpha+num" if use_alpha else "num only"
        roi_lbl     = roi_label_str()
        print(f"[{time.strftime('%H:%M:%S')}]  OCR ({mode_lbl}, {roi_lbl}): {last_result}")

        # Save to CSV if enabled
        if do_save:
            save_to_csv(last_result, mode_lbl, roi_lbl)

    except Exception as e:
        last_result = f"[error: {e}]"
        print(f"OCR error: {e}")
    finally:
        ocr_running = False


def trigger_ocr(frame):
    global ocr_running
    if ocr_running:
        return
    ocr_running = True
    threading.Thread(
        target=_ocr_thread,
        args=(frame.copy(), bool(settings["ocr_alpha"]), roi_poly, csv_saving),
        daemon=True
    ).start()


# Camera thread 
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
print("  P   toggle Preview            F   draw polygon ROI")
print("  S   toggle CSV saving         X   clear ROI (full frame)")
print("  Q   quit")
print("  In draw mode:  click=add pt   Enter=confirm   Z=undo   C=clear   Esc=cancel")
print(f"  CSV path: {CSV_PATH}")
print("───────────────────────────────────────────────────────────────\n")







# MAIN LOOP
while True:
    now = time.time()

    # ── Key input — ALWAYS checked every tick (fixes dropped P presses) ───────
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    elif key == ord('p'):
        settings["show_preview"] = not settings["show_preview"]
        if not settings["show_preview"]:
            last_capture = time.time()
        print(f"Preview: {'ON' if settings['show_preview'] else 'OFF'}")

    elif key == ord('f') and not draw_mode:
        with cam_lock:
            snap = cam_frame.copy() if cam_frame is not None else None
        if snap is not None:
            enter_draw_mode(snap)

    elif key == ord('s'):
        csv_saving = not csv_saving
        if csv_saving:
            _ensure_csv()
            print(f"CSV saving ON  →  {CSV_PATH}")
        else:
            print(f"CSV saving OFF  ({csv_row_count} rows written this session)")

    elif key == ord('x'):
        roi_poly = None
        print("ROI cleared — full frame OCR.")

    # Draw-mode specific keys
    elif draw_mode:
        if key == 13:                   # Enter
            _confirm_polygon()
            if not draw_mode:
                last_capture = time.time()
        elif key == ord('z'):
            if poly_pts:
                poly_pts.pop()
                print(f"Undo — {len(poly_pts)} points remaining.")
        elif key == ord('c'):
            poly_pts = []
            print("All points cleared.")
        elif key == 27:                 # Esc
            draw_mode    = False
            poly_pts     = []
            last_capture = time.time()
            print("Draw cancelled." + (" Previous ROI kept." if roi_poly else ""))

    show_preview = settings["show_preview"]

    # Frame-rate throttle — RENDER only, keys already handled above 
    if (show_preview or draw_mode) and (now - prev_time) < 1.0 / PREVIEW_FPS_LIMIT:
        continue
    prev_time = now

    read_trackbars()

    with cam_lock:
        frame = cam_frame.copy() if cam_frame is not None else None
    if frame is None:
        continue
    h, w = frame.shape[:2]






    # DRAW MODE — render frozen frame + overlay   
    if draw_mode:
        display = render_draw_overlay(draw_frame, poly_pts, mouse_pos, SNAP_RADIUS)
        cv2.imshow(WIN, display)
        continue




   
    # NORMAL / PREVIEW MODE    
    if not show_preview and (now - last_capture) >= CAPTURE_INTERVAL:
        last_capture = now
        trigger_ocr(frame)

    if show_preview:
        crop = crop_poly_bbox(frame, roi_poly) if roi_poly else frame
        proc = preprocess(crop)
        if roi_poly:
            arr = np.array(roi_poly, dtype=np.int32)
            x, y, bw, bh = cv2.boundingRect(arr)
            x  = max(0,x); y  = max(0,y)
            x2 = min(w,x+bw); y2 = min(h,y+bh)
            rh, rw = y2-y, x2-x
            display = np.zeros_like(frame)
            proc_rs = cv2.resize(proc, (rw, rh), interpolation=cv2.INTER_NEAREST)
            display[y:y2, x:x2] = proc_rs
            render_roi_overlay(display, roi_poly, dim=False)
        else:
            display = proc
            cv2.rectangle(display,(0,0),(w-1,h-1),(0,210,80),3)
    else:
        display = frame.copy()
        if roi_poly:
            render_roi_overlay(display, roi_poly)

    h_d, w_d = display.shape[:2]

    # Status bar 
    next_in  = max(0.0, CAPTURE_INTERVAL - (now - last_capture))
    ocr_lbl  = "alpha+num" if settings["ocr_alpha"] else "num only"
    roi_lbl  = roi_label_str()
    csv_lbl  = f"REC {csv_row_count}rows" if csv_saving else "rec=OFF"

    if show_preview:
        ch_lbl  = ["R","G","B","Gray"][settings["channel"]]
        thr_lbl = "Otsu" if settings["auto_thresh"] else str(settings["thresh_val"])
        sat_lbl = f"sat<{settings['max_sat']}" if settings["sat_remove"] else "sat=OFF"
        bar_txt = (f"PREVIEW  [{ch_lbl} thr={thr_lbl} {sat_lbl} x{settings['upscale']}"
                   f" ocr={ocr_lbl} {roi_lbl} csv={csv_lbl}]  capturing PAUSED  [settings LIVE]")
        bar_col = (0, 110, 40)
    else:
        bar_txt = (f"LIVE  OCR in {next_in:.1f}s  [{ocr_lbl} {roi_lbl} csv={csv_lbl}]"
                   f"  last: {last_result}")
        bar_col = (30, 30, 30)

    cv2.rectangle(display,(0,h_d-34),(w_d,h_d),bar_col,-1)
    cv2.putText(display, bar_txt,(8,h_d-10),cv2.FONT_HERSHEY_SIMPLEX,0.36,(255,255,255),1,cv2.LINE_AA)

    # Badges 
    def badge(x1, y1, x2, y2, col, txt):
        cv2.rectangle(display,(x1,y1),(x2,y2),col,-1)
        cv2.rectangle(display,(x1,y1),(x2,y2),(160,160,160),1)
        cv2.putText(display,txt,(x1+7,(y1+y2)//2+5),
                    cv2.FONT_HERSHEY_SIMPLEX,0.46,(255,255,255),1,cv2.LINE_AA)

    bc = (0,180,65) if show_preview else (50,50,50)
    badge(8,   8, 168, 42, bc,                   "PREVIEW ON"  if show_preview else "PREVIEW OFF")
    badge(176, 8, 316, 42, (150,70,0) if settings["ocr_alpha"] else (0,80,180),
                                                  f"OCR {ocr_lbl}")
    badge(324, 8, 460, 42, (0,140,200) if roi_poly else (55,55,55),
                                                  f"ROI {roi_lbl}")

    # CSV badge — red pulse when saving, grey when off
    csv_col = (0, 60, 200) if csv_saving else (55, 55, 55)
    csv_txt = f"● REC {csv_row_count}" if csv_saving else "CSV off"
    badge(468, 8, 590, 42, csv_col, csv_txt)

    # Big OCR result
    if not show_preview and last_result not in (
            "waiting for EasyOCR...", "[no text detected]", "[EasyOCR still loading]"):
        cv2.putText(display, last_result,(10,h_d-55),
                    cv2.FONT_HERSHEY_SIMPLEX,1.4,(0,255,128),3,cv2.LINE_AA)

    cv2.imshow(WIN, display)


cam_active = False
cv2.destroyAllWindows()
print(f"\nSession ended. {csv_row_count} rows written to {CSV_PATH}" if csv_row_count else "\nSession ended.")
