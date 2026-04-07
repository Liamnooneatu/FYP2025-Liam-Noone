"""
preprocess_ocr.py
-----------------
OCR for glowing white/cyan digits AND letters on coloured backgrounds.

POLYGON FOCUS AREA (new):
  Press F to enter draw mode.
  - Click to place corner points one at a time (any shape you like)
  - A live preview line follows your cursor between clicks
  - Click near the first point (within 15px) OR press Enter to close & confirm
  - Press Z to undo the last point
  - Press C to clear all points and start again
  - Press Esc to cancel and keep the previous ROI
  When confirmed:
  - The full frame is shown with a cyan outline around your polygon
  - Everything outside is dimmed — only the inside goes to OCR
  - Press F again to redraw, X to clear back to full frame

OTHER CONTROLS:
  P  — toggle preprocessing Preview
  F  — draw polygon Focus ROI        (capturing pauses)
  X  — clear ROI, back to full frame
  Q  — quit

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

CAPTURE_INTERVAL  = 5
PREVIEW_FPS_LIMIT = 15

#asyOCR 
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

# Polygon ROI state 
roi_poly   = None       # confirmed list of (x,y) points, or None
draw_mode  = False
poly_pts   = []         # points placed so far this session
mouse_pos  = (0, 0)     # live cursor for preview line
draw_frame = None       # frozen frame while drawing

SNAP_RADIUS  = 15       # px — how close to first point triggers auto-close
PT_COLOUR    = (0, 220, 255)    # cyan points / outline
LINE_COLOUR  = (0, 220, 255)
PREVIEW_COL  = (255, 180, 0)   # amber ghost line to cursor
FILL_ALPHA   = 0.18
MIN_POINTS   = 3

WIN  = "OCR Feed  |  P=preview  F=focus  X=clear-ROI  Q=quit"
SWIN = "Settings"


#  Mouse callback 
def mouse_cb(event, x, y, flags, param):
    global poly_pts, mouse_pos, roi_poly, draw_mode

    if not draw_mode:
        return

    mouse_pos = (x, y)

    if event == cv2.EVENT_LBUTTONDOWN:
        # Snap-to-close: if near first point and we have enough pts, close polygon
        if len(poly_pts) >= MIN_POINTS:
            fx, fy = poly_pts[0]
            if math.hypot(x - fx, y - fy) <= SNAP_RADIUS:
                _confirm_polygon()
                return
        poly_pts.append((x, y))


def _confirm_polygon():
    global roi_poly, draw_mode, poly_pts, mouse_pos
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
    print("  Click near 1st point (cyan dot) OR Enter  → close & confirm")
    print("  Z = undo last point | C = clear all | Esc = cancel")
    print("─────────────────────────────────────────────────────────────\n")


# olygon mask helpers
def poly_mask(shape, pts):
    """Binary mask with the polygon interior filled white."""
    mask = np.zeros(shape[:2], dtype=np.uint8)
    if pts and len(pts) >= MIN_POINTS:
        cv2.fillPoly(mask, [np.array(pts, dtype=np.int32)], 255)
    return mask


def crop_poly_bbox(frame, pts):
    """Return the bounding-box crop of the polygon region (for OCR)."""
    if not pts:
        return frame
    arr  = np.array(pts, dtype=np.int32)
    x, y, bw, bh = cv2.boundingRect(arr)
    h, w = frame.shape[:2]
    x  = max(0, x);  y  = max(0, y)
    x2 = min(w, x + bw);  y2 = min(h, y + bh)
    crop = frame[y:y2, x:x2].copy()
    # Black-out the part of the bounding box that is OUTSIDE the polygon
    local_pts = arr - np.array([x, y])
    local_mask = np.zeros(crop.shape[:2], dtype=np.uint8)
    cv2.fillPoly(local_mask, [local_pts], 255)
    crop[local_mask == 0] = 0
    return crop


# Draw-mode overlay 
def render_draw_overlay(base, pts, cursor, snap_radius):
    display = base.copy()
    h, w    = display.shape[:2]

    # Dim everything outside the partial polygon
    if len(pts) >= MIN_POINTS:
        # Close with cursor to give a preview of the fill
        preview_pts = pts + [cursor]
        mask    = poly_mask(display.shape, preview_pts)
        outside = cv2.bitwise_not(mask)
        overlay = display.copy()
        overlay[outside == 255] = (overlay[outside == 255] * 0.35).astype(np.uint8)
        display = overlay

    # Filled semi-transparent polygon preview (if >=3 pts)
    if len(pts) >= MIN_POINTS:
        preview_pts = pts + [cursor]
        overlay = display.copy()
        cv2.fillPoly(overlay, [np.array(preview_pts, dtype=np.int32)], (0, 180, 80))
        cv2.addWeighted(overlay, FILL_ALPHA, display, 1 - FILL_ALPHA, 0, display)

    # Draw confirmed edges
    for i in range(1, len(pts)):
        cv2.line(display, pts[i-1], pts[i], LINE_COLOUR, 2, cv2.LINE_AA)

    # Ghost line from last point to cursor
    if pts:
        cv2.line(display, pts[-1], cursor, PREVIEW_COL, 1, cv2.LINE_AA)
        # Ghost line from cursor back to first point (close preview)
        if len(pts) >= MIN_POINTS:
            cv2.line(display, cursor, pts[0], PREVIEW_COL, 1, cv2.LINE_AA)

    # Draw points
    for i, pt in enumerate(pts):
        is_first = (i == 0)
        near     = is_first and len(pts) >= MIN_POINTS and math.hypot(cursor[0]-pt[0], cursor[1]-pt[1]) <= snap_radius
        radius   = 9 if is_first else 5
        colour   = (0, 255, 100) if near else PT_COLOUR
        cv2.circle(display, pt, radius, colour, -1)
        cv2.circle(display, pt, radius, (255,255,255), 1)
        if is_first:
            # Snap target ring
            cv2.circle(display, pt, snap_radius, (0, 255, 100) if near else (100,100,100), 1)

    # Cursor crosshair
    cv2.line(display, (cursor[0]-10, cursor[1]), (cursor[0]+10, cursor[1]), PREVIEW_COL, 1)
    cv2.line(display, (cursor[0], cursor[1]-10), (cursor[0], cursor[1]+10), PREVIEW_COL, 1)

    # Counter
    pt_txt = f"{len(pts)} pts  —  {'click near START or Enter to close' if len(pts)>=MIN_POINTS else f'need {MIN_POINTS-len(pts)} more'}"
    cv2.rectangle(display, (0,0),(w,30),(15,15,15),-1)
    cv2.putText(display, pt_txt, (6,21), cv2.FONT_HERSHEY_SIMPLEX, 0.46, PREVIEW_COL, 1, cv2.LINE_AA)

    # Instruction strip at bottom
    hint = "DRAW MODE  |  Click=add point   Enter=confirm   Z=undo   C=clear   Esc=cancel"
    cv2.rectangle(display,(0,h-28),(w,h),(15,15,15),-1)
    cv2.putText(display, hint, (6,h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200,200,200), 1, cv2.LINE_AA)

    return display


# Confirmed ROI overlay (live feed) 
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

    # Outline
    cv2.polylines(display, [arr], isClosed=True, color=PT_COLOUR, thickness=2, lineType=cv2.LINE_AA)

    # Corner dots
    for pt in pts:
        cv2.circle(display, pt, 4, PT_COLOUR, -1)

    # Bounding-box label
    x, y, bw, bh = cv2.boundingRect(arr)
    label = f"FOCUS  {bw}x{bh}px  ({len(pts)} pts)  |  F=redraw  X=clear"
    (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    lx = x + 4
    ly = y - 6 if y > 22 else y + bh + lh + 6
    cv2.rectangle(display, (lx-2, ly-lh-2), (lx+lw+2, ly+2), (0,0,0), -1)
    cv2.putText(display, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.42, PT_COLOUR, 1, cv2.LINE_AA)


#  Settings window 
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
    print("Settings window open. Press F on the main window to draw a polygon ROI.\n")


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


#Preprocessing 
def preprocess(image_bgr):
    s = settings
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


#  OCR thread 
ocr_running = False
last_result = "waiting for EasyOCR..."

def _ocr_thread(frame_snapshot, use_alpha, poly):
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
        mode_lbl = "alpha+num" if use_alpha else "num only"
        roi_lbl  = f"polygon {len(poly)}pt" if poly else "full frame"
        print(f"[{time.strftime('%H:%M:%S')}]  OCR ({mode_lbl}, {roi_lbl}): {last_result}")
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
        args=(frame.copy(), bool(settings["ocr_alpha"]), roi_poly),
        daemon=True
    ).start()


#  Camera thread
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
print("  X   clear ROI (full frame)    Q   quit")
print("  In draw mode:  click=add pt   Enter=confirm   Z=undo   C=clear   Esc=cancel")
print("───────────────────────────────────────────────────────────────\n")


while True:
    now = time.time()

    show_preview = settings["show_preview"]

    if (show_preview or draw_mode) and (now - prev_time) < 1.0 / PREVIEW_FPS_LIMIT:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        continue
    prev_time = now

    read_trackbars()

    with cam_lock:
        frame = cam_frame.copy() if cam_frame is not None else None
    if frame is None:
        continue
    h, w = frame.shape[:2]

    # ════════════════════════════════════════════════════════════════════════
    # DRAW MODE
    # ════════════════════════════════════════════════════════════════════════
    if draw_mode:
        display = render_draw_overlay(draw_frame, poly_pts, mouse_pos, SNAP_RADIUS)
        cv2.imshow(WIN, display)
        key = cv2.waitKey(1) & 0xFF

        if key == 13:                   # Enter — confirm
            _confirm_polygon()
            if not draw_mode:
                last_capture = time.time()

        elif key == ord('z'):           # Undo last point
            if poly_pts:
                poly_pts.pop()
                print(f"Undo — {len(poly_pts)} points remaining.")

        elif key == ord('c'):           # Clear all
            poly_pts = []
            print("All points cleared.")

        elif key == 27:                 # Esc — cancel
            draw_mode    = False
            poly_pts     = []
            last_capture = time.time()
            print("Draw cancelled." + (" Previous ROI kept." if roi_poly else ""))

        elif key == ord('q'):
            break
        continue

    # ════════════════════════════════════════════════════════════════════════
    # NORMAL / PREVIEW MODE
    # ════════════════════════════════════════════════════════════════════════
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
            x2 = min(w, x+bw); y2 = min(h, y+bh)
            rh, rw = y2-y, x2-x
            display = np.zeros_like(frame)
            proc_rs = cv2.resize(proc, (rw, rh), interpolation=cv2.INTER_NEAREST)
            display[y:y2, x:x2] = proc_rs
            render_roi_overlay(display, roi_poly, dim=False)
        else:
            display = proc
            cv2.rectangle(display, (0,0),(w-1,h-1),(0,210,80),3)
    else:
        display = frame.copy()
        if roi_poly:
            render_roi_overlay(display, roi_poly)

    h_d, w_d = display.shape[:2]

    # Status bar
    next_in   = max(0.0, CAPTURE_INTERVAL - (now - last_capture))
    ocr_lbl   = "alpha+num" if settings["ocr_alpha"] else "num only"
    roi_lbl   = f"polygon {len(roi_poly)}pt" if roi_poly else "full frame"

    if show_preview:
        ch_lbl  = ["R","G","B","Gray"][settings["channel"]]
        thr_lbl = "Otsu" if settings["auto_thresh"] else str(settings["thresh_val"])
        sat_lbl = f"sat<{settings['max_sat']}" if settings["sat_remove"] else "sat=OFF"
        bar_txt = (f"PREVIEW  [{ch_lbl} thr={thr_lbl} {sat_lbl} x{settings['upscale']}"
                   f" ocr={ocr_lbl} {roi_lbl}]  capturing PAUSED  [settings LIVE]")
        bar_col = (0,110,40)
    else:
        bar_txt = f"LIVE  |  OCR in {next_in:.1f}s  [{ocr_lbl}  {roi_lbl}]  |  last: {last_result}"
        bar_col = (30,30,30)

    cv2.rectangle(display,(0,h_d-34),(w_d,h_d),bar_col,-1)
    cv2.putText(display, bar_txt,(8,h_d-10),cv2.FONT_HERSHEY_SIMPLEX,0.37,(255,255,255),1,cv2.LINE_AA)

    # Badges
    def badge(x1,y1,x2,y2,col,txt):
        cv2.rectangle(display,(x1,y1),(x2,y2),col,-1)
        cv2.rectangle(display,(x1,y1),(x2,y2),(170,170,170),1)
        cv2.putText(display,txt,(x1+7,(y1+y2)//2+5),cv2.FONT_HERSHEY_SIMPLEX,0.48,(255,255,255),1,cv2.LINE_AA)

    bc = (0,180,65) if show_preview else (50,50,50)
    badge(8,8,175,42, bc, "PREVIEW ON" if show_preview else "PREVIEW OFF")

    oc = (150,70,0) if settings["ocr_alpha"] else (0,80,180)
    badge(183,8,333,42, oc, f"OCR {ocr_lbl}")

    rc = (0,140,200) if roi_poly else (55,55,55)
    badge(341,8,510,42, rc, f"ROI {roi_lbl}")

    # Big OCR result
    if not show_preview and last_result not in (
            "waiting for EasyOCR...", "[no text detected]", "[EasyOCR still loading]"):
        cv2.putText(display, last_result,(10,h_d-55),
                    cv2.FONT_HERSHEY_SIMPLEX,1.4,(0,255,128),3,cv2.LINE_AA)

    cv2.imshow(WIN, display)
    key = cv2.waitKey(1) & 0xFF

    if   key == ord('q'): break
    elif key == ord('p'):
        settings["show_preview"] = not settings["show_preview"]
        if not settings["show_preview"]: last_capture = time.time()
        print(f"Preview: {'ON' if settings['show_preview'] else 'OFF'}")
    elif key == ord('f'):
        with cam_lock:
            snap = cam_frame.copy() if cam_frame is not None else frame
        enter_draw_mode(snap)
    elif key == ord('x'):
        roi_poly = None
        print("ROI cleared — full frame OCR.")

cam_active = False
cv2.destroyAllWindows()
