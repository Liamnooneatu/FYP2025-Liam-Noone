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
    # ── Anti-glare (LCD under light) ──────────────────────────────────────
    # 1. Highlight suppression — kills bright specular hotspots before anything else
    "glare_suppress":   0,       # 0=off 1=on
    "glare_thresh":     220,     # pixels brighter than this are "glare" (0-255)
    "glare_inpaint_r":  7,       # inpaint radius to fill suppressed region (1-15)
    # 2. Polarisation simulation — rolling blur difference removes uniform
    #    reflection layer (works even without a physical polariser)
    "glare_rollingdiff":0,       # 0=off 1=on
    "glare_blur_ksize": 21,      # local-mean window size, must be odd (3-61)
    # 3. Gamma correction — lifts dark segments crushed by glare
    "glare_gamma":      10,      # gamma x0.1  → 10=1.0 (off), <10=brighten, >10=darken
    # 4. Bilateral denoise — smooths glare halos without blurring digit edges
    "glare_bilateral":  0,       # 0=off 1=on
    "glare_bilat_d":    9,       # diameter of pixel neighbourhood (3-15)
    "glare_bilat_sig":  50,      # sigma colour & space (1-150)
}

# CSV / saving state
csv_saving    = False
csv_row_count = 0

def _ensure_csv():
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(CSV_HEADER)
        print(f"Created CSV: {CSV_PATH}")

def save_to_csv(result, ocr_mode, roi_label):
    global csv_row_count
    if result in ("", "[no text detected]", "[EasyOCR still loading]",
                  "waiting for EasyOCR...", None):
        return False
    _ensure_csv()
    row = [time.strftime("%Y-%m-%d %H:%M:%S"), result, ocr_mode, roi_label]
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)
    csv_row_count += 1
    print(f"[CSV] row {csv_row_count} saved ({roi_label}): {result}")
    return True

# ─── Dual-zone state ───────────────────────────────────────────────────────
# Each zone is a dict so we can loop over them cleanly.
ZONES = {
    1: {
        "roi":         None,          # confirmed polygon pts list | None
        "last_result": "waiting...",
        "ocr_running": False,
        "colour":      (0, 220, 255), # cyan
        "label":       "zone1",
    },
    2: {
        "roi":         None,
        "last_result": "waiting...",
        "ocr_running": False,
        "colour":      (255,  80, 255), # magenta
        "label":       "zone2",
    },
}

draw_mode      = False
draw_zone_id   = None        # which zone (1 or 2) we are drawing for
poly_pts       = []
mouse_pos      = (0, 0)
draw_frame     = None

SNAP_RADIUS = 15
MIN_POINTS  = 3
PREVIEW_COL = (255, 180, 0)
FILL_ALPHA  = 0.18

WIN  = "OCR Feed  |  P=preview  1/2=zone ROI  S=save-CSV  X=clear ROIs  Q=quit"
SWIN = "Settings"


# Mouse callback
def mouse_cb(event, x, y, flags, param):
    global poly_pts, mouse_pos, draw_mode
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
    global draw_mode, poly_pts, draw_zone_id
    if len(poly_pts) >= MIN_POINTS:
        ZONES[draw_zone_id]["roi"] = poly_pts.copy()
        print(f"Zone {draw_zone_id} ROI confirmed: {len(poly_pts)} points")
        draw_mode    = False
        poly_pts     = []
        draw_zone_id = None
    else:
        print(f"Need at least {MIN_POINTS} points.")


def enter_draw_mode(frame, zone_id):
    global draw_mode, poly_pts, mouse_pos, draw_frame, draw_zone_id
    draw_mode    = True
    draw_zone_id = zone_id
    poly_pts     = []
    mouse_pos    = (0, 0)
    draw_frame   = frame.copy()
    col_name     = "CYAN" if zone_id == 1 else "MAGENTA"
    print(f"\n── DRAW MODE  (Zone {zone_id} — {col_name}) ─────────────────────────────")
    print("  Click to place corner points.")
    print("  Click near 1st point (snap ring) OR Enter → confirm")
    print("  Z=undo   C=clear all   Esc=cancel")
    print("──────────────────────────────────────────────────────────────\n")


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


def roi_label_str(zone_id):
    z = ZONES[zone_id]
    return f"zone{zone_id} {len(z['roi'])}pt" if z["roi"] else f"zone{zone_id} none"


# Draw-mode overlay
def render_draw_overlay(base, pts, cursor, snap_radius, zone_id):
    display  = base.copy()
    h, w     = display.shape[:2]
    pt_col   = ZONES[zone_id]["colour"]

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
        cv2.line(display, pts[i-1], pts[i], pt_col, 2, cv2.LINE_AA)

    if pts:
        cv2.line(display, pts[-1], cursor, PREVIEW_COL, 1, cv2.LINE_AA)
        if len(pts) >= MIN_POINTS:
            cv2.line(display, cursor, pts[0], PREVIEW_COL, 1, cv2.LINE_AA)

    for i, pt in enumerate(pts):
        is_first = (i == 0)
        near     = is_first and len(pts) >= MIN_POINTS and \
                   math.hypot(cursor[0]-pt[0], cursor[1]-pt[1]) <= snap_radius
        r        = 9 if is_first else 5
        col      = (0, 255, 100) if near else pt_col
        cv2.circle(display, pt, r, col, -1)
        cv2.circle(display, pt, r, (255,255,255), 1)
        if is_first:
            cv2.circle(display, pt, snap_radius, (0,255,100) if near else (100,100,100), 1)

    cv2.line(display, (cursor[0]-10, cursor[1]), (cursor[0]+10, cursor[1]), PREVIEW_COL, 1)
    cv2.line(display, (cursor[0], cursor[1]-10), (cursor[0], cursor[1]+10), PREVIEW_COL, 1)

    need   = f"need {MIN_POINTS-len(pts)} more" if len(pts) < MIN_POINTS else "click START or Enter to close"
    pt_txt = f"ZONE {zone_id}  —  {len(pts)} pts  —  {need}"
    cv2.rectangle(display, (0,0), (w,30), (15,15,15), -1)
    cv2.putText(display, pt_txt, (6,21), cv2.FONT_HERSHEY_SIMPLEX, 0.46, PREVIEW_COL, 1, cv2.LINE_AA)

    hint = f"DRAW MODE (Zone {zone_id})  |  Click=add pt   Enter=confirm   Z=undo   C=clear   Esc=cancel"
    cv2.rectangle(display, (0,h-28), (w,h), (15,15,15), -1)
    cv2.putText(display, hint, (6,h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200,200,200), 1, cv2.LINE_AA)
    return display


def render_roi_overlay(display, zone_id, dim=True):
    z   = ZONES[zone_id]
    pts = z["roi"]
    if not pts or len(pts) < MIN_POINTS:
        return
    h, w  = display.shape[:2]
    arr   = np.array(pts, dtype=np.int32)
    col   = z["colour"]
    if dim:
        mask    = poly_mask(display.shape, pts)
        outside = cv2.bitwise_not(mask)
        overlay = display.copy()
        overlay[outside == 255] = (overlay[outside == 255] * 0.38).astype(np.uint8)
        display[:] = overlay
    cv2.polylines(display, [arr], isClosed=True, color=col, thickness=2, lineType=cv2.LINE_AA)
    for pt in pts:
        cv2.circle(display, pt, 4, col, -1)
    x, y, bw, bh = cv2.boundingRect(arr)
    label = f"ZONE {zone_id}  {bw}x{bh}px  ({len(pts)}pt)"
    (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    lx = x+4; ly = y-6 if y > 22 else y+bh+lh+6
    cv2.rectangle(display, (lx-2,ly-lh-2), (lx+lw+2,ly+2), (0,0,0), -1)
    cv2.putText(display, label, (lx,ly), cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)


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
    cv2.createTrackbar("-- OCR MODE --",            SWIN, 0,                          0,   _n)
    cv2.createTrackbar("AlphaNum 0=nums 1=full",    SWIN, settings["ocr_alpha"],      1,   _n)
    cv2.createTrackbar("-- ANTI-GLARE LCD --",      SWIN, 0,                          0,   _n)
    cv2.createTrackbar("GlareSuppress 0=off 1=on",  SWIN, settings["glare_suppress"], 1,   _n)
    cv2.createTrackbar("GlareThresh (0-255)",        SWIN, settings["glare_thresh"],   255, _n)
    cv2.createTrackbar("GlareInpaint radius 1-15",  SWIN, settings["glare_inpaint_r"],15,  _n)
    cv2.createTrackbar("RollingDiff 0=off 1=on",    SWIN, settings["glare_rollingdiff"],1, _n)
    cv2.createTrackbar("RollingBlur size odd 3-61", SWIN, settings["glare_blur_ksize"],61, _n)
    cv2.createTrackbar("Gamma x0.1  10=neutral",    SWIN, settings["glare_gamma"],    30,  _n)
    cv2.createTrackbar("Bilateral 0=off 1=on",      SWIN, settings["glare_bilateral"], 1,  _n)
    cv2.createTrackbar("BilateralDiam 3-15",        SWIN, settings["glare_bilat_d"],   15, _n)
    cv2.createTrackbar("BilateralSigma 1-150",      SWIN, settings["glare_bilat_sig"], 150,_n)
    print("Settings window open.\n")


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
    settings["morph_close"]        = max(1, get("Morph close"))
    settings["morph_open"]         = max(1, get("Morph open"))
    settings["ocr_alpha"]          = get("AlphaNum 0=nums 1=full")
    # anti-glare
    settings["glare_suppress"]     = get("GlareSuppress 0=off 1=on")
    settings["glare_thresh"]       = get("GlareThresh (0-255)")
    settings["glare_inpaint_r"]    = max(1, get("GlareInpaint radius 1-15"))
    kraw = get("RollingBlur size odd 3-61")
    settings["glare_blur_ksize"]   = kraw if kraw % 2 == 1 else max(3, kraw - 1)
    settings["glare_rollingdiff"]  = get("RollingDiff 0=off 1=on")
    settings["glare_gamma"]        = max(1, get("Gamma x0.1  10=neutral"))
    settings["glare_bilateral"]    = get("Bilateral 0=off 1=on")
    settings["glare_bilat_d"]      = max(3, get("BilateralDiam 3-15"))
    settings["glare_bilat_sig"]    = max(1, get("BilateralSigma 1-150"))


# Preprocessing
def _antialre(img, s):
    """
    Four-stage LCD anti-glare pipeline applied to a BGR image.
    Each stage is independently toggled via settings trackbars.

    Stage 1 — Highlight suppression + inpainting
        Detects pixels above glare_thresh in all 3 channels (true specular
        hotspots are white/near-white).  Fills them with cv2.inpaint so the
        surrounding LCD colour bleeds in, removing the blown-out patch before
        any thresholding sees it.

    Stage 2 — Rolling-difference (reflection layer removal)
        A large Gaussian blur estimates the slow-varying reflection layer.
        Subtracting it and rescaling leaves only the high-frequency digit
        structure, which is what OCR actually needs.  Works like a software
        polariser — no physical filter required.

    Stage 3 — Gamma correction
        Gamma < 1.0 (trackbar < 10) brightens segments that were crushed dark
        by surrounding glare.  Gamma > 1.0 darkens an over-exposed display.
        Applied in float32 for accuracy then converted back.

    Stage 4 — Bilateral filter
        Smooths the glare halo noise while preserving digit edges — unlike a
        plain Gaussian which blurs edges and hurts OCR.
    """
    # Stage 1: highlight suppression + inpaint
    if s["glare_suppress"]:
        gt   = s["glare_thresh"]
        mask = np.all(img > gt, axis=2).astype(np.uint8) * 255
        if mask.any():
            img = cv2.inpaint(img, mask, s["glare_inpaint_r"], cv2.INPAINT_TELEA)

    # Stage 2: rolling difference (reflection layer removal)
    if s["glare_rollingdiff"]:
        k    = s["glare_blur_ksize"]
        blur = cv2.GaussianBlur(img, (k, k), 0)
        diff = cv2.subtract(img, blur)
        img  = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)

    # Stage 3: gamma correction
    gamma = s["glare_gamma"] / 10.0
    if abs(gamma - 1.0) > 0.05:          # skip trivial no-op
        lut   = np.array([((i / 255.0) ** (1.0 / gamma)) * 255
                          for i in range(256)], dtype=np.uint8)
        img   = cv2.LUT(img, lut)

    # Stage 4: bilateral filter
    if s["glare_bilateral"]:
        img = cv2.bilateralFilter(img,
                                  d=s["glare_bilat_d"],
                                  sigmaColor=s["glare_bilat_sig"],
                                  sigmaSpace=s["glare_bilat_sig"])
    return img


def preprocess(image_bgr):
    s   = settings
    img = (cv2.resize(image_bgr, None, fx=s["upscale"], fy=s["upscale"],
                      interpolation=cv2.INTER_CUBIC)
           if s["upscale"] > 1 else image_bgr.copy())

    # ── Anti-glare (runs before everything else so glare doesn't corrupt later stages)
    if s["glare_suppress"] or s["glare_rollingdiff"] or \
       s["glare_bilateral"] or s["glare_gamma"] != 10:
        img = _antialre(img, s)

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


# OCR thread — per zone
def _ocr_thread(frame_snapshot, use_alpha, zone_id, do_save):
    z = ZONES[zone_id]
    try:
        poly = z["roi"]
        crop = crop_poly_bbox(frame_snapshot, poly) if poly else frame_snapshot
        proc = preprocess(crop)
        with reader_lock:
            r = reader
        if r is None:
            z["last_result"] = "[EasyOCR still loading]"; return

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

        z["last_result"] = text if text else "[no text detected]"
        mode_lbl         = "alpha+num" if use_alpha else "num only"
        roi_lbl          = f"zone{zone_id}"
        print(f"[{time.strftime('%H:%M:%S')}]  Zone {zone_id} OCR ({mode_lbl}): {z['last_result']}")

        if do_save:
            save_to_csv(z["last_result"], mode_lbl, roi_lbl)

    except Exception as e:
        z["last_result"] = f"[error: {e}]"
        print(f"Zone {zone_id} OCR error: {e}")
    finally:
        z["ocr_running"] = False


def trigger_ocr(frame, zone_id):
    z = ZONES[zone_id]
    if z["ocr_running"]:
        return
    z["ocr_running"] = True
    threading.Thread(
        target=_ocr_thread,
        args=(frame.copy(), bool(settings["ocr_alpha"]), zone_id, csv_saving),
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
print("  P   toggle Preview")
print("  1   draw/redraw Zone 1 ROI  (cyan)")
print("  2   draw/redraw Zone 2 ROI  (magenta)")
print("  S   toggle CSV saving")
print("  X   clear ALL ROIs (full frame)")
print("  Q   quit")
print("  In draw mode:  click=add pt   Enter=confirm   Z=undo   C=clear   Esc=cancel")
print(f"  CSV path: {CSV_PATH}")
print("───────────────────────────────────────────────────────────────\n")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ═══════════════════════════════════════════════════════════════════════════
while True:
    now = time.time()
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    elif key == ord('p'):
        settings["show_preview"] = not settings["show_preview"]
        if not settings["show_preview"]:
            last_capture = time.time()
        print(f"Preview: {'ON' if settings['show_preview'] else 'OFF'}")

    # Zone draw keys: 1 and 2  (also keep F as alias for zone 1)
    elif key in (ord('1'), ord('f')) and not draw_mode:
        with cam_lock:
            snap = cam_frame.copy() if cam_frame is not None else None
        if snap is not None:
            enter_draw_mode(snap, zone_id=1)

    elif key == ord('2') and not draw_mode:
        with cam_lock:
            snap = cam_frame.copy() if cam_frame is not None else None
        if snap is not None:
            enter_draw_mode(snap, zone_id=2)

    elif key == ord('s'):
        csv_saving = not csv_saving
        if csv_saving:
            _ensure_csv()
            print(f"CSV saving ON  →  {CSV_PATH}")
        else:
            print(f"CSV saving OFF  ({csv_row_count} rows written this session)")

    elif key == ord('x'):
        ZONES[1]["roi"] = None
        ZONES[2]["roi"] = None
        print("All ROIs cleared — full frame OCR.")

    # Draw-mode specific keys
    elif draw_mode:
        if key == 13:           # Enter
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
        elif key == 27:         # Esc
            draw_mode    = False
            poly_pts     = []
            draw_zone_id = None
            last_capture = time.time()
            print("Draw cancelled.")

    show_preview = settings["show_preview"]

    # Frame-rate throttle
    if (show_preview or draw_mode) and (now - prev_time) < 1.0 / PREVIEW_FPS_LIMIT:
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
        display = render_draw_overlay(draw_frame, poly_pts, mouse_pos, SNAP_RADIUS, draw_zone_id)
        cv2.imshow(WIN, display)
        continue

    # ════════════════════════════════════════════════════════════════════════
    # OCR TRIGGER (both zones independently)
    # ════════════════════════════════════════════════════════════════════════
    if not show_preview and (now - last_capture) >= CAPTURE_INTERVAL:
        last_capture = now
        trigger_ocr(frame, zone_id=1)
        trigger_ocr(frame, zone_id=2)

    # ════════════════════════════════════════════════════════════════════════
    # PREVIEW / NORMAL DISPLAY
    # ════════════════════════════════════════════════════════════════════════
    if show_preview:
        # Show preprocessed view for zone 1 if it exists, else full frame
        active_zone = 1 if ZONES[1]["roi"] else (2 if ZONES[2]["roi"] else None)
        if active_zone:
            z   = ZONES[active_zone]
            crop = crop_poly_bbox(frame, z["roi"])
            proc = preprocess(crop)
            arr  = np.array(z["roi"], dtype=np.int32)
            x, y, bw, bh = cv2.boundingRect(arr)
            x  = max(0,x); y  = max(0,y)
            x2 = min(w,x+bw); y2 = min(h,y+bh)
            rh, rw = y2-y, x2-x
            display = np.zeros_like(frame)
            proc_rs = cv2.resize(proc, (rw, rh), interpolation=cv2.INTER_NEAREST)
            display[y:y2, x:x2] = proc_rs
            render_roi_overlay(display, active_zone, dim=False)
        else:
            display = preprocess(frame)
            cv2.rectangle(display, (0,0), (w-1,h-1), (0,210,80), 3)
    else:
        display = frame.copy()
        # Draw both zone overlays
        for zid in (1, 2):
            if ZONES[zid]["roi"]:
                render_roi_overlay(display, zid)

    h_d, w_d = display.shape[:2]

    # ── Status bar ──────────────────────────────────────────────────────────
    next_in  = max(0.0, CAPTURE_INTERVAL - (now - last_capture))
    ocr_lbl  = "alpha+num" if settings["ocr_alpha"] else "num only"
    csv_lbl  = f"REC {csv_row_count}rows" if csv_saving else "rec=OFF"

    z1_res = ZONES[1]["last_result"]
    z2_res = ZONES[2]["last_result"]

    if show_preview:
        ch_lbl  = ["R","G","B","Gray"][settings["channel"]]
        thr_lbl = "Otsu" if settings["auto_thresh"] else str(settings["thresh_val"])
        sat_lbl = f"sat<{settings['max_sat']}" if settings["sat_remove"] else "sat=OFF"
        bar_txt = (f"PREVIEW  [{ch_lbl} thr={thr_lbl} {sat_lbl} x{settings['upscale']}"
                   f" ocr={ocr_lbl} csv={csv_lbl}]  capturing PAUSED")
        bar_col = (0, 110, 40)
    else:
        bar_txt = (f"LIVE  next OCR {next_in:.1f}s  [{ocr_lbl} csv={csv_lbl}]"
                   f"  Z1: {z1_res}  |  Z2: {z2_res}")
        bar_col = (30, 30, 30)

    cv2.rectangle(display, (0,h_d-34), (w_d,h_d), bar_col, -1)
    cv2.putText(display, bar_txt, (8,h_d-10), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255,255,255), 1, cv2.LINE_AA)

    # ── Badges ──────────────────────────────────────────────────────────────
    def badge(x1, y1, x2, y2, col, txt):
        cv2.rectangle(display, (x1,y1), (x2,y2), col, -1)
        cv2.rectangle(display, (x1,y1), (x2,y2), (160,160,160), 1)
        cv2.putText(display, txt, (x1+7,(y1+y2)//2+5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255,255,255), 1, cv2.LINE_AA)

    # Preview badge
    bc = (0,180,65) if show_preview else (50,50,50)
    badge(8,   8, 148, 42, bc, "PREVIEW ON" if show_preview else "PREVIEW OFF")

    # Zone 1 badge (cyan)
    z1_col = (120, 140, 0) if ZONES[1]["roi"] else (55, 55, 55)
    badge(156, 8, 296, 42, z1_col,
          f"Z1: {z1_res[:14]}" if ZONES[1]["roi"] else "Z1: no ROI")

    # Zone 2 badge (magenta)
    z2_col = (120, 0, 140) if ZONES[2]["roi"] else (55, 55, 55)
    badge(304, 8, 444, 42, z2_col,
          f"Z2: {z2_res[:14]}" if ZONES[2]["roi"] else "Z2: no ROI")

    # CSV badge
    csv_col = (0, 60, 200) if csv_saving else (55, 55, 55)
    csv_txt = f"● REC {csv_row_count}" if csv_saving else "CSV off"
    badge(452, 8, 574, 42, csv_col, csv_txt)

    # ── Big OCR results (two lines) ─────────────────────────────────────────
    if not show_preview:
        good = lambda r: r not in ("waiting...", "[no text detected]",
                                   "[EasyOCR still loading]", "waiting for EasyOCR...")
        y_offset = h_d - 75
        if good(z1_res):
            cv2.putText(display, f"Z1: {z1_res}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 220, 255), 3, cv2.LINE_AA)
        if good(z2_res):
            cv2.putText(display, f"Z2: {z2_res}", (10, y_offset - 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 80, 255), 3, cv2.LINE_AA)

    cv2.imshow(WIN, display)


cam_active = False
cv2.destroyAllWindows()
print(f"\nSession ended. {csv_row_count} rows written to {CSV_PATH}" if csv_row_count else "\nSession ended.")