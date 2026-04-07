import cv2
import torch
import lpips
from PIL import Image
import torchvision.transforms as transforms
import time
import os
import easyocr
from datetime import datetime
import csv
import threading
import queue
from concurrent.futures import ThreadPoolExecutor

LPIPS_THRESHOLD = 0.045
CHECK_INTERVAL = 2
SAVE_FOLDER = "detected_changes"
CSV_FILE = os.path.join(SAVE_FOLDER, "detections.csv")

print("Loading EasyOCR model...")
reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
print("EasyOCR ready.")

if not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)


# ── CSV ───────────────────────────────────────────────────────────────────────
HEADER = ["Zone 1", "Z1_Timestamp", "Z1_Text",
          "Zone 2", "Z2_Timestamp", "Z2_Text"]

csv_lock = threading.Lock()

def ensure_csv_header():
    if not os.path.exists(CSV_FILE) or os.path.getsize(CSV_FILE) == 0:
        with open(CSV_FILE, mode="w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(HEADER)

def append_to_csv(zone: int, timestamp: str, text: str):
    """Thread-safe CSV append — called from background thread only."""
    ensure_csv_header()
    entry = [f"Zone {zone}", timestamp, text.strip()]
    blank = ["", "", ""]

    with csv_lock:
        with open(CSV_FILE, mode="r", newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))

        header = rows[0]
        data = rows[1:]

        paired = False
        for i, row in enumerate(data):
            if len(row) < 6:
                row += [""] * (6 - len(row))
            if zone == 1 and row[0] == "" and row[3] != "":
                data[i] = entry + row[3:]
                paired = True
                break
            elif zone == 2 and row[3] == "" and row[0] != "":
                data[i] = row[:3] + entry
                paired = True
                break

        if not paired:
            data.append(entry + blank if zone == 1 else blank + entry)

        with open(CSV_FILE, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(data)

ensure_csv_header()


# ── LPIPS (CPU-only for laptops — much lighter than GPU init overhead) ─────────
device = torch.device('cpu')   # Force CPU; LPIPS on CPU is fine for 128x128
loss_fn = lpips.LPIPS(net='alex').to(device).eval()

LINE_GRAB_PX = 10
MIN_ROI_SIZE = 30

# ── Background worker ─────────────────────────────────────────────────────────
# OCR and CSV writes go here so the camera loop never blocks.
ocr_queue = queue.Queue(maxsize=4)   # drop if falling too far behind

def ocr_worker():
    executor = ThreadPoolExecutor(max_workers=1)
    while True:
        item = ocr_queue.get()
        if item is None:
            break
        zid, roi, timestamp = item
        try:
            extracted_text = extract_text_from_roi(roi)
            append_to_csv(zid, timestamp, extracted_text)
            if zid == 1:
                print(f"CHANGE DETECTED Zone 1 | {timestamp} | {extracted_text}")
            else:
                print(f"CHANGE DETECTED Zone 2 | {timestamp} | {extracted_text}")
        except Exception as e:
            print(f"OCR worker error: {e}")
        finally:
            ocr_queue.task_done()

ocr_thread = threading.Thread(target=ocr_worker, daemon=True)
ocr_thread.start()


# ── Zone state ────────────────────────────────────────────────────────────────
zones = {
    1: {"x1": 0, "y1": 0, "x2": 0, "y2": 0,
        "drag_left": False, "drag_right": False, "drag_top": False, "drag_bottom": False,
        "moved": False, "prev_tensor": None,
        "color": (0, 220, 0), "label": "Zone 1"},
    2: {"x1": 0, "y1": 0, "x2": 0, "y2": 0,
        "drag_left": False, "drag_right": False, "drag_top": False, "drag_bottom": False,
        "moved": False, "prev_tensor": None,
        "color": (0, 165, 255), "label": "Zone 2"},
}

BUTTONS = {
    1: {"x1": 10,  "y1": 10, "x2": 150, "y2": 50, "label": "RESET Z1"},
    2: {"x1": 160, "y1": 10, "x2": 300, "y2": 50, "label": "RESET Z2"},
}


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def normalize_roi(x1_, y1_, x2_, y2_, w, h):
    x1_ = clamp(x1_, 0, w - 1);  x2_ = clamp(x2_, 0, w - 1)
    y1_ = clamp(y1_, 0, h - 1);  y2_ = clamp(y2_, 0, h - 1)
    if x1_ > x2_: x1_, x2_ = x2_, x1_
    if y1_ > y2_: y1_, y2_ = y2_, y1_
    if (x2_ - x1_) < MIN_ROI_SIZE:
        x2_ = clamp(x1_ + MIN_ROI_SIZE, 0, w - 1)
        x1_ = clamp(x2_ - MIN_ROI_SIZE, 0, w - 1)
    if (y2_ - y1_) < MIN_ROI_SIZE:
        y2_ = clamp(y1_ + MIN_ROI_SIZE, 0, h - 1)
        y1_ = clamp(y2_ - MIN_ROI_SIZE, 0, h - 1)
    return x1_, y1_, x2_, y2_


def reset_zone(zone_id, w, h):
    z = zones[zone_id]
    cx, cy = w // 2, h // 2
    offset = -(w // 5) if zone_id == 1 else (w // 5)
    z["x1"] = clamp(cx + offset - w // 8, 0, w - 1)
    z["y1"] = clamp(cy - h // 8, 0, h - 1)
    z["x2"] = clamp(cx + offset + w // 8, 0, w - 1)
    z["y2"] = clamp(cy + h // 8, 0, h - 1)
    z["x1"], z["y1"], z["x2"], z["y2"] = normalize_roi(
        z["x1"], z["y1"], z["x2"], z["y2"], w, h)
    z["moved"] = True
    z["prev_tensor"] = None
    print(f"Zone {zone_id} reset.")


def cv2_to_tensor(img_bgr):
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img)
    t = transforms.Compose([
        transforms.Resize((64, 64)),          # ↓ from 128 — much faster, still works
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return t(pil_img).unsqueeze(0).to(device)


def preprocess_for_ocr(image_bgr):
    upscaled = cv2.resize(image_bgr, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    _, thresh = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)


def extract_text_from_roi(roi_bgr):
    preprocessed = preprocess_for_ocr(roi_bgr)
    results = reader.readtext(preprocessed, allowlist='0123456789.-', detail=0, paragraph=False)
    text = ' '.join(results).strip()
    return text if text else "[no text detected]"


# ── Mouse callback ────────────────────────────────────────────────────────────
active_zone = None

def on_mouse(event, mx, my, flags, param):
    global active_zone
    w, h = param["w"], param["h"]

    if event == cv2.EVENT_LBUTTONDOWN:
        for zid, btn in BUTTONS.items():
            if btn["x1"] <= mx <= btn["x2"] and btn["y1"] <= my <= btn["y2"]:
                reset_zone(zid, w, h)
                active_zone = None
                return

        best_zone, best_side, best_dist = None, None, float('inf')
        for zid, z in zones.items():
            for side, dist in [("left",   abs(mx - z["x1"])),
                                ("right",  abs(mx - z["x2"])),
                                ("top",    abs(my - z["y1"])),
                                ("bottom", abs(my - z["y2"]))]:
                if dist < best_dist:
                    best_dist, best_zone, best_side = dist, zid, side

        if best_dist <= LINE_GRAB_PX:
            active_zone = best_zone
            z = zones[best_zone]
            z["drag_left"]   = best_side == "left"
            z["drag_right"]  = best_side == "right"
            z["drag_top"]    = best_side == "top"
            z["drag_bottom"] = best_side == "bottom"

    elif event == cv2.EVENT_MOUSEMOVE and active_zone is not None:
        z = zones[active_zone]
        changed = False
        if z["drag_left"]:   z["x1"] = mx; changed = True
        if z["drag_right"]:  z["x2"] = mx; changed = True
        if z["drag_top"]:    z["y1"] = my; changed = True
        if z["drag_bottom"]: z["y2"] = my; changed = True
        if changed:
            z["x1"], z["y1"], z["x2"], z["y2"] = normalize_roi(
                z["x1"], z["y1"], z["x2"], z["y2"], w, h)
            z["moved"] = True

    elif event == cv2.EVENT_LBUTTONUP:
        if active_zone:
            z = zones[active_zone]
            z["drag_left"] = z["drag_right"] = z["drag_top"] = z["drag_bottom"] = False
        active_zone = None


# ── Camera setup ──────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open camera")

# Lower resolution to reduce per-frame processing load
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

WINDOW_NAME = "Camera Feed - Dual Zone"
cv2.namedWindow(WINDOW_NAME)

ret, frame = cap.read()
if not ret:
    raise RuntimeError("Failed to grab initial frame")

h, w = frame.shape[:2]
reset_zone(1, w, h)
reset_zone(2, w, h)
cv2.setMouseCallback(WINDOW_NAME, on_mouse, param={"w": w, "h": h})

last_check = 0



print("System running... press 'q' to quit.")
print("Press '1' or '2' to reset individual zones.")
print(f"CSV logging to: {CSV_FILE}")

# ── Main loop ─────────────────────────────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Draw reset buttons
    for zid, btn in BUTTONS.items():
        color = zones[zid]["color"]
        cv2.rectangle(overlay, (btn["x1"], btn["y1"]), (btn["x2"], btn["y2"]), (40, 40, 40), -1)
        cv2.rectangle(overlay, (btn["x1"], btn["y1"]), (btn["x2"], btn["y2"]), color, 2)
        cv2.putText(overlay, btn["label"], (btn["x1"] + 10, btn["y1"] + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Draw both zones
    for zid, z in zones.items():
        z["x1"], z["y1"], z["x2"], z["y2"] = normalize_roi(
            z["x1"], z["y1"], z["x2"], z["y2"], w, h)
        x1, y1, x2, y2 = z["x1"], z["y1"], z["x2"], z["y2"]
        color = z["color"]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        cv2.putText(overlay, z["label"], (x1 + 5, y1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(overlay, f"{x2-x1}x{y2-y1}", (x1 + 5, y2 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    cv2.imshow(WINDOW_NAME, overlay)

    # ── LPIPS check (lightweight — just tensor math, no OCR) ──────────────────
    now = time.time()
    if now - last_check >= CHECK_INTERVAL:
        last_check = now

        for zid, z in zones.items():
            x1, y1, x2, y2 = z["x1"], z["y1"], z["x2"], z["y2"]
            roi = frame[y1:y2, x1:x2]

            if roi.size == 0 or roi.shape[0] < MIN_ROI_SIZE or roi.shape[1] < MIN_ROI_SIZE:
                continue

            if z["moved"]:
                z["prev_tensor"] = None
                z["moved"] = False
                print(f"Zone {zid} moved -> baseline reset.")
                continue

            curr_tensor = cv2_to_tensor(roi)

            if z["prev_tensor"] is not None:
                with torch.no_grad():
                    dist = loss_fn(z["prev_tensor"], curr_tensor).item()

                if dist > LPIPS_THRESHOLD:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    roi_copy = roi.copy()   # copy before next frame overwrites
                    roi_image_file = os.path.join(
                        SAVE_FOLDER,
                        f"zone{zid}_change_{timestamp.replace(':', '-')}.jpg")
                    cv2.imwrite(roi_image_file, roi_copy)

                    # Push to background — non-blocking
                    try:
                        ocr_queue.put_nowait((zid, roi_copy, timestamp))
                    except queue.Full:
                        print(f"Zone {zid}: OCR queue full, skipping this detection.")

                    # Advance baseline so next check compares against the new
                    # state rather than the old one — prevents repeat triggers
                    z["prev_tensor"] = curr_tensor
                    continue   # skip the update below

            z["prev_tensor"] = curr_tensor

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('1'):
        reset_zone(1, w, h)
    if key == ord('2'):
        reset_zone(2, w, h)

# ── Cleanup ───────────────────────────────────────────────────────────────────
ocr_queue.put(None)   # signal worker to stop
ocr_thread.join(timeout=10)
cap.release()
cv2.destroyAllWindows()