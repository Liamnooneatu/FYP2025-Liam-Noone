import cv2
import torch
import lpips
from PIL import Image
import torchvision.transforms as transforms
import time
import os

# ----------- Configuration ----------- #
REF_IMAGE_PATH = r"D:/college/Year 4/final year code/test1/WIN_20251110_14_42_47_Pro.jpg"  # Reference image path
LPIPS_THRESHOLD = 0.1             # Adjust sensitivity (lower = more sensitive)
CHECK_INTERVAL = 5                # Seconds between checks
SAVE_CHANGES = True               # Save image if change detected
SAVE_FOLDER = "detected_changes"  # Folder to save changed frames

# ----------- Create folder to save changes ----------- #
if SAVE_CHANGES and not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)

# ----------- LPIPS Model Setup ----------- #
loss_fn = lpips.LPIPS(net='alex')  # Options: 'alex', 'vgg', 'squeeze'
loss_fn.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loss_fn = loss_fn.to(device)

# ----------- Helper: Convert CV2 Image to LPIPS Tensor ----------- #
def cv2_to_tensor(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # LPIPS requires fixed size
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
    ])
    return transform(pil_img).unsqueeze(0).to(device)

# ----------- Load Reference Image ----------- #
ref_img = cv2.imread(REF_IMAGE_PATH)
if ref_img is None:
    raise FileNotFoundError(f"Reference image '{REF_IMAGE_PATH}' not found")
ref_tensor = cv2_to_tensor(ref_img)

# ----------- Camera Setup ----------- #
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open camera")

last_check = 0

# ----------- Main Loop ----------- #
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Set ROI to full frame
    height, width, _ = frame.shape
    x, y, w, h = 0, 0, width, height
    roi = frame[y:y+h, x:x+w]

    # Show live camera feed
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
    cv2.imshow("Camera Feed", frame)

    # Check every CHECK_INTERVAL seconds
    if time.time() - last_check >= CHECK_INTERVAL:
        last_check = time.time()
        roi_tensor = cv2_to_tensor(roi)
        dist = loss_fn(ref_tensor, roi_tensor).item()
        print(f"LPIPS distance: {dist:.3f}")

        if dist > LPIPS_THRESHOLD:
            print(">>> CHANGE DETECTED <<<")
            if SAVE_CHANGES:
                filename = os.path.join(SAVE_FOLDER, f"change_{int(time.time())}.jpg")
                cv2.imwrite(filename, roi)
                print(f"Saved changed frame: {filename}")
        else:
            print("No significant change")

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
