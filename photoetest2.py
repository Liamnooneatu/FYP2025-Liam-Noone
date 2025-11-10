import cv2
import torch
import lpips
from PIL import Image
import torchvision.transforms as transforms
import time
import os

# ----------- Configuration ----------- #
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

# ----------- Camera Setup ----------- #
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open camera")

last_check = 0
prev_tensor = None  # Stores the previous frame

# ----------- Main Loop ----------- #
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Display the live camera feed
    cv2.imshow("Camera Feed", frame)

    # Every CHECK_INTERVAL seconds, compare current frame to previous
    if time.time() - last_check >= CHECK_INTERVAL:
        last_check = time.time()
        curr_tensor = cv2_to_tensor(frame)

        if prev_tensor is not None:
            dist = loss_fn(prev_tensor, curr_tensor).item()
            print(f"LPIPS distance: {dist:.3f}")

            if dist > LPIPS_THRESHOLD:
                print(">>> CHANGE DETECTED <<<")
                if SAVE_CHANGES:
                    filename = os.path.join(SAVE_FOLDER, f"change_{int(time.time())}.jpg")
                    cv2.imwrite(filename, frame)
                    print(f"Saved changed frame: {filename}")
            else:
                print("No significant change")
        else:
            print("First frame captured, no comparison yet.")

        # Update previous frame
        prev_tensor = curr_tensor

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
