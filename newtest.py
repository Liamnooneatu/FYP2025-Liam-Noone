import cv2
import torch
import lpips
from PIL import Image
import torchvision.transforms as transforms
import time
import os
import pytesseract
from docx import Document
from datetime import datetime

# ----------- CONFIG ----------- #
LPIPS_THRESHOLD = 0.1
CHECK_INTERVAL = 5
SAVE_FOLDER = "detected_changes"
pytesseract.pytesseract.tesseract_cmd = r"D:\Tesseract-OCR\tesseract.exe"

# ----------- CREATE SAVE FOLDER ----------- #
if not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)

# ----------- LPIPS MODEL ----------- #
loss_fn = lpips.LPIPS(net='alex')
loss_fn.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loss_fn = loss_fn.to(device)

# ----------- CONVERT CV2 -> LPIPS TENSOR ----------- #
def cv2_to_tensor(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
    ])
    return transform(pil_img).unsqueeze(0).to(device)

# ----------- OCR FUNCTION ----------- #
def extract_text_from_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Thresholding for better OCR
    gray = cv2.threshold(gray, 0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # Temp file
    temp_file = "temp_ocr.jpg"
    cv2.imwrite(temp_file, gray)

    text = pytesseract.image_to_string(Image.open(temp_file))
    os.remove(temp_file)
    return text

# ----------- WRITE TO WORD DOC ----------- #
def write_to_word(timestamp, text):
    doc = Document()
    doc.add_heading("Camera Change Detection", level=1)

    doc.add_paragraph(f"Time detected: {timestamp}")
    doc.add_paragraph("Extracted text:")
    doc.add_paragraph(text)

    filename = f"Camera_Detection_{timestamp.replace(':','-')}.docx"
    doc.save(filename)
    print(f">>> Word document saved as: {filename}")

# ----------- CAMERA SETUP ----------- #
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open camera")

last_check = 0
prev_tensor = None

print("System running... press 'q' to quit.")

# ----------- MAIN LOOP ----------- #
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    cv2.imshow("Camera Feed", frame)

    # Run LPIPS comparison at fixed intervals
    if time.time() - last_check >= CHECK_INTERVAL:
        last_check = time.time()
        curr_tensor = cv2_to_tensor(frame)

        if prev_tensor is not None:
            dist = loss_fn(prev_tensor, curr_tensor).item()
            print(f"LPIPS Distance: {dist:.3f}")

            if dist > LPIPS_THRESHOLD:
                print(">>> CHANGE DETECTED <<<")

                timestamp = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
                image_file = os.path.join(SAVE_FOLDER, f"change_{timestamp}.jpg")

                # Save the changed frame
                cv2.imwrite(image_file, frame)
                print(f"Saved changed frame: {image_file}")

                # OCR
                extracted_text = extract_text_from_image(image_file)
                print("Extracted text:")
                print(extracted_text)

                # Write Word document
                write_to_word(timestamp, extracted_text)
            else:
                print("No significant change")

        prev_tensor = curr_tensor

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
