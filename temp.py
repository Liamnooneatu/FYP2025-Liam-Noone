import cv2, os
import pytesseract
from PIL import Image

# Set your Tesseract path
pytesseract.pytesseract.tesseract_cmd = r"D:\Tesseract-OCR\tesseract.exe"

# Define your image and preprocessor manually
image_path = r"D:\college\Year 4\final year code\testimage3.jpg"
pre_processor = "thresh"

# Load image
images = cv2.imread(image_path)
gray = cv2.cvtColor(images, cv2.COLOR_BGR2GRAY)

# Apply preprocessing
if pre_processor == "thresh":
    gray = cv2.threshold(gray, 0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
elif pre_processor == "blur":
    gray = cv2.medianBlur(gray, 3)

# Save temporary file and extract text
filename = "temp_image.jpg"
cv2.imwrite(filename, gray)
text = pytesseract.image_to_string(Image.open(filename))
os.remove(filename)

print("Extracted text:")
print(text)

# Show the output
cv2.imshow("Original", images)
cv2.imshow("Processed", gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
