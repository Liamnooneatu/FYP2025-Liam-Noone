import cv2

# Start camera capture (0 = default webcam)
cap = cv2.VideoCapture(0)

# Define ROI coordinates (x, y, width, height)
x, y, w, h = 300, 300, 300, 300

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Crop to ROI
    roi = frame[y:y+h, x:x+w]

    cv2.imshow("ROI", roi)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
