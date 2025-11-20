import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import time
import threading
import cv2
from tkinter import Tk, Button, Label, Entry, Frame, messagebox
from tkinter import ttk
from PIL import Image, ImageTk
from datetime import datetime
import tkinter as tk

# Load and resize the logo
logo = cv2.imread('TK.png')
logo = cv2.resize(logo, (25, 25))

# Function to add logo to the frame
def add_logo(frame):
    x_offset = frame.shape[1] - logo.shape[1] - 10
    y_offset = 10
    frame[y_offset:y_offset+logo.shape[0], x_offset:x_offset+logo.shape[1]] = logo
    return frame

def capture_snapshots():
    global capturing, img_label, timer_label, start_time, next_capture_time, folder_entry, test_number, last_images, status_label, interval, selected_camera_index
    cap = cv2.VideoCapture(selected_camera_index)
    if not cap.isOpened():
        print("Cannot open camera")
        status_label.config(text="Cannot open camera", foreground="red")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

    # Create the folder if it doesn't exist
    folder_name = folder_entry.get()
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    status_label.config(text="Image Recording", foreground="red")

    while capturing:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            status_label.config(text="Failed to grab frame", foreground="red")
            break

        # Add timestamp to the bottom right of the frame
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        timestamp2 = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        text_size = cv2.getTextSize(timestamp, font, font_scale, font_thickness)[0]
        text_x = frame.shape[1] - text_size[0] - 10
        text_y = frame.shape[0] - 10
        cv2.putText(frame, timestamp, (text_x, text_y), font, font_scale, (0, 255, 0), font_thickness, cv2.LINE_AA)

        # Add logo to the frame
        frame = add_logo(frame)

        # Save the frame as an image file with test number and timestamp in the filename
        filename = f'{folder_name}/test_{test_number}_{timestamp2}.jpeg'
        cv2.imwrite(filename, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        test_number += 1

        # Update the image in the GUI with smaller size
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img = img.resize((320, 320), Image.LANCZOS)
        imgtk = ImageTk.PhotoImage(image=img)
        img_label.config(image=imgtk)
        img_label.image = imgtk

        # Update the last images list and display them
        last_images.append(filename)
        if len(last_images) > 5:
            last_images.pop(0)
        update_last_images()

        # Reset the timer and set the next capture time
        start_time = time.time()
        next_capture_time = start_time + interval

        # Wait for the selected interval
        for i in range(interval):
            if not capturing:
                break
            seconds_left = int(next_capture_time - time.time())
            root.after(0, update_timer_label, test_number - 1, seconds_left)
            time.sleep(1)

    cap.release()
    status_label.config(text="Not Recording", foreground="black")

def capture_video():
    global capturing, folder_entry, test_number, status_label, selected_camera_index

    # Stop capturing images
    capturing = False

    cap = cv2.VideoCapture(selected_camera_index)
    if not cap.isOpened():
        print("Cannot open camera")
        status_label.config(text="Cannot open camera", foreground="red")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

    # Create the folder if it doesn't exist
    folder_name = folder_entry.get()
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(f'{folder_name}/test_{test_number}.avi', fourcc, 20.0, (640, 480))

    # Update status label to "Video Recording"
    status_label.config(text="Video Recording", foreground="red")

    start_time = time.time()
    while time.time() - start_time < 10:  # Capture video for 10 seconds
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            status_label.config(text="Failed to grab frame", foreground="red")
            break

        # Add timestamp to the bottom right of the frame
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        text_size = cv2.getTextSize(timestamp, font, font_scale, font_thickness)[0]
        text_x = frame.shape[1] - text_size[0] - 10
        text_y = frame.shape[0] - 10
        cv2.putText(frame, timestamp, (text_x, text_y), font, font_scale, (0, 255, 0), font_thickness, cv2.LINE_AA)

        # Add logo to the frame
        frame = add_logo(frame)

        out.write(frame)

    cap.release()
    out.release()
    test_number += 1

    # Update status label to "Not Recording"
    status_label.config(text="Not Recording", foreground="black")

    # Resume capturing images
    capturing = True
    threading.Thread(target=capture_snapshots).start()

def update_timer_label(pictures_saved, seconds_left):
    timer_label.config(text=f'Pictures Saved: {pictures_saved} | Next Capture In: {seconds_left} seconds')

def update_last_images():
    for i, img_path in enumerate(last_images):
        img = Image.open(img_path)
        img = img.resize((80, 80), Image.LANCZOS)
        imgtk = ImageTk.PhotoImage(image=img)
        last_image_labels[i].config(image=imgtk)
        last_image_labels[i].image = imgtk

# Function to start capturing snapshots and take the first image immediately
def start_capture():
    global capturing, start_time, next_capture_time, test_number, interval, selected_camera_index
    folder_name = folder_entry.get()
    if not folder_name:
        messagebox.showerror("Error", "You need to write the folder name")
        return

    # Retrieve the interval value from the dropdown menu and convert it to an integer
    interval = int(interval_var.get())
    selected_camera_index = int(camera_var.get())

    capturing = True
    start_time = time.time()
    next_capture_time = start_time + interval
    test_number = 1

    # Hide the start button, folder entry, and interval dropdown, show stop and pause buttons
    start_button.grid_remove()
    folder_label.grid_remove()
    folder_entry.grid_remove()
    interval_label.grid_remove()
    interval_dropdown.grid_remove()
    camera_label.grid_remove()
    camera_dropdown.grid_remove()
    stop_button.grid(row=2, column=0, padx=5, pady=5)
    pause_button.grid(row=2, column=1, padx=5, pady=5)
    video_button.grid(row=2, column=2, padx=5, pady=5)

    # Take the first image immediately
    threading.Thread(target=capture_snapshots).start()

# Function to stop capturing snapshots and reset the program
def stop_capture():
    global capturing
    capturing = False

    # Reset the GUI elements
    stop_button.grid_remove()
    pause_button.grid_remove()
    resume_button.grid_remove()
    video_button.grid_remove()

    folder_label.grid(row=0, column=0, padx=5, pady=5)
    folder_entry.grid(row=0, column=1, padx=5, pady=5)
    interval_label.grid(row=1, column=0, padx=5, pady=5)
    interval_dropdown.grid(row=1, column=1, padx=5, pady=5)
    camera_label.grid(row=3, column=0, padx=5, pady=5)
    camera_dropdown.grid(row=3, column=1, padx=5, pady=5)
    start_button.grid(row=4, column=0, columnspan=2, pady=10)

# Function to pause capturing snapshots and reset the timer
def pause_capture():
    global capturing, start_time, next_capture_time
    capturing = False

    # Reset the timer
    start_time = time.time()
    next_capture_time = start_time + interval

    # Show resume button and hide pause button
    pause_button.grid_remove()
    resume_button.grid(row=2, column=1, padx=5, pady=5)

# Function to resume capturing snapshots
def resume_capture():
    global capturing, start_time, next_capture_time
    capturing = True

    # Show pause button and hide resume button
    resume_button.grid_remove()
    pause_button.grid(row=2, column=1, padx=5, pady=5)

    start_time = time.time()
    next_capture_time = start_time + interval

    threading.Thread(target=capture_snapshots).start()

# Function to handle window close event
def on_closing():
    global capturing
    capturing = False
    root.destroy()

# Function to show help message
def show_help():
    help_message = (
        "This program captures snapshots from your webcam every 5, 10, 15, or 20 seconds and saves them to a specified folder.\n"
        "Here's how it works:\n\n"
        "1. **Start**: Enter the folder name, select the interval, select the camera and click 'Start' to begin capturing snapshots.\n"
        "2. **Pause**: Click 'Pause' to temporarily stop capturing snapshots.\n"
        "3. **Resume**: Click 'Resume' to continue capturing snapshots after pausing.\n"
        "4. **Stop**: Click 'Stop' to end the capturing process.\n"
        "5. **Capture Video**: Click 'Capture Video' to record a video for the selected interval.\n"
        "6. **Status**: The status label indicates whether the program is recording images or video.\n"
        "7. **Timer**: The timer label shows the number of pictures saved and the time left until the next capture.\n"
        "8. **Last Images**: The last 5 captured images are displayed at the bottom of the window.\n"
        "\n\nThermo King - Galway"
        "\n\nAuthor: Liam Noone"
    )
    messagebox.showinfo("Help", help_message)

# Create a simple GUI with start and stop buttons and an image label
root = Tk()
root.title("TK Snapshot")
root.geometry("600x700")

# Bind the window close event to the on_closing function
root.protocol("WM_DELETE_WINDOW", on_closing)

# Use ttk for modern look
style = ttk.Style()
style.configure('TButton', font=('Helvetica', 12))
style.configure('TLabel', font=('Helvetica', 12))

main_frame = Frame(root)
main_frame.pack(pady=20)

folder_label = ttk.Label(main_frame, text="Folder Name:")
folder_label.grid(row=0, column=0, padx=5, pady=5)

folder_entry = ttk.Entry(main_frame)
folder_entry.grid(row=0, column=1, padx=5, pady=5)

interval_label = ttk.Label(main_frame, text="Interval (seconds):")
interval_label.grid(row=1, column=0, padx=5, pady=5)

interval_var = tk.StringVar(value="10")
interval_dropdown = ttk.Combobox(main_frame, textvariable=interval_var, values=["5", "10", "15", "20", "30"], state="readonly")
interval_dropdown.grid(row=1, column=1, padx=5, pady=5)

# Detect available cameras
def detect_cameras():
    cameras = []
    for i in range(10):  # Check up to 10 cameras
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            cameras.append(str(i))
            cap.release()
        else:
            cap.release()
    return cameras

camera_label = ttk.Label(main_frame, text="Camera:")
camera_label.grid(row=3, column=0, padx=5, pady=5)

available_cameras = detect_cameras()
camera_var = tk.StringVar(value=available_cameras[0] if available_cameras else "0") # Default to the first camera or 0
camera_dropdown = ttk.Combobox(main_frame, textvariable=camera_var, values=available_cameras, state="readonly")
camera_dropdown.grid(row=3, column=1, padx=5, pady=5)

start_button = ttk.Button(main_frame, text="Start", command=start_capture)
start_button.grid(row=4, column=0, columnspan=2, pady=10)

stop_button = ttk.Button(main_frame, text="Stop", command=stop_capture)
pause_button = ttk.Button(main_frame, text="Pause", command=pause_capture)
resume_button = ttk.Button(main_frame, text="Resume", command=resume_capture)
video_button = ttk.Button(main_frame, text="Capture Video", command=capture_video)

# Add the help button
help_button = ttk.Button(main_frame, text="Help", command=show_help)
help_button.grid(row=5, column=0, columnspan=2, pady=5)

img_label = ttk.Label(root)
img_label.pack(pady=10)

timer_label = ttk.Label(root, text='Pictures Saved: 0 | Next Capture In: 10 seconds')
timer_label.pack(pady=10)

# Add status label for recording status
status_label = ttk.Label(root, text='Not Recording', foreground='black')
status_label.pack(pady=10)

# Create labels for the last 5 images
last_images_frame = Frame(root)
last_images_frame.pack(pady=10)
last_image_labels = [ttk.Label(last_images_frame) for _ in range(5)]
for i, label in enumerate(last_image_labels):
    label.grid(row=0, column=i, padx=5)

last_images = []

capturing = False

# Bind the window close event to the on_closing function
root.protocol("WM_DELETE_WINDOW", on_closing)

root.mainloop()