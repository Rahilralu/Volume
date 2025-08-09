import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
import os

# ==== USER SETTINGS ====
known_object_real_width_m = 0.9  # meters (example: door width)

# ==== GLOBAL VARIABLES ====
points = []
img = None
img_display = None

def click_event(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(img_display, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Room Image - Click Two Points", img_display)

        if len(points) == 2:
            # Calculate pixel distance
            pixel_distance = np.linalg.norm(np.array(points[0]) - np.array(points[1]))
            meters_per_pixel = known_object_real_width_m / pixel_distance
            print(f"\n[INFO] Scale: {meters_per_pixel:.5f} meters per pixel")

            # Estimate room size based on image size
            img_height, img_width = img.shape[:2]
            room_width_m = img_width * meters_per_pixel
            room_height_m = img_height * meters_per_pixel

            print(f"[RESULT] Approximate Room Width: {room_width_m:.2f} m")
            print(f"[RESULT] Approximate Room Height: {room_height_m:.2f} m")


# ==== MAIN ====
# Hide root Tkinter window
root = tk.Tk()
root.withdraw()

# Ask user to select an image
file_path = filedialog.askopenfilename(
    title="Select a room image",
    filetypes=[("Image files", "*.jpg *.jpeg *.png")]
)

if not file_path:
    print("❌ No file selected. Exiting.")
    exit()

if not os.path.exists(file_path):
    print("❌ File path does not exist. Exiting.")
    exit()

# Load image
img = cv2.imread(file_path)

if img is None:
    print("❌ Could not read the image. Try re-saving it as a JPEG or PNG.")
    exit()

# Display and wait for clicks
img_display = img.copy()
cv2.imshow("Room Image - Click Two Points", img_display)
cv2.setMouseCallback("Room Image - Click Two Points", click_event)

cv2.waitKey(0)
cv2.destroyAllWindows()
