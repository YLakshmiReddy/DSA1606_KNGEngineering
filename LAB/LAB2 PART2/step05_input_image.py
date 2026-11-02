"""
Step 5 - Input Image: read a car image (JPG/JPEG/PNG) and display it.
Install:
    pip install opencv-python pytesseract matplotlib numpy
    (and the Tesseract OCR program itself - see the Q4 notes)
Run:  python step05_input_image.py car.jpg
"""
import re
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytesseract

# Windows users: uncomment and fix this path if Tesseract is not on PATH
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def load_image(path):
    """Step 5: read the image after checking its file type."""
    if not path.lower().endswith((".jpg", ".jpeg", ".png")):
        print("Error: only JPG, JPEG or PNG images are supported.")
        return None
    img = cv2.imread(path)
    if img is None:
        print(f"Error: could not read image '{path}'. Check the file path.")
    return img


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else input("Enter car image path: ").strip().strip('"')
    img = load_image(path)
    if img is None:
        return
    print("Image loaded successfully. Size:", img.shape[1], "x", img.shape[0], "pixels")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Input car image")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()
