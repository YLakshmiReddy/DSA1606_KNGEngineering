"""
Step 8 - Text Extraction: use Tesseract OCR to read the characters.
Install:
    pip install opencv-python pytesseract matplotlib numpy
    (and the Tesseract OCR program itself - see the Q4 notes)
Run:  python step08_text_extraction.py car.jpg
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


def detect_plate_haar(img):
    """Detect plate using OpenCV's pretrained Haar cascade."""
    cascade_path = cv2.data.haarcascades + "haarcascade_russian_plate_number.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plates = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 20))
    if len(plates) == 0:
        return None
    x, y, w, h = max(plates, key=lambda p: p[2] * p[3])   # largest detection
    return x, y, w, h


def detect_plate_contour(img):
    """Fallback: look for a rectangular contour with a plate-like aspect ratio."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    edges = cv2.Canny(gray, 30, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]

    img_area = img.shape[0] * img.shape[1]
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * peri, True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect = w / float(h)
            area_ratio = (w * h) / img_area
            if 2.0 <= aspect <= 6.5 and 0.005 <= area_ratio <= 0.5:
                return x, y, w, h
    return None


def detect_plate(img):
    box = detect_plate_haar(img)
    if box is None:
        box = detect_plate_contour(img)
    if box is None:
        return None, None
    x, y, w, h = box
    pad = 4
    x0, y0 = max(x - pad, 0), max(y - pad, 0)
    x1, y1 = min(x + w + pad, img.shape[1]), min(y + h + pad, img.shape[0])
    return img[y0:y1, x0:x1], (x0, y0, x1 - x0, y1 - y0)


def preprocess_plate(plate):
    """Return a list of preprocessed versions of the plate for OCR."""
    h, w = plate.shape[:2]
    scale = max(1.0, 300.0 / h)                                   # enlarge small plates
    plate = cv2.resize(plate, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)                # grayscale
    blur = cv2.bilateralFilter(gray, 9, 75, 75)                   # noise reduction
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast = clahe.apply(blur)                                  # contrast enhancement

    _, otsu = cv2.threshold(contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(contrast, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 31, 10)

    # Add a white border so characters do not touch the image edge
    otsu = cv2.copyMakeBorder(otsu, 15, 15, 15, 15, cv2.BORDER_CONSTANT, value=255)
    adaptive = cv2.copyMakeBorder(adaptive, 15, 15, 15, 15, cv2.BORDER_CONSTANT, value=255)
    return [otsu, adaptive, contrast]


def run_ocr(image):
    """Run Tesseract with a few page-segmentation modes; return cleaned candidates."""
    whitelist = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    candidates = []
    for psm in (7, 8, 13):
        config = f"--oem 3 --psm {psm} -c tessedit_char_whitelist={whitelist}"
        text = pytesseract.image_to_string(image, config=config)
        cleaned = re.sub(r"[^A-Z0-9]", "", text.upper())
        if cleaned:
            candidates.append(cleaned)
    return candidates


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else input("Enter car image path: ").strip().strip('"')
    img = load_image(path)
    if img is None:
        return
    plate, _ = detect_plate(img)
    if plate is None:
        print("Error: number plate could not be detected.")
        return

    names = ["Otsu threshold", "Adaptive threshold", "Contrast enhanced"]
    for name, version in zip(names, preprocess_plate(plate)):
        print(f"OCR on {name}: {run_ocr(version) or '(nothing read)'}")
    print("\nNote: raw OCR text can contain mistakes. Step 10 validates it.")


if __name__ == "__main__":
    main()
