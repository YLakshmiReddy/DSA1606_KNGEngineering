"""
Step 11 - Result Display: show the original image, cropped plate,
extracted registration number and validation result.
Install:
    pip install opencv-python pytesseract matplotlib numpy
    (and the Tesseract OCR program itself - see the Q4 notes)
Run:  python step11_result_display.py car.jpg
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


KNOWLEDGE_BASE = {
    "format_description": "SS DD LLL NNNN  (State code, District number, Series letters, 4-digit number)",
    "pattern": re.compile(r"^[A-Z]{2}[0-9]{2}[A-Z]{1,3}[0-9]{4}$"),
    "state_codes": {
        "AN", "AP", "AR", "AS", "BR", "CH", "CG", "DD", "DL", "DN", "GA", "GJ",
        "HP", "HR", "JH", "JK", "KA", "KL", "LA", "LD", "MH", "ML", "MN", "MP",
        "MZ", "NL", "OD", "OR", "PB", "PY", "RJ", "SK", "TN", "TR", "TS", "UK",
        "UP", "WB",
    },
    # Characters OCR commonly confuses
    "to_digit": {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "Z": "2",
                 "S": "5", "B": "8", "G": "6", "T": "7"},
    "to_letter": {"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B", "6": "G"},
}


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


def apply_ocr_corrections(text):
    """
    Rule: the plate layout is  L L D D  L(1-3)  D D D D.
    Using the position of each character, replace look-alike characters
    (e.g. 'O' where a digit is expected -> '0').
    """
    n = len(text)
    if not 9 <= n <= 11:
        return text, []
    series_len = n - 8
    layout = ["L", "L", "D", "D"] + ["L"] * series_len + ["D"] * 4

    fixed, changes = [], []
    for i, (ch, expected) in enumerate(zip(text, layout)):
        new = ch
        if expected == "D" and ch.isalpha():
            new = KNOWLEDGE_BASE["to_digit"].get(ch, ch)
        elif expected == "L" and ch.isdigit():
            new = KNOWLEDGE_BASE["to_letter"].get(ch, ch)
        if new != ch:
            changes.append(f"position {i + 1}: '{ch}' -> '{new}'")
        fixed.append(new)
    return "".join(fixed), changes


def validate_plate(text):
    """Return (is_valid, final_text, messages)."""
    messages = []
    if not text:
        return False, "", ["Rule 1 failed: no characters were read"]

    if not 9 <= len(text) <= 11:
        return False, text, [f"Rule 2 failed: length {len(text)} is not between 9 and 11"]

    corrected, changes = apply_ocr_corrections(text)
    if changes:
        messages.append("OCR correction applied: " + "; ".join(changes))

    if not KNOWLEDGE_BASE["pattern"].match(corrected):
        messages.append("Rule 3 failed: text does not match format SS DD LLL NNNN")
        return False, corrected, messages

    if corrected[:2] not in KNOWLEDGE_BASE["state_codes"]:
        messages.append(f"Rule 4 failed: '{corrected[:2]}' is not a valid state/UT code")
        return False, corrected, messages

    messages.append("All rules satisfied")
    return True, corrected, messages


def show_results(img, box, plate, text, status):
    vis = img.copy()
    if box is not None:
        x, y, w, h = box
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 3)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original image")
    axes[0].axis("off")

    if plate is not None:
        axes[1].imshow(cv2.cvtColor(plate, cv2.COLOR_BGR2RGB))
    else:
        axes[1].imshow(np.ones((50, 200, 3), dtype=np.uint8) * 230)
    axes[1].set_title(f"Cropped plate\nNumber: {text}   Result: {status}")
    axes[1].axis("off")
    plt.tight_layout()
    plt.show()


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else input("Enter car image path: ").strip().strip('"')
    img = load_image(path)
    if img is None:
        return
    plate, box = detect_plate(img)
    if plate is None:
        show_results(img, None, None, "NOT DETECTED", "Plate not found")
        return

    final, status = "", "INVALID"
    for version in preprocess_plate(plate):
        for candidate in run_ocr(version):
            ok, text, _ = validate_plate(candidate)
            if ok:
                final, status = text, "VALID"
                break
            final = final or candidate
        if status == "VALID":
            break

    print("Extracted registration number:", final or "(nothing read)")
    print("Validation result            :", status)
    show_results(img, box, plate, final or "UNREADABLE", status)


if __name__ == "__main__":
    main()
