"""
Q3: Knowledge-driven object identification (laptop, book, chair, bottle, mobile phone, table).

How it works
  1. A pretrained image classifier (MobileNetV3 trained on ImageNet) analyses the image.
  2. Its top predictions are matched against a hand-made KNOWLEDGE BASE.
  3. IF-THEN rules decide the object, category, function and whether the system is confident.

Install:
    pip install torch torchvision pillow matplotlib numpy
    (the pretrained model is downloaded automatically the first time - internet needed once)

Run:
    python q3_object_identification.py photo.jpg              # single image
    python q3_object_identification.py --folder test_images   # test many images and get accuracy
        (name test files like  laptop_1.jpg, book_2.png, chair_3.jpeg ...  so the
         correct answer can be read from the file name)
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ALLOWED_EXTENSIONS = (".jpg", ".jpeg", ".png")

# --------------------------------------------------------------------------
# KNOWLEDGE BASE
# "imagenet_labels" links the classifier's output labels to our objects.
# --------------------------------------------------------------------------
KNOWLEDGE_BASE = {
    "laptop": {
        "category": "Electronic device",
        "properties": ["electronic", "portable", "has screen", "has keyboard"],
        "function": "Used for computing, browsing, programming and online classes.",
        "description": "A portable personal computer with a screen and keyboard in one folding unit.",
        "imagenet_labels": ["laptop", "notebook"],
    },
    "book": {
        "category": "Stationery / Learning material",
        "properties": ["made of paper", "portable", "has pages", "non-electronic"],
        "function": "Used for reading, studying and storing written knowledge.",
        "description": "A set of printed pages bound together inside a cover.",
        "imagenet_labels": ["book jacket", "comic book", "binder"],
    },
    "chair": {
        "category": "Furniture",
        "properties": ["has legs", "has seat", "has backrest", "non-electronic"],
        "function": "Used for sitting.",
        "description": "A piece of furniture with a seat, legs and usually a backrest for one person.",
        "imagenet_labels": ["folding chair", "rocking chair", "barber chair", "throne", "park bench"],
    },
    "bottle": {
        "category": "Container",
        "properties": ["hollow", "has narrow neck", "holds liquid", "non-electronic"],
        "function": "Used for storing and carrying liquids such as water.",
        "description": "A container with a narrow neck and mouth, usually made of plastic or glass.",
        "imagenet_labels": ["water bottle", "pop bottle", "beer bottle", "wine bottle",
                            "water jug", "pill bottle", "whiskey jug"],
    },
    "mobile phone": {
        "category": "Electronic device",
        "properties": ["electronic", "portable", "has touch screen", "wireless"],
        "function": "Used for calling, messaging, internet access and photography.",
        "description": "A handheld wireless device for communication and many other apps.",
        "imagenet_labels": ["cellular telephone", "iPod", "hand-held computer"],
    },
    "table": {
        "category": "Furniture",
        "properties": ["has flat top", "has legs", "non-electronic"],
        "function": "Used as a flat surface for writing, working, eating or keeping things.",
        "description": "A piece of furniture with a flat top supported by legs.",
        "imagenet_labels": ["desk", "dining table", "coffee table", "folding table"],
    },
}

# Rule thresholds
HIGH_CONFIDENCE = 0.30   # R1: identified confidently
LOW_CONFIDENCE = 0.10    # R2: possible match, but not sure


# --------------------------------------------------------------------------
# IMAGE ANALYSIS
# --------------------------------------------------------------------------
_model_cache = {}


def classify_image(path, top_k=10):
    """Return a list of (imagenet_label, probability) using a pretrained CNN."""
    import torch
    from torchvision import models

    if "model" not in _model_cache:
        weights = models.MobileNet_V3_Large_Weights.DEFAULT
        _model_cache["model"] = models.mobilenet_v3_large(weights=weights).eval()
        _model_cache["transform"] = weights.transforms()
        _model_cache["labels"] = weights.meta["categories"]

    image = Image.open(path).convert("RGB")
    batch = _model_cache["transform"](image).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(_model_cache["model"](batch)[0], dim=0)
    values, indices = probs.topk(top_k)
    return [(_model_cache["labels"][i], float(v)) for v, i in zip(values, indices)]


def basic_visual_features(path):
    """Simple visual information: size, average colour and brightness."""
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=float)
    r, g, b = arr[..., 0].mean(), arr[..., 1].mean(), arr[..., 2].mean()
    return {
        "size": f"{img.width} x {img.height} px",
        "average colour (R,G,B)": f"({r:.0f}, {g:.0f}, {b:.0f})",
        "brightness": f"{arr.mean():.0f} / 255",
    }


# --------------------------------------------------------------------------
# RULE-BASED REASONING
# --------------------------------------------------------------------------
def reason_over_predictions(predictions):
    """
    R0: add up the probability of all classifier labels that belong to the same
        knowledge-base object.
    R1: IF best score >= 0.30 THEN object is identified confidently.
    R2: IF 0.10 <= best score < 0.30 THEN object is only a possible match.
    R3: ELSE the object cannot be identified.
    Returns (object_name or None, score, status).
    """
    scores = {name: 0.0 for name in KNOWLEDGE_BASE}
    for label, prob in predictions:
        for name, info in KNOWLEDGE_BASE.items():
            if label.lower() in [l.lower() for l in info["imagenet_labels"]]:
                scores[name] += prob

    best = max(scores, key=scores.get)
    best_score = scores[best]

    if best_score >= HIGH_CONFIDENCE:
        return best, best_score, "confident"
    if best_score >= LOW_CONFIDENCE:
        return best, best_score, "uncertain"
    return None, best_score, "unknown"


def derived_facts(name):
    """Extra IF-THEN rules applied to the knowledge-base properties."""
    info = KNOWLEDGE_BASE[name]
    facts = []
    if "electronic" in info["properties"]:
        facts.append("RULE: electronic object -> needs a power source / battery")
    else:
        facts.append("RULE: non-electronic object -> needs no power source")
    if info["category"] == "Furniture":
        facts.append("RULE: furniture -> normally kept fixed in a classroom, not carried around")
    if "portable" in info["properties"]:
        facts.append("RULE: portable object -> can be carried in a bag")
    return facts


# --------------------------------------------------------------------------
# DISPLAY
# --------------------------------------------------------------------------
def identify(path, show=True):
    """Run the whole pipeline on one image. Returns the identified object name or None."""
    if not path.lower().endswith(ALLOWED_EXTENSIONS):
        print("Error: only JPG, JPEG or PNG images are supported.")
        return None
    if not os.path.isfile(path):
        print(f"Error: file '{path}' not found.")
        return None

    predictions = classify_image(path)
    name, score, status = reason_over_predictions(predictions)

    print("\n" + "=" * 55)
    print("Image:", path)
    print("Visual features:")
    for k, v in basic_visual_features(path).items():
        print(f"  {k}: {v}")
    print("Top classifier predictions:")
    for label, p in predictions[:3]:
        print(f"  {label}: {p * 100:.1f}%")

    if status == "unknown":
        title = "Object cannot be identified confidently"
        print("\nResult: the object could not be identified confidently.")
        print("Please try a clearer, closer image of one object (laptop, book, chair, bottle, mobile phone or table).")
    else:
        info = KNOWLEDGE_BASE[name]
        if status == "uncertain":
            print(f"\nWarning: low confidence ({score * 100:.1f}%). This may be a {name}, but please verify.")
        print(f"\nObject      : {name.title()}  (confidence {score * 100:.1f}%)")
        print(f"Category    : {info['category']}")
        print(f"Properties  : {', '.join(info['properties'])}")
        print(f"Function    : {info['function']}")
        print(f"Description : {info['description']}")
        for fact in derived_facts(name):
            print("  ", fact)
        title = f"{name.title()} ({score * 100:.0f}%)" + (" - low confidence" if status == "uncertain" else "")

    if show:
        plt.figure(figsize=(5, 5))
        plt.imshow(Image.open(path).convert("RGB"))
        plt.title(title)
        plt.axis("off")
        plt.show()

    return name if status != "unknown" else None


# --------------------------------------------------------------------------
# TESTING / EVALUATION  (task h)
# --------------------------------------------------------------------------
def evaluate_folder(folder):
    files = sorted(f for f in os.listdir(folder) if f.lower().endswith(ALLOWED_EXTENSIONS))
    if not files:
        print("No JPG/PNG images found in", folder)
        return
    correct = 0
    rows = []
    for f in files:
        expected = os.path.splitext(f)[0].split("_")[0].replace("-", " ").lower()
        got = identify(os.path.join(folder, f), show=False)
        ok = got == expected
        correct += ok
        rows.append((f, expected, got or "unidentified", "PASS" if ok else "FAIL"))

    print("\n" + "=" * 55)
    print(f"{'File':25}{'Expected':15}{'Predicted':15}{'Result'}")
    for r in rows:
        print(f"{r[0]:25}{r[1]:15}{r[2]:15}{r[3]}")
    print(f"\nAccuracy: {correct}/{len(files)} = {correct / len(files) * 100:.1f}%")


def main():
    args = sys.argv[1:]
    if len(args) == 2 and args[0] == "--folder":
        evaluate_folder(args[1])
    elif len(args) == 1:
        identify(args[0])
    else:
        while True:
            path = input("\nEnter image path (or 'q' to quit): ").strip().strip('"')
            if path.lower() == "q":
                break
            identify(path)


if __name__ == "__main__":
    main()
