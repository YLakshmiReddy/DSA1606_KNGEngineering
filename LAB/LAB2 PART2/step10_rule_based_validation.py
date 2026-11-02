"""
Step 10 - Rule-Based Validation: expert-system rules check the OCR text
against the registration format, allowing for common OCR errors.
Run:  python step10_rule_based_validation.py      (no image needed)
"""
import re


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


def show(text):
    ok, final, messages = validate_plate(text)
    print(f"\nInput text : {text or '(empty)'}")
    print(f"Final text : {final}")
    print("Result     :", "VALID" if ok else "INVALID")
    for m in messages:
        print("  -", m)


def main():
    samples = [
        "TN09AB1234",   # correct
        "TNO9AB1234",   # O read instead of 0   -> corrected
        "TN09A81234",   # 8 read instead of B   -> corrected
        "XX09AB1234",   # invalid state code
        "TN09AB12",     # too short
        "",             # nothing read
    ]
    print("Testing sample OCR outputs")
    for s in samples:
        show(s)

    print("\n--- Try your own ---")
    while True:
        text = input("\nEnter OCR text to validate (or 'q' to quit): ").strip().upper()
        if text == "Q":
            break
        show(re.sub(r"[^A-Z0-9]", "", text))


if __name__ == "__main__":
    main()
