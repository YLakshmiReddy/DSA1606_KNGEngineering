"""
Step 9 - Knowledge Base Construction: registration format, character
patterns and validation rules for Indian number plates.
Run:  python step09_knowledge_base.py      (no image needed)
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


def main():
    kb = KNOWLEDGE_BASE
    print("KNOWLEDGE BASE - Indian vehicle registration number")
    print("Format        :", kb["format_description"])
    print("Example       : TN09AB1234  (TN = Tamil Nadu, 09 = district, AB = series, 1234 = number)")
    print("Regex pattern :", kb["pattern"].pattern)
    print(f"State codes   : {len(kb['state_codes'])} -> {', '.join(sorted(kb['state_codes']))}")
    print("\nCharacter patterns (position-wise):")
    print("  Positions 1-2  : letters (state/UT code)")
    print("  Positions 3-4  : digits  (district number)")
    print("  Next 1-3 chars : letters (series)")
    print("  Last 4 chars   : digits  (vehicle number)")
    print("\nCommon OCR confusions:")
    print("  Letter read where a digit is expected :", kb["to_digit"])
    print("  Digit read where a letter is expected :", kb["to_letter"])
    print("\nValidation rules:")
    print("  Rule 1: text must not be empty")
    print("  Rule 2: length must be 9 to 11 characters")
    print("  Rule 3: text must match the pattern after OCR corrections")
    print("  Rule 4: first two letters must be a valid state/UT code")


if __name__ == "__main__":
    main()
