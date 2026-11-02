"""
Q2: Read student grades from a CSV file using pandas, validate, analyse and plot.
Requires: pip install pandas matplotlib
"""
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

CSV_FILE = "student_grades.csv"
EXPECTED_GRADES = ["F", "E", "D", "C", "B", "A", "S"]
EXPECTED_TOTAL = 40

# 1 & 2. Create the CSV file (only if it does not already exist)
if not os.path.exists(CSV_FILE):
    pd.DataFrame(
        {
            "Grade": EXPECTED_GRADES,
            "Number_of_Students": [3, 4, 5, 8, 9, 7, 4],
        }
    ).to_csv(CSV_FILE, index=False)
    print(f"Created {CSV_FILE}")

# 3. Read the CSV file using pandas
df = pd.read_csv(CSV_FILE)

# 4. Validate the data
df["Grade"] = df["Grade"].astype(str).str.strip().str.upper()

missing = [g for g in EXPECTED_GRADES if g not in df["Grade"].values]
if missing:
    sys.exit(f"Validation failed: missing grades {missing}")

if df["Grade"].duplicated().any():
    sys.exit("Validation failed: duplicate grades found")

if not pd.api.types.is_numeric_dtype(df["Number_of_Students"]) or (df["Number_of_Students"] < 0).any():
    sys.exit("Validation failed: Number_of_Students must be non-negative numbers")

total = int(df["Number_of_Students"].sum())
if total != EXPECTED_TOTAL:
    sys.exit(f"Validation failed: total is {total}, expected {EXPECTED_TOTAL}")

print("Validation successful: all 7 grades present and total = 40\n")

# Order the rows from F (lowest) to S (highest)
df["Grade"] = pd.Categorical(df["Grade"], categories=EXPECTED_GRADES, ordered=True)
df = df.sort_values("Grade").reset_index(drop=True)

# 5. Calculate percentages
df["Percentage"] = (df["Number_of_Students"] / total * 100).round(2)

# 9. Display number and percentage for each grade
print("Grade-wise Summary")
print(df.to_string(index=False))

# 10. Highest and lowest
highest = df.loc[df["Number_of_Students"].idxmax()]
lowest = df.loc[df["Number_of_Students"].idxmin()]
print(f"\nHighest: Grade {highest['Grade']} with {int(highest['Number_of_Students'])} students")
print(f"Lowest : Grade {lowest['Grade']} with {int(lowest['Number_of_Students'])} students")

colors = ["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71", "#1abc9c", "#3498db", "#9b59b6"]

# 6 & 8. Pie chart
plt.figure(figsize=(7, 7))
plt.pie(
    df["Number_of_Students"],
    labels=df["Grade"],
    autopct="%1.1f%%",
    colors=colors,
    startangle=90,
    counterclock=False,
)
plt.title("Percentage Distribution of Students by Grade")
plt.legend(df["Grade"], title="Grades", loc="center left", bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.savefig("grade_pie_chart.png", dpi=150)   # 11. Save as image
plt.show()

# 7 & 8. Bar chart
plt.figure(figsize=(8, 6))
bars = plt.bar(df["Grade"].astype(str), df["Number_of_Students"], color=colors, label="Number of students")
plt.xlabel("Grades")
plt.ylabel("Number of Students")
plt.title("Number of Students Securing Each Grade")
plt.ylim(0, df["Number_of_Students"].max() + 2)
for bar, count in zip(bars, df["Number_of_Students"]):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
             str(int(count)), ha="center", va="bottom", fontweight="bold")
plt.legend()
plt.tight_layout()
plt.savefig("grade_bar_chart.png", dpi=150)   # 11. Save as image
plt.show()

print("\nCharts saved as grade_pie_chart.png and grade_bar_chart.png")
