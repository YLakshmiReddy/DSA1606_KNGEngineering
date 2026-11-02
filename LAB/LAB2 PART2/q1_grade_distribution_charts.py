"""
Q1: Grade distribution of 40 students - pie chart and bar chart.
Requires: pip install matplotlib
"""
import matplotlib.pyplot as plt

# 1. Store grades and number of students (F = lowest, S = highest)
grades = ["F", "E", "D", "C", "B", "A", "S"]
students = [3, 4, 5, 8, 9, 7, 4]
total = sum(students)

assert total == 40, "Total number of students must be 40"

colors = ["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71", "#1abc9c", "#3498db", "#9b59b6"]

# 2 & 3. Pie chart with grade labels and percentage values
plt.figure(figsize=(7, 7))
plt.pie(
    students,
    labels=grades,
    autopct="%1.1f%%",
    colors=colors,
    startangle=90,
    counterclock=False,
)
plt.title("Percentage Distribution of Students by Grade")
plt.legend(grades, title="Grades", loc="center left", bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.savefig("q1_pie_chart.png", dpi=150)
plt.show()

# 4, 5, 6, 7. Bar chart with axis labels and counts above bars
plt.figure(figsize=(8, 6))
bars = plt.bar(grades, students, color=colors, label="Number of students")
plt.xlabel("Grades")
plt.ylabel("Number of Students")
plt.title("Number of Students Securing Each Grade")
plt.ylim(0, max(students) + 2)

for bar, count in zip(bars, students):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.1,
        str(count),
        ha="center",
        va="bottom",
        fontweight="bold",
    )

plt.legend()
plt.tight_layout()
plt.savefig("q1_bar_chart.png", dpi=150)
plt.show()

# 8. Grades with maximum and minimum number of students
max_count = max(students)
min_count = min(students)
max_grades = [g for g, n in zip(grades, students) if n == max_count]
min_grades = [g for g, n in zip(grades, students) if n == min_count]

print(f"Grade with maximum students: {', '.join(max_grades)} ({max_count} students)")
print(f"Grade with minimum students: {', '.join(min_grades)} ({min_count} students)")

# 9. Percentage of students with grades A and S combined
a_and_s = students[grades.index("A")] + students[grades.index("S")]
print(f"Students with grades A and S combined: {a_and_s} out of {total}")
print(f"Percentage of A and S combined: {a_and_s / total * 100:.2f}%")
