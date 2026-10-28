interest = input("Enter your interest (AI, Data, Web): ").strip().lower()

if interest == "ai":
    course = "Artificial Intelligence"
elif interest == "data":
    course = "Data Science"
elif interest == "web":
    course = "Web Development"
else:
    course = "No matching course found"

print("Recommended Course:", course)
