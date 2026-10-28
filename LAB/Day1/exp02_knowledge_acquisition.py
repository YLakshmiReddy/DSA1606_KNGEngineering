name = input("Enter expert name: ")
domain = input("Enter area of expertise: ")
experience = int(input("Enter years of experience: "))

knowledge = {
    "expert": name,
    "domain": domain,
    "experience": experience
}

print("\nAcquired Knowledge")
print("Expert:", knowledge["expert"])
print("Domain:", knowledge["domain"])
print("Experience:", knowledge["experience"], "years")
