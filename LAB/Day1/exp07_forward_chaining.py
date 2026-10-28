facts = {"has_fever", "has_cough"}
rules = [
    ({"has_fever", "has_cough"}, "possible_flu"),
    ({"possible_flu"}, "recommend_medical_review")
]

changed = True
while changed:
    changed = False
    for conditions, conclusion in rules:
        if conditions.issubset(facts) and conclusion not in facts:
            facts.add(conclusion)
            print("Inferred:", conclusion)
            changed = True

print("\nFinal Facts:")
for fact in sorted(facts):
    print(fact)
