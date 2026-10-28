knowledge_base = {
    "python": "Python is a high-level programming language.",
    "ontology": "An ontology formally defines concepts and relationships.",
    "knowledge graph": "A knowledge graph connects entities through relationships.",
    "expert system": "An expert system uses rules and knowledge to solve problems."
}

print("Knowledge Management System")
print("Available topics:", ", ".join(knowledge_base.keys()))
query = input("\nEnter a topic to search: ").strip().lower()

if query in knowledge_base:
    print("Knowledge Found:")
    print(knowledge_base[query])
else:
    print("No information found for this topic.")
