# Requires: pip install networkx
import networkx as nx

knowledge_graph = nx.DiGraph()
knowledge_graph.add_edge("Arun", "Python", relation="studies")
knowledge_graph.add_edge("Arun", "Computer Science", relation="belongs_to")
knowledge_graph.add_edge("Python", "Programming Language", relation="is_a")

print("Knowledge Graph Triples:")
for subject, obj, data in knowledge_graph.edges(data=True):
    print(subject, "--", data["relation"], "-->", obj)
