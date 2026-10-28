# Requires: pip install networkx
import networkx as nx

G = nx.Graph()
G.add_edges_from([
    ("Arun", "Python"),
    ("Python", "Programming"),
    ("Programming", "Computer Science"),
    ("Arun", "College")
])

start = input("Enter starting entity: ").strip()
target = input("Enter target entity: ").strip()

try:
    path = nx.shortest_path(G, source=start, target=target)
    print("Connection Path:", " -> ".join(path))
except (nx.NetworkXNoPath, nx.NodeNotFound):
    print("No path found between the given entities.")
