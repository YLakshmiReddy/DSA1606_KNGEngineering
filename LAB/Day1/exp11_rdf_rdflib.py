# Requires: pip install rdflib
from rdflib import Graph, Namespace, Literal

g = Graph()
EX = Namespace("http://example.org/")

g.add((EX.Arun, EX.studies, EX.Python))
g.add((EX.Python, EX.type, Literal("Programming Language")))

print("RDF Knowledge Triples:")
for subject, predicate, obj in g:
    print(subject, predicate, obj)
