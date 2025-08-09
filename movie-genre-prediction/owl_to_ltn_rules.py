"""
Parse a movie ontology (OWL/RDF) and auto-generate LTN-compatible implication and exclusion rules.

Outputs:
  - implication_rules: list of (antecedent_label, consequent_label)
  - exclusion_rules: list of (label_a, label_b)  # mutual exclusion / disjoint

Also provides helpers to convert label-name pairs to index pairs given a LABEL_TO_IDX dict
and to save rules to disk as JSON.

Dependencies:
  pip install rdflib

Usage:
  python owl_to_ltn_rules.py /path/to/movie_ontology.owl --out rules.json

The script is conservative: it extracts rdfs:subClassOf as implications (subclass -> superclass)
and owl:disjointWith as exclusions (mutually exclusive). It also expands relations to include
named subclasses (transitively) to produce more useful pairs.

Note: Ontologies vary a lot; you may need to adapt label normalization and heuristics to your dataset's labels.
"""

import sys
import json
import argparse
from rdflib import Graph, RDFS, OWL, URIRef, RDF, Literal
from collections import defaultdict, deque
from typing import List, Tuple, Dict, Set


def qname_local(uri: URIRef) -> str:
    """Return a human-friendly local name for a URIRef.
    Prefer rdfs:label if available; otherwise use the last fragment of the URI.
    """
    s = str(uri)
    if '#' in s:
        return s.split('#')[-1]
    elif '/' in s:
        return s.rstrip('/').split('/')[-1]
    return s


def get_label(g: Graph, node: URIRef) -> str:
    # prefer rdfs:label (literal), otherwise use local qname
    for lbl in g.objects(node, RDFS.label):
        if isinstance(lbl, Literal):
            return str(lbl)
    return qname_local(node)


def collect_named_classes(g: Graph) -> Set[URIRef]:
    """Collect URIRefs that are rdfs:Class or owl:Class or appear as subjects/objects in subclass/disjoint triples."""
    classes = set()
    for c in g.subjects(RDF.type, OWL.Class):
        classes.add(c)
    for c in g.subjects(RDF.type, RDFS.Class):
        classes.add(c)

    # Also include classes that appear in subclass/disjoint relations
    for s, p, o in g.triples((None, RDFS.subClassOf, None)):
        classes.add(s)
        classes.add(o)
    for s, p, o in g.triples((None, OWL.disjointWith, None)):
        classes.add(s)
        classes.add(o)
    return classes


def build_subclass_graph(g: Graph) -> Dict[URIRef, Set[URIRef]]:
    """Return mapping class -> set(direct_superclasses)"""
    subs = defaultdict(set)
    for s, p, o in g.triples((None, RDFS.subClassOf, None)):
        subs[s].add(o)
    return subs


def invert_graph(subs: Dict[URIRef, Set[URIRef]]) -> Dict[URIRef, Set[URIRef]]:
    parents = defaultdict(set)
    for child, sup_set in subs.items():
        for sup in sup_set:
            parents[sup].add(child)
    return parents


def transitive_closure_parents(parents: Dict[URIRef, Set[URIRef]]) -> Dict[URIRef, Set[URIRef]]:
    """Given parent->children mapping, compute transitive closure children of each node (all descendants).
    Returns: node -> set(descendants)
    """
    closure = {}
    for node in parents:
        # BFS/DFS to collect all descendants
        seen = set()
        dq = deque([node])
        while dq:
            cur = dq.popleft()
            for child in parents.get(cur, []):
                if child not in seen:
                    seen.add(child)
                    dq.append(child)
        # remove the node itself if present
        if node in seen:
            seen.remove(node)
        closure[node] = seen
    return closure


def extract_rules_from_owl(path: str):
    g = Graph()
    g.parse(path, format=None)  # let rdflib guess format (xml, ttl, etc.)

    classes = collect_named_classes(g)
    subs = build_subclass_graph(g)
    parents = invert_graph(subs)  # parent -> direct children
    descendants = transitive_closure_parents(parents)  # parent -> all descendants

    # Build human-friendly labels mapping
    uri_to_label = {uri: get_label(g, uri) for uri in classes}

    # --- Implication rules from subclass relationships ---
    # For every subclass A subClassOf B, we create implication (A -> B).
    implication_rules = set()
    for child, sup_set in subs.items():
        for sup in sup_set:
            if child in uri_to_label and sup in uri_to_label:
                a_label = uri_to_label[child]
                c_label = uri_to_label[sup]
                implication_rules.add((a_label, c_label))
            # Also incorporate descendants: any descendant of child implies sup as well
            if child in descendants:
                for desc in descendants[child]:
                    if desc in uri_to_label and sup in uri_to_label:
                        implication_rules.add((uri_to_label[desc], uri_to_label[sup]))

    # --- Exclusion rules from owl:disjointWith ---
    exclusion_rules = set()
    for s, p, o in g.triples((None, OWL.disjointWith, None)):
        if s in uri_to_label and o in uri_to_label:
            exclusion_rules.add((uri_to_label[s], uri_to_label[o]))
        # also expand: if s has descendants, mark descendants disjoint with o and vice versa
        if s in descendants:
            for desc in descendants[s]:
                if desc in uri_to_label and o in uri_to_label:
                    exclusion_rules.add((uri_to_label[desc], uri_to_label[o]))
        if o in descendants:
            for desc in descendants[o]:
                if desc in uri_to_label and s in uri_to_label:
                    exclusion_rules.add((uri_to_label[s], uri_to_label[desc]))

    # Normalize exclusion pairs to canonical order (tuple(sorted)) and remove reflexive
    normalized_exclusions = set()
    for a, b in exclusion_rules:
        if a == b:
            continue
        # keep as (a,b) and (b,a) both? For mutual exclusion we can canonicalize to sorted
        normalized_exclusions.add(tuple(sorted((a, b))))

    # Convert back to consistent ordering (first, second)
    exclusion_final = set()
    for a, b in normalized_exclusions:
        exclusion_final.add((a, b))

    # Sort and return lists
    implication_list = sorted(list(implication_rules))
    exclusion_list = sorted(list(exclusion_final))

    return {
        'implication_rules': implication_list,
        'exclusion_rules': exclusion_list,
        'uri_to_label': {str(k): v for k, v in uri_to_label.items()}
    }


def labels_to_index_pairs(pairs: List[Tuple[str, str]], label_to_idx: Dict[str, int], clean_fn=None):
    """Convert list of (labelA, labelB) to index pairs using label_to_idx mapping.
    clean_fn: optional function(label) -> cleaned_label used to match keys in label_to_idx.
    Returns two lists: valid_index_pairs, missing_labels (set)
    """
    valid = []
    missing = set()
    for a, b in pairs:
        a_k = clean_fn(a) if clean_fn else a
        b_k = clean_fn(b) if clean_fn else b
        if a_k in label_to_idx and b_k in label_to_idx:
            valid.append((label_to_idx[a_k], label_to_idx[b_k]))
        else:
            if a_k not in label_to_idx:
                missing.add(a_k)
            if b_k not in label_to_idx:
                missing.add(b_k)
    return valid, missing


def simple_clean(label: str) -> str:
    return label.strip().lower().replace(' ', '-').replace('_', '-')


def main():
    parser = argparse.ArgumentParser(description='Extract LTN rules from OWL ontology')
    parser.add_argument('owl_path', help='Path to ontology file (OWL/RDF/Turtle)')
    parser.add_argument('--out', '-o', help='Output JSON file for rules', default='ltn_rules.json')
    args = parser.parse_args()

    res = extract_rules_from_owl(args.owl_path)

    # Save to JSON
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(res['implication_rules'])} implication rules and {len(res['exclusion_rules'])} exclusion rules to {args.out}")
    print("Sample implication rules:")
    for r in res['implication_rules'][:20]:
        print(r)
    print("Sample exclusion rules:")
    for r in res['exclusion_rules'][:20]:
        print(r)


if __name__ == '__main__':
    main()
