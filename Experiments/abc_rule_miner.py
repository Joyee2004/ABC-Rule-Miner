from collections import Counter
import math
class Node:
    def __init__(self, context=None, value=None, behavior=None, confidence=1.0):
        self.context = context
        self.value = value
        self.behavior = behavior
        self.confidence = confidence
        self.children = []
        self.node_type = "NORMAL"

    def add_child(self, child):
        self.children.append(child)


def entropy(ds, behavior_col):
    """Compute entropy of behavior distribution in dataset."""
    counts = Counter([row[behavior_col] for row in ds])
    total = len(ds)
    return -sum((c / total) * math.log2(c / total) for c in counts.values() if c > 0)


def information_gain(ds, context, behavior_col):
    """Compute information gain for a given context."""
    base_entropy = entropy(ds, behavior_col)
    values = {row[context] for row in ds}
    total = len(ds)
    weighted_entropy = 0.0
    for v in values:
        subset = [row for row in ds if row[context] == v]
        weighted_entropy += (len(subset) / total) * entropy(subset, behavior_col)
    return base_entropy - weighted_entropy


def calculate_dominant_behavior(ds, behavior_col):
    """Return dominant behavior and its confidence for the given dataset subset."""
    if not ds:
        return None, 0.0

    # Compute class frequencies and confidence
    behavior_counts = Counter([row[behavior_col] for row in ds])
    total = len(ds)
    confidences = {b: c / total for b, c in behavior_counts.items()}

    # Pick the most frequent (dominant) behavior
    behavior = max(confidences, key=lambda b: confidences[b])
    confidence = confidences[behavior]

    return behavior, confidence


def select_highest_precedence_context(ds, context_list, behavior_col):
    """Select context with the highest information gain."""
    gains = {ctx: information_gain(ds, ctx, behavior_col) for ctx in context_list}
    return max(gains, key=gains.get)



def AGT(DS, context_list, behavior_col, t=0.7):
    """
    Association Generation Tree (AGT) Algorithm.
    DS: dataset (list of dicts)
    context_list: context attribute names
    behavior_col: target column
    t: confidence threshold for pruning redundant nodes
    """
    root = Node()
    min_samples = 100000

    # if all instances  in DS have the same behavior
    # line 3 in algorithm
    behaviors = {row[behavior_col] for row in DS}
    if len(behaviors) == 1: #same behavior 
        root.behavior = list(behaviors)[0]
        root.confidence = 1.0
        return root

    # assign dominant behavior at root
    if root.behavior is None:
        bh, conf = calculate_dominant_behavior(DS, behavior_col)
        root.behavior = bh
        root.confidence = conf

    # stop if no more contexts or context list is empty
    if not context_list:
        bh, conf = calculate_dominant_behavior(DS, behavior_col)
        root.behavior = bh
        root.confidence = conf
        return root

    # Identify the highest precedence context (most informative)
    C_split = select_highest_precedence_context(DS, context_list, behavior_col)
    root.context = C_split

    for val in {row[C_split] for row in DS}:
        DS_sub = [row for row in DS if row[C_split] == val]
       
        if not DS_sub :
            continue

        dom_behavior, conf = calculate_dominant_behavior(DS_sub, behavior_col)
        child = Node(context=C_split, value=val, behavior=dom_behavior, confidence=conf)

        # --- 🟡 Redundancy pruning logic ---
        if root.behavior == dom_behavior and conf >= t  and  root.confidence >= t:
            child.node_type = "REDUNDANT"
            # ✅ Skip expanding this branch
            # root.add_child(child)
            # continue

        # Continue building subtree
        remaining = [c for c in context_list if c != C_split]
        subtree = AGT(DS_sub, remaining, behavior_col, t)
        for subchild in subtree.children:
            child.add_child(subchild)
        
        root.add_child(child)

    return root


