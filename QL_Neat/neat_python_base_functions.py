import math

def sigmoid(x):
    """Return the S-Curve activation of x."""
    return 1/(1+math.exp(-x))

def tanh(x):
    """Wrapper function for hyperbolic tangent activation."""
    return math.tanh(x)

def LReLU(x):
    """Leaky ReLU function for x"""
    if x >= 0:
        return x
    else:
        return 0.01 * x

def genomic_distance(a, b, distance_weights):
    """Calculate the genomic distance between two genomes."""
    a_edges = set(a._edges)
    b_edges = set(b._edges)

    # Does not distinguish between disjoint and excess
    matching_edges = a_edges & b_edges
    disjoint_edges = (a_edges - b_edges) | (b_edges - a_edges)
    N_edges = len(max(a_edges, b_edges, key=len))
    N_nodes = min(a._max_node, b._max_node)

    weight_diff = 0
    for i in matching_edges:
        weight_diff += abs(a._edges[i].weight - b._edges[i].weight)

    bias_diff = 0
    for i in range(N_nodes):
        bias_diff += abs(a._nodes[i].bias - b._nodes[i].bias)

    t1 = distance_weights['edge'] * len(disjoint_edges)/N_edges
    t2 = distance_weights['weight'] * weight_diff/len(matching_edges)
    t3 = distance_weights['bias'] * bias_diff/N_nodes
    return t1 + t2 + t3


