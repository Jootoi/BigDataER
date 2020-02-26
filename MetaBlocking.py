import networkx as nx
import numpy as np
from statistics import mean
from math import ceil
import heapq

# entity1 is from first entity collection, entity2 is from the second.
# Adds only non-duplicate nodes and edges, keep count of both to help with weighting.
# Only pairs entities between collections, not within itself.
# Returns dict-in-dict, where 'nodes' consist of entity represented by node (key) and its count (value),
# and 'edges' consist of tuples (i, j) (key) and its count (value).
# Adds EC1maxIndex + 1 to EC2 entities, because indices are not unique between ECs.

maxIndex = 0
def GraphBuilder(blockCollection):
    nodes = {}
    edges = {}
    global maxIndex
    for block in blockCollection:
        # entity1 is from first entity collection, entity2 is from the second
        # Add only non-duplicates, keep count of the nodes and edges
        # Only pairs entities between collections, not within itself
        for entity1 in blockCollection[block][0]:
            if entity1 not in nodes:
                nodes[entity1] = 1
            else:
                nodes[entity1] += 1
            if entity1 > maxIndex:
                maxIndex = entity1
                    
    for block in blockCollection:  
        for entity2 in blockCollection[block][1]:
            entity2 = entity2 + maxIndex + 1
            if entity2 not in nodes:
                nodes[entity2] = 1
            else:
                nodes[entity2] += 1
            
            for entity1 in blockCollection[block][0]:
                if (entity1, entity2) not in edges:
                    edges[(entity1, entity2)] = 1
                else:
                    edges[(entity1, entity2)] += 1
    
    return {'nodes': nodes, 'edges': edges}

# Adds Jaccard weight info. Edges are tuples (i, j), and work as dictionary keys, their
# value is the weight of the edge.
def JaccardWeighting(nodesAndEdges):
    nodes = nodesAndEdges['nodes']
    edges = nodesAndEdges['edges']
    jaccardWeights = {}
    for pair in edges:
        jaccardWeights[pair] = edges[pair] / (nodes[pair[0]] + nodes[pair[1]] - edges[pair])
    return {'nodes': nodes, 'edges': jaccardWeights}

# Adds common blocks scheme weight info (normalized). Edges are tuples (i, j), and work as dictionary keys, their
# value is the weight of the edge.
def CBSWeighting(nodesAndEdges):
    edges = nodesAndEdges['edges']
    max_value = max(edges.values())
    # normalize the values between [0, 1]
    for edge in edges:
        edges[edge] = edges[edge] / max_value
    return {'nodes': nodesAndEdges['nodes'], 'edges': edges}

# Prunes the edges, of which weights are below global average
def WeightEdgePruning(nodesAndEdges):
    edges = nodesAndEdges['edges']
    avgEdgeWeight = mean(edges[k] for k in edges)
    remainingEdges = {}
    for edge in edges:
        if edges[edge] >= avgEdgeWeight:
            remainingEdges[edge] = edges[edge]
    return [*remainingEdges]

# From each node's neighborhood, prunes the edges that are below the local top 10 % (k-value) based on their weights.
# Rounds up, so minimum is always 1. Edges are represented by tuples (i, j); returns the remaining edges in a list of tuples.
def CardinalityNodePruning(nodesAndEdges):
    nodes = nodesAndEdges['nodes']
    edges = nodesAndEdges['edges']
    remainingEdges = {}
    graph = nx.Graph()
    for edge in edges:
        graph.add_edge(edge[0], edge[1], weight=edges[edge])
    for node in nodes:
        stack = []
        heapq.heapify(stack)
        neighborhood = graph[node]
        k = ceil((len(neighborhood) * 0.1))
        for edge in neighborhood:
            nodeAndWeight = (edge, neighborhood[edge]['weight'])
            stack.append(nodeAndWeight)
        stack.sort(key=lambda tup: tup[1], reverse=True)        
        while(len(stack)>k):
            stack.pop()

        for neighbor in stack:
            remainingEdges[(node, neighbor[0])] = 0
    return [*remainingEdges]

# Makes a new block collection for possible next step
# Each block is partitioned into two: first part is the entity, and second part is everything still connected to it
def BlockCollecting(remainingEdges):
    blocks = {}
    for edge in remainingEdges:
        if edge[0] not in blocks:
            blocks[edge[0]] = [edge[1]]
        else:
            blocks[edge[0]].append(edge[1])
    return blocks

# These two are from Blocking.py, EvaluateMetaBlockCollection has been slightly modified
def _goldStandardToIndexArray(EntityCollection1, EntityCollection2, goldStandard):
    """Extracts all the required comparisons from the gold standard and maps them to indices in the original entity collections. 
    Returns the comparisons as list of tuples. """
    s = goldStandard.shape
    N = s[0]
    goldStandardIndices = np.ndarray(shape=s, dtype=np.int32)
    pk1 = EntityCollection1[:, 0]
    pk2 = EntityCollection2[:, 0]
    global maxIndex
    for i in np.arange(N):
        # Those are supposed to be primary keys so there should ever be only one match
        # maxIndex + 1 to prevent indices clashing
        goldStandardIndices[i, 0] =  np.where(pk1 == goldStandard[i, 0])[0]
        goldStandardIndices[i, 1] =  np.where(pk2 == goldStandard[i, 1])[0] + maxIndex + 1
    return(list(map(tuple, goldStandardIndices)))

def EvaluateMetaBlockCollection(EntityCollection1, EntityCollection2, comparisons, goldStandard):
    """Evaluate a block collection against gold standard. Calculates pair completenes, reduction ratio (vs. brute force)  and reduction ratio with redundancy pruning (vs. brute force)"""
    gsIndexArray = _goldStandardToIndexArray(EntityCollection1, EntityCollection2, goldStandard)
    unique_comparisons = set(comparisons)
    matches = [comparison for comparison in gsIndexArray if comparison in unique_comparisons]
    pc = len(matches)/len(gsIndexArray)*100
    rr1 = (1 - len(comparisons)/(len(EntityCollection1)*len(EntityCollection2)))*100
    print("Pair Completeness: {0:.2f}%\n Reduction Ratio: {1:.2f}%\n".format(pc, rr1))

