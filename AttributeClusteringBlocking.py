import numpy as np
import networkx as nx
from Blocking import TokenBlocker

def _tokenizeColumns(EC, transformationFun, column_index, token_name_prefix):
    tokens = {}
    for i in column_index:
        tokens[str(i)+token_name_prefix] = transformationFun(EC, i)
    return(tokens)

def _flattenTokenDict(token_dict):
    flat_dict = {}
    for k in token_dict:
        flat_dict[k] = set([token for token_list in token_dict[k] for token in token_list])
    return(flat_dict)

def JaccardSimilarity(set1, set2):
    return(len(set.intersection(set1, set2))/len(set.union(set1, set2)))

def _createAdjacencyMatrix(link_dict):
    n = len(link_dict)
    a_mat = [[0 for i in range(0,n)] for j in range(0,n)]
    ind = link_dict.keys()
    for i,k in enumerate(link_dict):
        row = a_mat[i]
        row[ind.index(link_dict[k])] = 1
    return(a_mat)

def _linkAttributes(tokens1, tokens2, similarityFun):
    flat_dict_tokens1 = _flattenTokenDict(tokens1)
    flat_dict_tokens2 = _flattenTokenDict(tokens2)
    glue = []
    links = {}
    for k in flat_dict_tokens1:
        highest = 0
        linkTo = -1
        for k2 in flat_dict_tokens2:
            sim = similarityFun(flat_dict_tokens1[k], flat_dict_tokens2[k2])
            if(sim > highest):
                highest = sim
                linkTo = k2
        if(highest>0):
            links[k] = [linkTo]
        else:
            glue.append(k)
    for k in flat_dict_tokens2:
        highest = 0
        linkTo = -1
        for k2 in flat_dict_tokens1:
            sim = similarityFun(flat_dict_tokens2[k], flat_dict_tokens1[k2])
            if(sim > highest):
                highest = sim
                linkTo = k2
        if(highest>0):
            links[k] = [linkTo]
        else:
            glue.append(k)  
    clusters = []
    g = nx.Graph(links)
    g = g.to_undirected()
    sub_graphs = nx.connected_component_subgraphs(g)
    clusters = [sg.nodes() for sg in sub_graphs]
    return(clusters)

def mergeTokenLists(list1, list2):

    if(len(list1) != len(list2)):
        raise ValueError("List of tokens were not same size")
    else:
        concatenated = list1
        for i in range(0, len(list1)):
            concatenated[i] = concatenated[i].extend(list2[i])
    

def ClusterBlocker(EC, clusters):
    clusterBlocks = {}
    for i, cluster in enumerate(clusters):
        concatenated = []
        initialized = False
        for column in cluster:
            col = EC.get(column)
            if(col != None):
                if(initialized):
                    concatenated = mergeTokenLists(concatenated, col)
                else:
                    concatenated = col
                    initialized = True
        
        clusterBlocks[str(i)] = TokenBlocker(concatenated)
    return(clusterBlocks)


def _joinClusterBlocks(BC1, BC2):
    combined = {}
    for clusterKey in BC1:
        if(clusterKey in BC2):
            for blockKey in BC1[clusterKey]:
                if blockKey in BC2[clusterKey]:
                    combined[clusterKey+blockKey] = (BC1[clusterKey][blockKey],  BC2[clusterKey][blockKey])
    return(combined)
    


def AttributeClusteringBlocking(EntityCollection1, EntityCollection2, transformationFun, similarityFun, column_index=(1,2,3)):
    tokensEC1 = _tokenizeColumns(EntityCollection1, transformationFun, column_index, "_1")
    tokensEC2 = _tokenizeColumns(EntityCollection2, transformationFun, column_index, "_2")
    clusters = _linkAttributes(tokensEC1, tokensEC2, similarityFun)
    clusterBlocks1 = ClusterBlocker(tokensEC1, clusters)
    clusterBlocks2 = ClusterBlocker(tokensEC2, clusters)
    blockCollection = _joinClusterBlocks(clusterBlocks1, clusterBlocks2)
    return(blockCollection)