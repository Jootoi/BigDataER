import pandas
import numpy as np
import nltk
from nltk.stem import LancasterStemmer
nltk.download('stopwords')
nltk.download("punkt")
from nltk.corpus import stopwords
stop = stopwords.words('english')
lancaster = LancasterStemmer()


def titleTokenizer(EC, title_index=1):
    """Extract tokens from specified column of each entity
    Makes each token lower case, removes (english) stopwords and applies lancaster stemming"""
    titles = EC[:,title_index]
    stemmed = []
    for title in titles:
        tokens = nltk.word_tokenize(title.lower())
        filtered = [token for token in tokens if token not in stop and len(token)>1]
        stemmed.append([lancaster.stem(token) for token in filtered])
    return stemmed


def TokenBlocker(tokenArray):
    """ Makes a dictionary where each token is key and value is array of index values of entites having that token """
    token_dictionary = {}
    for index, tokens in enumerate(tokenArray):
        for token in tokens:
            a = token_dictionary.get(token, [])
            a.append(index)
            token_dictionary[token] = a
    return(token_dictionary)



def JoinBlocks(BC1, BC2):
    """ Joins block collections made from two different entity collections.
    Returns dictionary where tokens are keys and values are tuples containing lists of index values of entites having that token.
    First list of the tuple is the indices coming from entity collection 1 and second list comes from entity collection 2.
    If some token is not found from both entity collections it is not added. """
    combined = {}
    for key in BC1:
        if(key in BC2):
            combined[key] = (BC1[key], BC2[key])
    return(combined)

def TokenBlocking(EntityCollection1, EntityCollection2, transformationFun, constraintFun):
    """Glues together the different functions required to do token blocking.
    Input consists of two entity collections, a transformation function and a constrain function """
    transformedEC1 = transformationFun(EntityCollection1)
    transformedEC2 = transformationFun(EntityCollection2)

    blockedEC1 = constraintFun(transformedEC1)
    blockedEC2 = constraintFun(transformedEC2)

    blockCollection = JoinBlocks(blockedEC1, blockedEC2)
    return(blockCollection)


def _goldStandardToIndexArray(EntityCollection1, EntityCollection2, goldStandard):
    """Extracts all the required comparisons from the gold standard and maps them to indices in the original entity collections. 
    Returns the comparisons as list of tuples. """
    s = goldStandard.shape
    N = s[0]
    goldStandardIndices = np.ndarray(shape=s, dtype=np.int32)
    pk1 = EntityCollection1[:, 0]
    pk2 = EntityCollection2[:, 0]
    for i in np.arange(N):
        # Those are supposed to be primary keys so there should ever be only one match
        goldStandardIndices[i, 0] =  np.where(pk1 == goldStandard[i, 0])[0]
        goldStandardIndices[i, 1] =  np.where(pk2 == goldStandard[i, 1])[0]
    return(list(map(tuple, goldStandardIndices)))

def _reduceBlockToComparisons(block):
    """Given a block as tuple of two arrays of indices, constructs all the pairwise comparisons.
    Returns comparisons as list of tuples """
    comparisons = []
    part1 = block[0]
    part2 = block[1]
    for item in part1:
        for item2 in part2:
            comparisons.append((item, item2))
    return(comparisons)

def _reduceBlockCollectionToComparisons(bc):
    """Given a block collection, finds all pairwise comparisons and returns them as list of tuples """
    comparisons = [_reduceBlockToComparisons(block) for key, block in bc.items()]
    comparisons =  [item for sublist in comparisons for item in sublist]
    return(comparisons)


def EvaluateBlockCollection(EntityCollection1, EntityCollection2, blockCollection, goldStandard):
    """Evaluate a block collection against gold standard. Calculates pair completenes, reduction ratio (vs. brute force)  and reduction ratio with redundancy pruning (vs. brute force)"""
    gsIndexArray = _goldStandardToIndexArray(EntityCollection1, EntityCollection2, goldStandard)
    comparisons = _reduceBlockCollectionToComparisons(blockCollection)
    unique_comparisons = set(comparisons)
    matches = [comparison for comparison in gsIndexArray if comparison in unique_comparisons]
    pc = len(matches)/len(gsIndexArray)*100
    rr1 = (1 - len(comparisons)/(len(EntityCollection1)*len(EntityCollection2)))*100
    rr2 = (1 - len(unique_comparisons)/(len(EntityCollection1)*len(EntityCollection2)))*100
    print("Pair Completenes: {0}%, Reduction Ratio: {1}%, Reduction Ratio (with redundant pruned): {2}%".format(pc, rr1,rr2 ))


