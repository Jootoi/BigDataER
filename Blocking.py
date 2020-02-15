import pandas
import numpy as np
import nltk
from nltk.stem import LancasterStemmer
nltk.download('stopwords')
nltk.download("punkt")
from nltk.corpus import stopwords
stop = stopwords.words('english')
lancaster = LancasterStemmer()

# Maybe define these dataset specific function in main instead or make them generic
def AmazonGoogletitleTokenizer(EC):
    #Title is always second column in these datasets
    titles = EC[:,1]
    stemmed = []
    for title in titles:
        tokens = nltk.word_tokenize(title.lower())
        filtered = [token for token in tokens if token not in stop and len(token)>1]
        stemmed.append([lancaster.stem(token) for token in filtered])
    return stemmed


def TokenBlocker(tokenArray):
    token_dictionary = {}
    for index, tokens in enumerate(tokenArray):
        for token in tokens:
            a = token_dictionary.get(token, [])
            a.append(index)
            token_dictionary[token] = a
    return(token_dictionary)



def JoinBlocks(BC1, BC2):
    combined = {}
    for key in BC1:
        if(key in BC2):
            combined[key] = (BC1[key], BC2[key])
    return(combined)

def TokenBlocking(EntityCollection1, EntityCollection2, transformationFun, constraintFun, matchFun):
    transformedEC1 = transformationFun(EntityCollection1)
    transformedEC2 = transformationFun(EntityCollection2)

    blockedEC1 = constraintFun(transformedEC1)
    blockedEC2 = constraintFun(transformedEC2)

    blockCollection = JoinBlocks(blockedEC1, blockedEC2)
    return(blockCollection)


def _goldStandardToIndexArray(EntityCollection1, EntityCollection2, goldStandard):
    #Extracts all the required comparisons from the goldStandard dataset as list of tuples
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
    comparisons = []
    part1 = block[0]
    part2 = block[1]
    for item in part1:
        for item2 in part2:
            comparisons.append((item, item2))
    return(comparisons)

def _reduceBlockCollectionToComparisons(bc):
    comparisons = [_reduceBlockToComparisons(block) for key, block in bc.items()]
    comparisons =  [item for sublist in comparisons for item in sublist]
    return(comparisons)


def EvaluateBlockCollection(EntityCollection1, EntityCollection2, blockCollection, goldStandard):
    gsIndexArray = _goldStandardToIndexArray(EntityCollection1, EntityCollection2, goldStandard)
    comparisons = _reduceBlockCollectionToComparisons(blockCollection)
    unique_comparisons = set(comparisons)
    matches = [comparison for comparison in gsIndexArray if comparison in unique_comparisons]
    pc = len(matches)/len(gsIndexArray)*100
    rr1 = (1 - len(comparisons)/(len(EntityCollection1)*len(EntityCollection2)))*100
    rr2 = (1 - len(unique_comparisons)/(len(EntityCollection1)*len(EntityCollection2)))*100
    print("Pair Completenes: {0}%, Reduction Ratio: {1}%, Reduction Ratio (with redundant pruned): {2}%".format(pc, rr1,rr2 ))


