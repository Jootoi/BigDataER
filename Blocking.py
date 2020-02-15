import pandas
import numpy as np
import nltk
from nltk.stem import LancasterStemmer
nltk.download('stopwords')
nltk.download("punkt")
from nltk.corpus import stopwords
stop = stopwords.words('english')
lancaster = LancasterStemmer()

def titleTokenizer(EC):
    #Title is always second column in these datasets
    titles = EC[:,1]
    stemmed = []
    for title in titles:
        tokens = nltk.word_tokenize(title.lower())
        filtered = [token for token in tokens if token not in stop and len(token)>1]
        stemmed.append([lancaster.stem(token) for token in filtered])
    return stemmed


def tokenBlocker(tokenArray):
    token_dictionary = {}
    for index, tokens in enumerate(tokenArray):
        for token in tokens:
            a = token_dictionary.get(token, [])
            a.append(index)
            token_dictionary[token] = a
    return(token_dictionary)

def joinBlocks(BC1, BC2):
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

    blockCollection = joinBlocks(blockedEC1, blockedEC2)

    duplicates = matchFun(blockCollection)
    return(duplicates)
