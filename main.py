from Blocking import TokenBlocking
from Blocking import TokenBlocker
from Blocking import MultiColumnTokenizer
from Blocking import ColumnTokenizer
from Blocking import EvaluateBlockCollection
from AttributeClusteringBlocking import AttributeClusteringBlocking
from AttributeClusteringBlocking import JaccardSimilarity
import timeit
import pandas as pd

def testTokenBlocking(entityCollection1, entityCollection2, goldStandard):
    print("Testing token blocking...\n")
    start = timeit.default_timer()
    blockCollection = TokenBlocking(entityCollection1, entityCollection2, MultiColumnTokenizer, TokenBlocker)
    stop = timeit.default_timer()
    EvaluateBlockCollection(entityCollection1, entityCollection2, blockCollection, goldStandard)
    print("Time taken by blocking: {0:.2f} \n".format(stop-start))

def testACBlocking(entityCollection1, entityCollection2, goldStandard):
    print("Testing attribute clustering blocking...\n")
    start = timeit.default_timer()
    blockCollection = AttributeClusteringBlocking(entityCollection1, entityCollection2, ColumnTokenizer, JaccardSimilarity)
    stop = timeit.default_timer()
    EvaluateBlockCollection(entityCollection1, entityCollection2, blockCollection, goldStandard)
    print("Time taken by blocking: {0:.2f} \n".format(stop-start))

def main():
    amazon = pd.read_csv("Amazon.csv", encoding = "ISO-8859-1").values
    google = pd.read_csv("GoogleProducts.csv", encoding = "ISO-8859-1").values
    goldStandard = pd.read_csv("Amzon_GoogleProducts_perfectMapping.csv", encoding = "ISO-8859-1").values

    testTokenBlocking(amazon, google, goldStandard)
    testACBlocking(amazon, google, goldStandard)
    return 0

if __name__ == "__main__":
    main()