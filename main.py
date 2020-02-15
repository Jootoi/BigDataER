from Blocking import TokenBlocking
from Blocking import TokenBlocker
from Blocking import TitleTokenizer
from Blocking import EvaluateBlockCollection
import timeit
import pandas as pd

def testTokenBlocking(entityCollection1, entityCollection2, goldStandard):
    print("Testing token blocking...\n")
    start = timeit.default_timer()
    blockCollection = TokenBlocking(entityCollection1, entityCollection2, TitleTokenizer, TokenBlocker)
    stop = timeit.default_timer()
    EvaluateBlockCollection(entityCollection1, entityCollection2, blockCollection, goldStandard)
    print("Time taken by blocking: {0} \n".format(stop-start))

def main():
    amazon = pd.read_csv("Amazon.csv", encoding = "ISO-8859-1").values
    google = pd.read_csv("GoogleProducts.csv", encoding = "ISO-8859-1").values
    goldStandard = pd.read_csv("Amzon_GoogleProducts_perfectMapping.csv", encoding = "ISO-8859-1").values

    testTokenBlocking(amazon, google, goldStandard)
    return 0

if __name__ == "__main__":
    main()