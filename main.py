from Blocking import TokenBlocking
from Blocking import TokenBlocker
from Blocking import titleTokenizer
import pandas as pd



def main():
    amazon = pandas.read_csv("Amazon.csv", encoding = "ISO-8859-1").values
    google = pandas.read_csv("GoogleProducts.csv", encoding = "ISO-8859-1").values
    return 0

if __name__ == "__main__":
    main()