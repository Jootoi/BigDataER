from Blocking import TokenBlocking
from Blocking import TokenBlocker
from Blocking import MultiColumnTokenizer
from Blocking import ColumnTokenizer
from Blocking import EvaluateBlockCollection
from AttributeClusteringBlocking import AttributeClusteringBlocking
from AttributeClusteringBlocking import JaccardSimilarity
from MetaBlocking import GraphBuilder
from MetaBlocking import JaccardWeighting
from MetaBlocking import CBSWeighting
from MetaBlocking import WeightEdgePruning
from MetaBlocking import BlockCollecting
from MetaBlocking import CardinalityNodePruning
from MetaBlocking import EvaluateMetaBlockCollection
import timeit
import pandas as pd

def testTokenBlocking(entityCollection1, entityCollection2, goldStandard):
    print("Testing token blocking...\n")
    start = timeit.default_timer()
    blockCollection = TokenBlocking(entityCollection1, entityCollection2, MultiColumnTokenizer, TokenBlocker)
    stop = timeit.default_timer()
    EvaluateBlockCollection(entityCollection1, entityCollection2, blockCollection, goldStandard)
    print("Time taken by blocking: {0:.2f} \n".format(stop-start))
    return blockCollection

def testACBlocking(entityCollection1, entityCollection2, goldStandard):
    print("Testing attribute clustering blocking...\n")
    start = timeit.default_timer()
    blockCollection = AttributeClusteringBlocking(entityCollection1, entityCollection2, ColumnTokenizer, JaccardSimilarity)
    stop = timeit.default_timer()
    EvaluateBlockCollection(entityCollection1, entityCollection2, blockCollection, goldStandard)
    print("Time taken by blocking: {0:.2f} \n".format(stop-start))
    return blockCollection

def main():
    amazon = pd.read_csv("Amazon.csv", encoding = "ISO-8859-1").values
    google = pd.read_csv("GoogleProducts.csv", encoding = "ISO-8859-1").values
    goldStandard = pd.read_csv("Amzon_GoogleProducts_perfectMapping.csv", encoding = "ISO-8859-1").values

    blockCollectionToken = testTokenBlocking(amazon, google, goldStandard)
    blockCollectionAC = testACBlocking(amazon, google, goldStandard)
    
    # Build the graphs from each blocking method
    tokenGraph = GraphBuilder(blockCollectionToken)
    ACGraph = GraphBuilder(blockCollectionAC)
    
    # Assign all required weights for each graph
    tokenGraphCBS = CBSWeighting(tokenGraph)
    tokenGraphJaccard = JaccardWeighting(tokenGraph)
    ACGraphCBS = CBSWeighting(ACGraph)
    ACGraphJaccard = JaccardWeighting(ACGraph)
    
    # Test each combination of 1) blocking method, 2) weighting method, 3) pruning method
    # tc = token with common blocks scheme (CBS); acj = attribute clustering with jaccard etc
    # wep = weight edge pruning, cnp = cardinality node pruning
    print("Token, CBS, WEP\n")
    start = timeit.default_timer()
    tc_wep_comparisons = WeightEdgePruning(tokenGraphCBS)
    stop = timeit.default_timer()
    EvaluateMetaBlockCollection(amazon, google, tc_wep_comparisons, goldStandard)
    print("Time taken: {0:.2f}s \n".format(stop-start))
    
    print("Token, Jaccard, WEP\n")
    start = timeit.default_timer()
    tj_wep_comparisons = WeightEdgePruning(tokenGraphJaccard)
    stop = timeit.default_timer()
    EvaluateMetaBlockCollection(amazon, google, tj_wep_comparisons, goldStandard)
    print("Time taken: {0:.2f}s \n".format(stop-start))
    
    print("Token, CBS, CNP\n")
    start = timeit.default_timer()
    tc_cnp_comparisons = CardinalityNodePruning(tokenGraphCBS)
    stop = timeit.default_timer()
    EvaluateMetaBlockCollection(amazon, google, tc_cnp_comparisons, goldStandard)
    print("Time taken: {0:.2f}s \n".format(stop-start))
    
    print("Token, Jaccard, CNP\n")
    start = timeit.default_timer()
    tj_cnp_comparisons = CardinalityNodePruning(tokenGraphJaccard)
    stop = timeit.default_timer()
    EvaluateMetaBlockCollection(amazon, google, tj_cnp_comparisons, goldStandard)
    print("Time taken: {0:.2f}s \n".format(stop-start))
    
    # AC tests
    print("AC, CBS, WEP\n")
    start = timeit.default_timer()
    acc_wep_comparisons = WeightEdgePruning(ACGraphCBS)
    stop = timeit.default_timer()
    EvaluateMetaBlockCollection(amazon, google, acc_wep_comparisons, goldStandard)
    print("Time taken: {0:.2f}s \n".format(stop-start))
    
    print("AC, Jaccard, WEP\n")
    start = timeit.default_timer()
    acj_wep_comparisons = WeightEdgePruning(ACGraphJaccard)
    stop = timeit.default_timer()
    EvaluateMetaBlockCollection(amazon, google, acj_wep_comparisons, goldStandard)
    print("Time taken: {0:.2f}s \n".format(stop-start))
    
    print("AC, CBS, CNP\n")
    start = timeit.default_timer()
    acc_cnp_comparisons = CardinalityNodePruning(ACGraphCBS)
    stop = timeit.default_timer()
    EvaluateMetaBlockCollection(amazon, google, acc_cnp_comparisons, goldStandard)
    print("Time taken: {0:.2f}s \n".format(stop-start))
    
    print("AC, Jaccard, CNP\n")
    start = timeit.default_timer()
    acj_cnp_comparisons = CardinalityNodePruning(ACGraphJaccard)
    stop = timeit.default_timer()
    EvaluateMetaBlockCollection(amazon, google, acj_cnp_comparisons, goldStandard)
    print("Time taken: {0:.2f}s \n".format(stop-start))

    # Makes the new block collections from each combination for possible next step
    bc_tc_wep = BlockCollecting(tc_wep_comparisons)
    bc_tj_wep = BlockCollecting(tj_wep_comparisons)
    bc_tc_cnp = BlockCollecting(tc_cnp_comparisons)
    bc_tj_cnp = BlockCollecting(tj_cnp_comparisons)
    
    bc_acc_wep = BlockCollecting(acc_wep_comparisons)
    bc_acj_wep = BlockCollecting(acj_wep_comparisons)
    bc_acc_cnp = BlockCollecting(acc_cnp_comparisons)
    bc_acj_cnp = BlockCollecting(acj_cnp_comparisons)
    return 0

if __name__ == "__main__":
    main()