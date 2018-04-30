from itertools import combinations
from sklearn.cluster import AgglomerativeClustering as AC
from sklearn.cluster import SpectralClustering as SC
from sklearn.mixture import GMM
from sklearn.mixture import GaussianMixture as GM
from sklearn.cluster import KMeans
import json
import operator


def unsupervised_clu(feature, part, model_selection):
    if part:
        if feature == 'graph':
            docFeature = json.loads(open('rmMultiPart1WOZeroGraph.json').read())
        if feature == 'doc2vec':
            docFeature = json.loads(open('rmMultiPart1Doc2vec.json').read())
        if feature == 'comb':
            walk = json.loads(open('rmMultiPart1WOZeroGraph.json').read())
            dv = json.loads(open('rmMultiPart1Doc2vec.json').read())
            docFeature = {}
            for doc in walk:
                val = walk[doc] + dv[doc]
                docFeature[doc] = val
        groundTruth = json.loads(open('rmMultiPart1CluInd.json').read())
        num_clu = len(groundTruth)  # number of clusters in each part
    else:
        rmMulti = True  # False #
        if rmMulti:
            if feature == 'graph':
                docFeature = json.loads(open('rmMultiCluDatabaseWOZeroGraph.json').read())
            if feature == 'doc2vec':
                docFeature = json.loads(open('rmMultiCluDatabaseDoc2vec.json').read())
            if feature == 'comb':
                walk = json.loads(open('rmMultiCluDatabaseWOZeroGraph.json').read())
                dv = json.loads(open('rmMultiCluDatabaseDoc2vec.json').read())
                docFeature = {}
                for doc in walk:
                    val = walk[doc] + dv[doc]
                    docFeature[doc] = val
            groundTruth = json.loads(open('rmMultiGroundTruth.json').read())
            num_clu = len(
                groundTruth)  # number of clusters after removing documents appearing multi-cluster, #doc = 1274 (3 all 0s for walk)
        else:
            if feature == 'graph':
                docFeature = json.loads(open('cluDatabaseWOZeroGraph.json').read())
            if feature == 'doc2vec':
                docFeature = json.loads(open('cluDatabaseDoc2vec.json').read())
            if feature == 'comb':
                walk = json.loads(open('cluDatabaseWOZeroGraph.json').read())
                dv = json.loads(open('cluDatabaseDoc2vec.json').read())
                docFeature = {}
                for doc in walk:
                    val = walk[doc] + dv[doc]
                    docFeature[doc] = val
            groundTruth = json.loads(open('groundTruth.json').read())
            num_clu = len(
                groundTruth)  # number of clusters before removing documents appearing multi-cluster, #doc = 1393 (3 all 0s for walk)

    features = docFeature.values()
    if model_selection == 'AC':
        model = AC(n_clusters=num_clu, affinity='cosine', linkage='average')
    if model_selection == 'SC':
        model = SC(n_clusters=num_clu, affinity='cosine')
    if model_selection == 'GMM':
        model = GMM(n_components=num_clu, covariance_type='full')
    if model_selection == 'KMeans':
        model = KMeans(n_clusters=num_clu)
    if model_selection == 'GM':
        model = GM(n_components=num_clu)
        model.fit(features)
        res = model.predict(features)
    else:
        res = model.fit_predict(features)
    resDic = {}
    for i in range(len(res)):
        if not resDic.has_key(res[i]):
            resDic[res[i]] = []
            resDic[res[i]].append(int(docFeature.keys()[i]))
        else:
            resDic[res[i]].append(int(docFeature.keys()[i]))
    result = resDic.values()

    return (result, groundTruth)


def purity(result, groundTruth):
    N = 0
    majorSum = 0
    docFractionCount = {}
    docClusterCount = {}
    for resClu in result:
        # calculate fraction of each doc
        for doc in resClu:
            N += 1
            for truClu in groundTruth:
                if doc in truClu:
                    docClusterCount[doc] = docClusterCount.get(doc, 0) + 1
            for doc in docClusterCount:
                docFractionCount[doc] = float(1) / docClusterCount[doc]
        # print docFractionCount

        docCount = {}
        fracCount = {}
        trueCluID = 0
        for truClu in groundTruth:
            for doc in resClu:
                if doc in truClu:
                    fracCount[trueCluID] = fracCount.get(trueCluID, 0.0) + docFractionCount[doc]
                    docCount[trueCluID] = docCount.get(trueCluID, 0.0) + 1.0
            trueCluID += 1
        # print len(resClu)
        # print fracCount
        # print docCount
        major = sorted(fracCount.items(), key=operator.itemgetter(1), reverse=True)[0]
        majorSum += docCount[major[0]]
        # print docCount[major[0]]
    return float(majorSum) / N


def fmeasure(result,
             groundTruth):  # if one pair of docs ever appear in the same cluster, it's labelled as pos. Only once.
    TP = 0
    FP = 0
    FN = 0
    posPair = 0
    for i in range(len(result)):
        for pair in combinations(result[i], 2):
            posPair += 1
            for truClu in groundTruth:
                if pair[0] in truClu and pair[1] in truClu:
                    TP += 1
                    break
        FP = posPair - TP
        for doc in result[i]:
            for resClu in result[(i + 1):]:
                for anotherDoc in resClu:
                    for truClu in groundTruth:
                        if doc in truClu and anotherDoc in truClu:
                            FN += 1
    # print TP, FP
    pre = float(TP) / (TP + FP)
    rec = float(TP) / (TP + FN)
    fone = 2 * pre * rec / float(pre + rec)

    return (pre, rec, fone)


if __name__ == "__main__":
    feature = 'graph'  # 'comb' #'doc2vec' #
    part = False  # experiment on part data set?True #
    selection = 'SC'  # selecting model here!!! Options: AgglomerativeClustering as AC, SpectralClustering as SC, GMM
    (result, groundTruth) = unsupervised_clu(feature, part, selection)
    purVal = purity(result, groundTruth)
    (pre, rec, fone) = fmeasure(result, groundTruth)

    print 'purity %.4f' % purVal, 'precision: %.4f' % pre, 'recall: %.4f' % rec, 'f1: %.4f' % fone