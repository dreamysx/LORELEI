import json
from sklearn.cluster import AgglomerativeClustering as AC
from sklearn.cluster import SpectralClustering as SC
from sklearn.mixture import GMM

from sklearn.mixture import GaussianMixture as GM
import numpy as np
from scipy.sparse import csc_matrix
from itertools import combinations
import operator

idTfidf = json.loads(open('gt_tfidf.json').read())
groundTruth = json.loads(open('gt_groundTruth.json').read())


def clustering(idTfidf, num_clu, term_num):
    docFeature = idTfidf
    vecTfidf = {}
    for file in idTfidf:
        row = np.zeros(len(idTfidf[file]))
        col = idTfidf[file].keys()
        val = idTfidf[file].values()
        vec = csc_matrix((np.array(val), (np.array(row), np.array(col))), shape=(1, term_num))
        vecTfidf[file] = vec.todense().tolist()[0]
    # print vecTfidf
    features = vecTfidf.values()
    # print features

    selection = 'GM'  # selecting model here!!! Options: AgglomerativeClustering as AC, SpectralClustering as SC, GMM

    if selection == 'AC':
        model = AC(n_clusters=num_clu, affinity='cosine', linkage='average')
    if selection == 'SC':
        model = SC(n_clusters=num_clu, affinity='cosine')
    if selection == 'GMM':
        model = GMM(n_components=num_clu, covariance_type='full')
    if selection == 'GM':
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
    # print result
    with open('gt_GMRes.json', 'w') as f:
        f.write(json.dumps(result))

    return result


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
    num_clu = len(groundTruth)
    term_num = 1992
    result = clustering(idTfidf, num_clu, term_num)
    purVal = purity(result, groundTruth)
    (pre, rec, fone) = fmeasure(result, groundTruth)

    print 'purity %.4f' % purVal, 'precision: %.4f' % pre, 'recall: %.4f' % rec, 'f1: %.4f' % fone