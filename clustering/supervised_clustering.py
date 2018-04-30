from __future__ import division
import json
import numpy as np
import time
import random
from sklearn.ensemble import RandomForestClassifier as RFC
import pandas as pd
from scipy.sparse import csc_matrix
from itertools import combinations
import operator
from sklearn.cluster import SpectralClustering as SC
from scipy.sparse import hstack

print 'preparing data...'

tfidf = json.loads(open('idxAndRefinedWorldCloudTfidf.json').read())

def gettfidf(dic):
    wordnum = 0
    for doc in tfidf:
        for id in map(eval, tfidf[doc].keys()):
            wordnum = max(wordnum, id)
    res = {}
    for pair in dic:
        left = csc_matrix((tfidf[str(pair[0])].values(), (np.zeros(len(tfidf[str(pair[0])])), map(eval, tfidf[str(pair[0])].keys()))), shape=(1, wordnum))
        right = csc_matrix((tfidf[str(pair[1])].values(), (np.zeros(len(tfidf[str(pair[1])])), map(eval, tfidf[str(pair[1])].keys()))), shape=(1, wordnum))
        key = '[' + str(pair[0]) + ', ' + str(pair[1]) + ']'
        res[key] = hstack([left,right])
    return res

def data_selection(feature, rmMulti):
    if feature == 'tfidf':
        if rmMulti:
            part1PosDic = map(eval, json.loads(open('rmMultiPart1PosDic_535.json').read()).keys())
            part1NegDic = map(eval, json.loads(open('rmMultiPart1NegDic_1636.json').read()).keys())
            part2PosDic = map(eval, json.loads(open('rmMultiPart2PosDic_11425.json').read()).keys())
            part2NegDic = map(eval, json.loads(open('rmMultiPart2NegDic_11548.json').read()).keys())
            part3PosDic = map(eval, json.loads(open('rmMultiPart3PosDic_276.json').read()).keys())
            part3NegDic = map(eval, json.loads(open('rmMultiPart3NegDic_387.json').read()).keys())
            part4PosDic = map(eval, json.loads(open('rmMultiPart4PosDic_1401.json').read()).keys())
            part4NegDic = map(eval, json.loads(open('rmMultiPart4NegDic_690.json').read()).keys())
            part5PosDic = map(eval, json.loads(open('rmMultiPart5PosDic_5117.json').read()).keys())
            part5NegDic = map(eval, json.loads(open('rmMultiPart5NegDic_10112.json').read()).keys())

            part1Pos = gettfidf(part1PosDic)
            part1Neg = gettfidf(part1NegDic)
            part2Pos = gettfidf(part2PosDic)
            part2Neg = gettfidf(part2NegDic)
            part3Pos = gettfidf(part3PosDic)
            part3Neg = gettfidf(part3NegDic)
            part4Pos = gettfidf(part4PosDic)
            part4Neg = gettfidf(part4NegDic)
            part5Pos = gettfidf(part5PosDic)
            part5Neg = gettfidf(part5NegDic)

    if feature == 'graph':
        if rmMulti:
            part1Pos = json.loads(open('rmMultiPart1PosDic_535.json').read())
            part1Neg = json.loads(open('rmMultiPart1NegDic_1636.json').read())
            part2Pos = json.loads(open('rmMultiPart2PosDic_11425.json').read())
            part2Neg = json.loads(open('rmMultiPart2NegDic_11548.json').read())
            part3Pos = json.loads(open('rmMultiPart3PosDic_276.json').read())
            part3Neg = json.loads(open('rmMultiPart3NegDic_387.json').read())
            part4Pos = json.loads(open('rmMultiPart4PosDic_1401.json').read())
            part4Neg = json.loads(open('rmMultiPart4NegDic_690.json').read())
            part5Pos = json.loads(open('rmMultiPart5PosDic_5117.json').read())
            part5Neg = json.loads(open('rmMultiPart5NegDic_10112.json').read())
        else:
            part1Pos = json.loads(open('Part1PosDic_651.json').read())
            part1Neg = json.loads(open('Part1NegDic_2290.json').read())
            part2Pos = json.loads(open('Part2PosDic_11954.json').read())
            part2Neg = json.loads(open('Part2NegDic_12366.json').read())
            part3Pos = json.loads(open('Part3PosDic_456.json').read())
            part3Neg = json.loads(open('Part3NegDic_612.json').read())
            part4Pos = json.loads(open('Part4PosDic_1401.json').read())
            part4Neg = json.loads(open('Part4NegDic_690.json').read())
            part5Pos = json.loads(open('Part5PosDic_5734.json').read())
            part5Neg = json.loads(open('Part5NegDic_11612.json').read())

    if feature == 'doc2vec':
        if rmMulti:
            part1Pos = json.loads(open('rmMultiPart1PosDVDic_535.json').read())
            part1Neg = json.loads(open('rmMultiPart1NegDVDic_1636.json').read())
            part2Pos = json.loads(open('rmMultiPart2PosDVDic_11425.json').read())
            part2Neg = json.loads(open('rmMultiPart2NegDVDic_11548.json').read())
            part3Pos = json.loads(open('rmMultiPart3PosDVDic_276.json').read())
            part3Neg = json.loads(open('rmMultiPart3NegDVDic_387.json').read())
            part4Pos = json.loads(open('rmMultiPart4PosDVDic_1401.json').read())
            part4Neg = json.loads(open('rmMultiPart4NegDVDic_690.json').read())
            part5Pos = json.loads(open('rmMultiPart5PosDVDic_5117.json').read())
            part5Neg = json.loads(open('rmMultiPart5NegDVDic_10112.json').read())
        else:
            part1Pos = json.loads(open('part1PosDVDic_651.json').read())
            part1Neg = json.loads(open('part1NegDVDic_2290.json').read())
            part2Pos = json.loads(open('part2PosDVDic_11954.json').read())
            part2Neg = json.loads(open('part2NegDVDic_12366.json').read())
            part3Pos = json.loads(open('part3PosDVDic_456.json').read())
            part3Neg = json.loads(open('part3NegDVDic_612.json').read())
            part4Pos = json.loads(open('part4PosDVDic_1401.json').read())
            part4Neg = json.loads(open('part4NegDVDic_690.json').read())
            part5Pos = json.loads(open('part5PosDVDic_5734.json').read())
            part5Neg = json.loads(open('part5NegDVDic_11612.json').read())
    if feature == 'comb':
        if rmMulti:
            part1Pos = json.loads(open('rmMultiPart1PosCombDic_535.json').read())
            part1Neg = json.loads(open('rmMultiPart1NegCombDic_1636.json').read())
            part2Pos = json.loads(open('rmMultiPart2PosCombDic_11425.json').read())
            part2Neg = json.loads(open('rmMultiPart2NegCombDic_11548.json').read())
            part3Pos = json.loads(open('rmMultiPart3PosCombDic_276.json').read())
            part3Neg = json.loads(open('rmMultiPart3NegCombDic_387.json').read())
            part4Pos = json.loads(open('rmMultiPart4PosCombDic_1401.json').read())
            part4Neg = json.loads(open('rmMultiPart4NegCombDic_690.json').read())
            part5Pos = json.loads(open('rmMultiPart5PosCombDic_5117.json').read())
            part5Neg = json.loads(open('rmMultiPart5NegCombDic_10112.json').read())
        else:
            part1Pos = json.loads(open('part1PosCombDic_651.json').read())
            part1Neg = json.loads(open('part1NegCombDic_2290.json').read())
            part2Pos = json.loads(open('part2PosCombDic_11954.json').read())
            part2Neg = json.loads(open('part2NegCombDic_12366.json').read())
            part3Pos = json.loads(open('part3PosCombDic_456.json').read())
            part3Neg = json.loads(open('part3NegCombDic_612.json').read())
            part4Pos = json.loads(open('part4PosCombDic_1401.json').read())
            part4Neg = json.loads(open('part4NegCombDic_690.json').read())
            part5Pos = json.loads(open('part5PosCombDic_5734.json').read())
            part5Neg = json.loads(open('part5NegCombDic_11612.json').read())

    if feature == 'fasttext':
        if rmMulti:
            part1Pos = json.loads(open('rmMultiPart1PosFasttext_535.json').read())
            part1Neg = json.loads(open('rmMultiPart1NegFasttext_1636.json').read())
            part2Pos = json.loads(open('rmMultiPart2PosFasttext_11425.json').read())
            part2Neg = json.loads(open('rmMultiPart2NegFasttext_11548.json').read())
            part3Pos = json.loads(open('rmMultiPart3PosFasttext_276.json').read())
            part3Neg = json.loads(open('rmMultiPart3NegFasttext_387.json').read())
            part4Pos = json.loads(open('rmMultiPart4PosFasttext_1401.json').read())
            part4Neg = json.loads(open('rmMultiPart4NegFasttext_690.json').read())
            part5Pos = json.loads(open('rmMultiPart5PosFasttext_5117.json').read())
            part5Neg = json.loads(open('rmMultiPart5NegFasttext_10112.json').read())
        # else:
        #     part1Pos = json.loads(open('part1PosFasttext_651.json').read())
        #     part1Neg = json.loads(open('part1NegFasttext_2290.json').read())
        #     part2Pos = json.loads(open('part2PosFasttext_11954.json').read())
        #     part2Neg = json.loads(open('part2NegFasttext_12366.json').read())
        #     part3Pos = json.loads(open('part3PosFasttext_456.json').read())
        #     part3Neg = json.loads(open('part3NegFasttext_612.json').read())
        #     part4Pos = json.loads(open('part4PosFasttext_1401.json').read())
        #     part4Neg = json.loads(open('part4NegFasttext_690.json').read())
        #     part5Pos = json.loads(open('part5PosFasttext_5734.json').read())
        #     part5Neg = json.loads(open('part5NegFasttext_11612.json').read())

    globalPos = json.loads(open('globalPosRef_20131.json').read())
    globalNeg = json.loads(open('globalNegRef_399888.json').read())
    print 'done preparing data!'
    return (part1Pos, part1Neg, part2Pos, part2Neg, part3Pos, part3Neg, part4Pos, part4Neg, part5Pos, part5Neg, globalPos, globalNeg)

print 'preparing sampling function...'
def takingSamples(alist, num=0, portion=0):
    assert ((num > 0 and portion == 0) or (num == 0 and portion > 0)), 'should offer only one method, num or portion'
    seed = int(round(time.time() * 1000)) % 100000000
    random.seed(seed)
    length_of_list = len(alist)
    listPicked = []
    listNotPicked = []

    if num > 0:
        chosen_ids = set()
        while len(chosen_ids) < num:
            tmpRandInt = random.randint(0, length_of_list - 1) # cover both head and tail
            chosen_ids.add(tmpRandInt)

        t_f_list = [False for i in range(length_of_list)]
        for i in chosen_ids:
            t_f_list[i] = True

        for i,j in enumerate(t_f_list):
            if j:
                listPicked.append(alist[i])
            else:
                listNotPicked.append(alist[i])

    if portion > 0:
        num = int(length_of_list * portion)
        chosen_ids = set()
        while len(chosen_ids) < num:
            tmpRandInt = random.randint(0, length_of_list - 1)  # cover both head and tail
            chosen_ids.add(tmpRandInt)

        t_f_list = [False for i in range(length_of_list)]
        for i in chosen_ids:
            t_f_list[i] = True

        for i, j in enumerate(t_f_list):
            if j:
                listPicked.append(alist[i])
            else:
                listNotPicked.append(alist[i])

    return (listPicked, listNotPicked)

def dic2List(dataset):
    result = []
    for key in dataset:
        temp = {}
        temp[key] = dataset[key]
        result.append(temp)
    return result

def list2Dic(dataset):
    result = {}
    for item in dataset:
        result[str(item.keys()[0])] = item.values()[0]
    return result
print 'done preparing sampling function!'


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


def fmeasure(result, groundTruth):  # if one pair of docs ever appear in the same cluster, it's labelled as pos. Only once.
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

print 'training model...'
def mat2arr(list):
    res = []
    for mat in list:
        res.append(mat.toarray())
    return res

def supervised_clu(feature, rmMulti, trial):
    (part1Pos, part1Neg, part2Pos, part2Neg, part3Pos, part3Neg, part4Pos, part4Neg, part5Pos, part5Neg, globalPos,
     globalNeg) = data_selection(feature, rmMulti)
    sumpurity = 0
    sumfone = 0
    for i in range(0, trial):
        print '#', i + 1, 'trial!!!'
        pos_dataset = dic2List(
            globalPos)  # dic2List(part1Pos) + dic2List(part2Pos) + dic2List(part3Pos) + dic2List(part4Pos) + dic2List(part5Pos)  #
        neg_dataset = dic2List(
            globalNeg)  # dic2List(part1Neg) + dic2List(part2Neg) + dic2List(part3Neg) + dic2List(part4Neg) + dic2List(part5Neg)  #
        # print len(pos_dataset)

        num_pos_sample = int(0.3 * len(pos_dataset))
        num_neg_sample = num_pos_sample

        (posPicked, posNotPicked) = takingSamples(pos_dataset, num=num_pos_sample)
        (negPicked, negNotPicked) = takingSamples(neg_dataset, num=num_neg_sample)
        # print len(posPicked),len(negPicked)
        # print posPicked, posNotPicked

        # train_X = pd.DataFrame(mat2arr(list2Dic(posPicked).values() + list2Dic(negPicked).values()))
        train_X = pd.DataFrame(list2Dic(posPicked).values() + list2Dic(negPicked).values())
        train_y = np.array(
            [1 for i in range(len(list2Dic(posPicked).values()))] + [0 for i in
                                                                     range(len(list2Dic(negPicked).values()))])
        print len(train_X), len(train_y)

        reg = RFC(n_estimators=200, max_features='log2')
        model = reg.fit(train_X, train_y)
        # print 'model ready!'

        # print 'get affinity matrix...'
        matrixVal = {}
        for item in posPicked:
            matrixVal[str(item.keys()[0])] = 1
        for item in negPicked:
            matrixVal[str(item.keys()[0])] = 0

        test_X = posNotPicked + negNotPicked
        modelIn = list2Dic(test_X)
        test_Y = model.predict_proba(modelIn.values())[:, 1]
        for i in range(0, len(modelIn)):
            matrixVal[modelIn.keys()[i]] = test_Y[i]

        # print matrixVal.keys()
        # print map(eval,matrixVal.keys())
        # print matrixVal.values()
        # print size
        row = []
        col = []
        docMap = {}
        mapDoc = {}
        size = 0
        for pair in map(eval, matrixVal.keys()):
            for doc in pair:
                if not docMap.has_key(doc):
                    docMap[doc] = size
                    mapDoc[size] = doc
                    size += 1
        # print mapDoc
        # print docMap
        for pair in map(eval, matrixVal.keys()):
            row.append(docMap[pair[0]])
            col.append(docMap[pair[1]])
        for pair in map(eval, matrixVal.keys()):
            row.append(docMap[pair[1]])
            col.append(docMap[pair[0]])
        data = matrixVal.values() + matrixVal.values()
        # print size
        affinity = csc_matrix((data, (row, col)), shape=(size, size)).toarray()
        # print 'affinity matrix get!'

        # print 'run clustering...'
        # groundTruth = json.loads(open('groundTruth.json').read())
        # groundTruth = json.loads(open('rmMultiGroundTruth.json').read()) # some documents appears in one part only once, but multiple time in global
        groundTruth = json.loads(open(
            'rmMultiGroundTruthNew.json').read())  # rmMultiGroundTruthNew.json is for simply combining all parts only
        # groundTruth = json.loads(open('part1CluInd.json').read())
        # groundTruth = json.loads(open('rmMultiPart5CluInd.json').read())
        num_clu = len(groundTruth)
        # print num_clu
        model = SC(n_clusters=num_clu, affinity='precomputed')
        res = model.fit_predict(affinity)
        # print res
        # print len(res), len(set(res))

        resDic = {}
        for i in range(len(res)):
            if not resDic.has_key(res[i]):
                resDic[res[i]] = []
                resDic[res[i]].append(mapDoc[i])
            else:
                resDic[res[i]].append(mapDoc[i])
        result = resDic.values()

        purVal = purity(result, groundTruth)
        (pre, rec, fone) = fmeasure(result, groundTruth)
        sumpurity += purVal
        sumfone += fone
        print 'purity %.4f' % purVal, 'precision: %.4f' % pre, 'recall: %.4f' % rec, 'f1: %.4f' % fone

        return (sumpurity, sumfone)

if __name__ == "__main__":
    feature = 'tfidf'  # 'fasttext' # 'graph' #'doc2vec' #'comb' #
    rmMulti = True  # False #
    trial = 1 # any number
    (sumpurity, sumfone) = supervised_clu(feature, rmMulti, trial)
    avgpurity = sumpurity / 10
    avgfone = sumfone / 10
    print 'average: purity %.4f' % avgpurity, 'f1: %.4f' % avgfone