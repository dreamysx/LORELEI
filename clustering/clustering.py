import json
import sys
import os
import MTokenizer
import networkx as nx
from sklearn.cluster import AgglomerativeClustering as AC
from sklearn.cluster import SpectralClustering as SC
from sklearn.ensemble import RandomForestClassifier as RFC
from itertools import combinations
import operator
from tsne import tsne
import numpy
import pylab as Plot
import random
import time
from scipy.sparse import csc_matrix
import pandas as pd
import numpy as np



#################### preparation part ######################
def lowerList(list):
    if list != None:
        res = []
        for temp in list:
            res.append(temp.lower())
        return res

def topic4id(id):
    file = json.loads(open(path + 'data_' + id + '.json').read())
    return file['loreleiJSONMapping']['topics']

def checkRelevance(uuid1, uuid2):
    for topic in topic4id(uuid1):
        if topic in topic4id(uuid2):
            return True
    return False

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

def mat2arr(list):
    res = []
    for mat in list:
        res.append(mat.toarray())
    return res

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


#################### pipeline part ######################
def textClean(name_list):
    docE = {}
    docRE = {}
    for file_name in name_list:
        file = json.loads(open(path + file_name).read())
        if file['situationFrame']['entities'] != None:
            docE[file['uuid']] = lowerList(file['situationFrame']['entities'])
            refinedE = []
            for entity in lowerList(file['situationFrame']['entities']):
                refinedE += MTokenizer.MTokenizer.tokenize_string(entity)
            docRE[file['uuid']] = list(set(refinedE))
    print 'Finish cleaning data...'
    return docRE

def buildEdgeList(textDict):
    edges = []
    adjs = []
    intRef = {}
    iddocRef = {}
    docidRef = {}
    index = 0
    id2 = 0
    for id in textDict:
        intRef[id] = index
        iddocRef[str(index)] = id
        docidRef[id] = str(index)
        id2 += 1
        index += 1
        for word in textDict[id]:
            if word not in intRef:
                intRef[word] = index
                index += 1
    for id in textDict:
        temp = []
        temp.append(intRef[id])
        for word in textDict[id]:
            edges.append((intRef[id], intRef[word]))
            temp.append(intRef[word])
        adjs.append(temp)
    G = nx.Graph(edges)
    print 'Number of nodes in graph:' + str(len(intRef))
    print 'Number of documents:' + str(len(iddocRef))
    with open('graphRef.json', 'w') as f:
        f.write(json.dumps(intRef))
    with open('graphIddocRef.json', 'w') as f:
        f.write(json.dumps(iddocRef))
    with open('graphDocidRef.json', 'w') as f:
        f.write(json.dumps(docidRef))

    print 'Finish getting edgelist...'
    return (edges, adjs, G)

def dwFeature(path):
    name_list = os.listdir(path)
    textDict = textClean(name_list)
    edgeList = buildEdgeList(textDict)[0]
    # adjsList = buildEdgeList(textDict)[1]

    with open('graph.edgelist', 'w') as f:
        for pair in edgeList:
            f.write(str(pair[0]))
            f.write('\t')
            f.write(str(pair[1]))
            f.write('\n')
            f.write(str(pair[1]))
            f.write('\t')
            f.write(str(pair[0]))
            f.write('\n')
    # with open('graph.adjlist', 'w') as f:
    #     for line in adjsList:
    #         for item in line:
    #             f.write(str(item))
    #             f.write(' ')
    #         f.write('\n')

    os.system('deepwalk --input graph.edgelist --representation-size 100 --walk-length 100  --output graph.embeddings')
    print 'Finish getting graph embeddings...'

def unsupervised(numClu, graphEmb):
    print 'Buidling unsupervised model...'
    model = AC(n_clusters=numClu, affinity='cosine', linkage='complete')
    res = model.fit_predict(graphEmb.values())
    return res

def supervised(numClu, affinity):
    print 'Buidling supervised model...'
    model = SC(n_clusters=numClu, affinity='precomputed')
    res = model.fit_predict(affinity)
    return res

def getGT(path):
    clusters = {}
    name_list = os.listdir(path)
    for file_name in name_list:
        file = json.loads(open(path + file_name).read())
        for topic in file['loreleiJSONMapping']['topics']:
            if topic not in clusters:
                clusters[topic] = []
            else:
                clusters[topic].append(file_name[5:-5])
    groundTruth = clusters.values()
    return groundTruth

def getAffinity(graphEmb):
    print 'Getting affinity matrix for supervised model...'
    docFeatureList = []
    pos_data = {}
    neg_data = {}
    docidRef = {}
    iddocRef = {}
    id = 0
    for uuid in graphEmb:
        docidRef[uuid] = id
        iddocRef[id] = uuid
        curDoc = [id, graphEmb[uuid]]
        docFeatureList.append(str(curDoc))
        id += 1
    with open('docidRef.json', 'w') as f:
        f.write(json.dumps(docidRef))
    with open('iddocRef.json', 'w') as f:
        f.write(json.dumps(iddocRef))
    # print 'start'
    for i in range(len(docFeatureList)):
        for j in range(i+1, len(docFeatureList)):
            left = eval(docFeatureList[i])
            right = eval(docFeatureList[j])
            if (checkRelevance(iddocRef[left[0]], iddocRef[right[0]])):
                key = [left[0], right[0]]
                pos_data[str(key)] = left[1] + right[1]
                print pos_data
            else:
                key = [left[0], right[0]]
                neg_data[str(key)] = left[1] + right[1]
                print neg_data
    # print 'end'
    pos_dataset = dic2List(pos_data)
    neg_dataset = dic2List(neg_data)
    num_pos_sample = int(0.3 * len(pos_dataset))
    num_neg_sample = num_pos_sample

    (posPicked, posNotPicked) = takingSamples(pos_dataset, num=num_pos_sample)
    (negPicked, negNotPicked) = takingSamples(neg_dataset, num=num_neg_sample)

    train_X = pd.DataFrame(list2Dic(posPicked).values() + list2Dic(negPicked).values())
    train_y = np.array(
        [1 for i in range(len(list2Dic(posPicked).values()))] + [0 for i in range(len(list2Dic(negPicked).values()))])
    print len(train_X), len(train_y)

    reg = RFC(n_estimators=50, max_features='log2')
    model = reg.fit(train_X, train_y)

    print 'Getting affinity matrix...'
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

    return affinity


def cluster(modelRes):
    print 'Clustering...'
    resDic = {}
    for i in range(len(modelRes)):
        if not resDic.has_key(modelRes[i]):
            resDic[modelRes[i]] = []
            resDic[modelRes[i]].append(graphEmb.keys()[i])
        else:
            resDic[modelRes[i]].append(graphEmb.keys()[i])
    result = resDic.values()
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

def measurement(result, groundTruth):
    purVal = purity(result, groundTruth)
    (pre, rec, fone) = fmeasure(result, groundTruth)
    return (purVal, pre, rec, fone)

def tsneVis(result):  # this will generate a .png file showing visualization of clusering
    print 'Visualizing by Tsne...'
    X = []
    labels = []
    index = 0
    for cluster in result:
        for doc in cluster:
            X.append(graphEmb[doc])
            labels.append(index)
        index += 1
    Y = tsne(numpy.array(X), 2, 50, 20.0)
    # print Y
    Plot.scatter(Y[:,0], Y[:,1], 20, numpy.array(labels))
    Plot.show()

if __name__ == "__main__":
    path = sys.argv[1] #'/Users/dreamysx/Documents/USC-DTIN/isi/reliefWebProcessed/' #
    dwFeature(path)
    modelType = sys.argv[2] #'supervised' #'unsupervised' #
    numClu = sys.argv[3] #30  #
    features = open('graph.embeddings').readlines()[1:]
    graphIddocRef = json.loads(open('graphIddocRef.json').read())
    graphEmb = {}

    for line in features:
        docid = line.split()[0]
        if docid in graphIddocRef:
            graphEmb[graphIddocRef[docid]] = map(eval, line.split()[1:])
    with open('deepwalk.json', 'w') as f:
        f.write(json.dumps(graphEmb))

    if modelType == 'unsupervised':
        modelRes = unsupervised(numClu, graphEmb)
        res = cluster(modelRes)
        print 'Unsupervised clustering result: ', res
    elif modelType == 'supervised':
        affinity = getAffinity(graphEmb)
        groundTruth = getGT(path)
        numClu = len(groundTruth)
        modelRes = supervised(numClu, affinity)
        res = cluster(modelRes)
        iddocRef = json.loads(open('iddocRef.json').read())
        result = []
        for tempCluster in res:
            temp = []
            for tempid in tempCluster:
                temp.append(iddocRef[tempid])
            result.append(temp)
        print 'Supervised clustering result: ', result
        (purVal, pre, rec, fone) = measurement(result, groundTruth)
        print 'purity: %.4f' % purVal, 'precision: %.4f' % pre, 'recall: %.4f' % rec, 'f1: %.4f' % fone

        if sys.argv[4]:
            tsneVis(res)