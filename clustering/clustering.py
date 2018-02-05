import json
import sys
import os
import MTokenizer
import networkx as nx
from sklearn.cluster import AgglomerativeClustering as AC
from sklearn.cluster import SpectralClustering as SC
from tsne import tsne
import numpy
import pylab as Plot

def lowerList(list):
    if list != None:
        res = []
        for temp in list:
            res.append(temp.lower())
        return res

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
    return docRE

def buildEdgeList(textDict):
    edges = []
    adjs = []
    intRef = {}
    idRef = {}
    index = 0
    for id in textDict:
        intRef[id] = index
        idRef[str(index)] = id
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
    print 'Number of documents:' + str(len(idRef))
    # with open('graphRef.json', 'w') as f:
    #     f.write(json.dumps(intRef))
    # with open('docidRef.json', 'w') as f:
    #     f.write(json.dumps(idRef))

    return (edges, adjs, G)

def dwFeature():
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

features = open('graph.embeddings').readlines()[1:]
docidRef = json.loads(open('docidRef.json').read())
graphEmb = {}

for line in features:
    docid = line.split()[0]
    if docid in docidRef:
        graphEmb[docidRef[docid]] = map(eval, line.split()[1:])

# print len(graphEmb), len(docidRef)
# with open('deepwalk.json', 'w') as f:
#     f.write(json.dumps(graphEmb))

def cluster(modelType, numClu):
    if modelType == 'unsupervised':
        model = AC(n_clusters=numClu, affinity='cosine', linkage='complete')
        res = model.fit_predict(graphEmb.values())
        resDic = {}
        for i in range(len(res)):
            if not resDic.has_key(res[i]):
                resDic[res[i]] = []
                resDic[res[i]].append(graphEmb.keys()[i])
            else:
                resDic[res[i]].append(graphEmb.keys()[i])
        result = resDic.values()

    return result

def tsneVis(result):  # this will generate a .png file showing visualization of clusering
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
    path = sys.argv[1] #'/Users/dreamysx/Documents/USC-DTIN/isi/reliefWebProcessed/'
    dwFeature()
    modelType = sys.argv[2] # 'unsupervised'  # 'unsupervised' #
    numClu = sys.argv[3] # 30  #
    res = cluster(modelType, numClu)
    tsneVis(res)