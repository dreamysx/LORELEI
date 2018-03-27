import json
import os
import sys
import time
import random
import numpy as np
import math
import operator
from MTokenizer import MTokenizer

def takingSamples(alist, portion=0):
    seed = int(round(time.time() * 1000)) % 100000000
    random.seed(seed)
    length_of_list = len(alist)
    listPicked = []
    listNotPicked = []

    if portion > 0:
        num = int(math.floor(length_of_list * portion))
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
                refinedE += MTokenizer.tokenize_string(entity)
            docRE[file['uuid']] = list(set(refinedE))
    print 'Finish cleaning data...'
    return docRE

################ three modules for implicit entity detection ##############
def buildEdgeList(name_list, threshold):
    edges = []
    frequent_entity_edges = []
    intRef = {}
    iddocRef = {}
    docidRef = {}
    identRef = {}
    entidRef = {}
    entCount = {}
    ent_doc = {}
    index = 0
    frequentEntityRef = {}
    frequentDocRef = {}
    textDict = textClean(name_list)
    for id in textDict:
        intRef[id] = str(index)
        iddocRef[str(index)] = id
        docidRef[id] = str(index)
        index += 1
        for word in textDict[id]:
            if word not in entCount:
                entCount[word] = 0
            entCount[word] += 1
            if word not in intRef:
                intRef[word] = str(index)
                identRef[str(index)] = word
                entidRef[word] = str(index)
                index += 1
    for id in textDict:
        for word in textDict[id]:
            edges.append((intRef[id], intRef[word]))
            if entCount[word] >= threshold:
                frequent_entity_edges.append((intRef[id], intRef[word]))
                frequentEntityRef[word] = entidRef[word]
                frequentDocRef[id] = docidRef[id]
                if word not in ent_doc:
                    ent_doc[word] = []
                ent_doc[word].append(id)

    print 'Number of nodes in graph: %d' % len(intRef)
    print 'Number of documents:%d' % len(iddocRef)
    print 'Number of entities:%d' % (len(entidRef))
    print 'Number of frequent documents:%d' % len(frequentDocRef)
    print 'Number of frequent entities:%d' % (len(frequentEntityRef))
    print 'Number of entities reduced:%d' % (len(entidRef)-len(frequentEntityRef))
    with open('implicit_entity/graphRef.json', 'w') as f:
        f.write(json.dumps(intRef))
    with open('implicit_entity/graphIddocRef.json', 'w') as f:
        f.write(json.dumps(iddocRef))
    with open('implicit_entity/graphDocidRef.json', 'w') as f:
        f.write(json.dumps(docidRef))
    with open('implicit_entity/graphIdentRef.json', 'w') as f:
        f.write(json.dumps(identRef))
    with open('implicit_entity/graphEntidRef.json', 'w') as f:
        f.write(json.dumps(entidRef))
    with open('implicit_entity/entityCount.json', 'w') as f:
        f.write(json.dumps(entCount))
    with open('implicit_entity/frequentEntityDocRef.json', 'w') as f:
        f.write(json.dumps(ent_doc))

    print 'Finish getting edgelist...'
    print 'Number of edges before preprocessing:%d' % len(edges)
    print 'Number of edges after preprocessing:%d' % len(frequent_entity_edges)
    return (frequent_entity_edges, edges)

def dwFeature(portion):
    entityDocRef = json.loads(open('implicit_entity/frequentEntityDocRef.json').read())
    graphRef = json.loads(open('implicit_entity/graphRef.json').read())
    training_edgeList = []
    testing_edgeList = []
    training_entity_doc_ref = {}
    testing_entity_doc_ref = {}
    for entity in entityDocRef:
        docs = entityDocRef[entity]
        (training, testing) = takingSamples(docs, portion)
        training_entity_doc_ref[entity] = training
        testing_entity_doc_ref[entity] = testing
        for doc in training:
            training_edgeList.append((graphRef[entity], graphRef[doc]))
        for doc in testing:
            testing_edgeList.append((graphRef[entity], graphRef[doc]))

    with open('implicit_entity/training.json', 'w') as f:
        f.write(json.dumps(training_entity_doc_ref))
    with open('implicit_entity/testing.json', 'w') as f:
        f.write(json.dumps(testing_entity_doc_ref))

    with open('implicit_entity/graph.edgelist', 'w') as f:
        for pair in training_edgeList:
            f.write(str(pair[0]))
            f.write('\t')
            f.write(str(pair[1]))
            f.write('\n')
            f.write(str(pair[1]))
            f.write('\t')
            f.write(str(pair[0]))
            f.write('\n')
    # with open('implicit_entity/graph.adjlist', 'w') as f:
    #     for line in adjsList:
    #         for item in line:
    #             f.write(str(item))
    #             f.write(' ')
    #         f.write('\n')

    print 'Start embedding algorithm... This may take up to hours based on the size of documents'
    os.system('deepwalk --input implicit_entity/graph.edgelist --representation-size 100 --walk-length 100  --output implicit_entity/graph.embeddings')
    print 'Finish getting graph embeddings...'
    return (training_edgeList, testing_edgeList)

def embedding_process(embedding_res):
    iddocRef = json.loads(open('implicit_entity/graphIddocRef.json').read())
    identRef = json.loads(open('implicit_entity/graphIdentRef.json').read())

    docEmb = {}
    entEmb = {}
    for line in embedding_res:
        lineid = line.split()[0]
        if lineid in iddocRef:
            docEmb[iddocRef[lineid]] = map(eval, line.split()[1:])
        if lineid in identRef:
            entEmb[identRef[lineid]] = map(eval, line.split()[1:])
    with open('implicit_entity/docDeepwalk.json', 'w') as f:
        f.write(json.dumps(docEmb))
    with open('implicit_entity/entDeepwalk.json', 'w') as f:
        f.write(json.dumps(entEmb))
    return (docEmb, entEmb)

def doc_entity_reverse(entity_doc):
    doc_entity = {}
    for entity in entity_doc:
        for doc in entity_doc[entity]:
            if doc not in doc_entity:
                doc_entity[doc] = []
            doc_entity[doc].append(entity)
    return doc_entity

def dot_product_top(docEmb, entEmb, doc, known_entities, candidate_entities, top):
    top_res = []
    temp_res = {}
    doc_feature = docEmb[doc]
    total_dis = 0
    total_count = 0
    for entity in known_entities:
        entity_feature = entEmb[entity]
        total_dis += np.dot(doc_feature, entity_feature)
        total_count += 1
    avg_dis = total_dis/ total_count
    for entity in candidate_entities:
        entity_feature = entEmb[entity]
        temp_res[entity] = abs(np.dot(doc_feature, entity_feature) - avg_dis)
    ranking_res = sorted(temp_res.items(), key=operator.itemgetter(1))
    with open('implicit_entity/ranking_res/'+doc+'.json', 'w') as f:
        f.write(json.dumps(ranking_res))
    for pair in ranking_res[:top]:
        top_res.append(pair[0])
    return top_res

def ranking(docEmb, entEmb, method='dot_product_mean'):
    training = json.loads(open('implicit_entity/training.json').read())
    testing = json.loads(open('implicit_entity/testing.json').read())
    training_doc_entities = doc_entity_reverse(training)
    doc_in_training = []
    # remove doc in testing, but not in training
    for entity in training:
        doc_in_training.extend(training[entity])
    doc_in_training = list(set(doc_in_training))
    for entity in testing:
        for doc in testing[entity]:
            if doc not in doc_in_training:
                testing[entity].remove(doc)
    testing_doc_entities = doc_entity_reverse(testing)
    with open('implicit_entity/trainingDocEnt.json', 'w') as f:
        f.write(json.dumps(training_doc_entities))
    with open('implicit_entity/testingDocEnt.json', 'w') as f:
        f.write(json.dumps(testing_doc_entities))

    entities = training.keys()
    doc_entities_ranking = {}
    for doc in testing_doc_entities:
        left = 10 - len(training_doc_entities[doc])
        if left <= 0:
            doc_entities_ranking[doc] = []
            continue
        else:
            known_entities = training_doc_entities[doc]
            cand_entities = []
            for entity in entities:
                if entity not in known_entities:
                    cand_entities.append(entity)
            if method == 'dot_product_mean':
                doc_entities_ranking[doc] = dot_product_top(docEmb, entEmb, doc, known_entities, cand_entities, left)
    with open('implicit_entity/ranking_top.json', 'w') as f:
        f.write(json.dumps(doc_entities_ranking))

    return doc_entities_ranking

def exe(file_path, threshold=3, training_portion=0.9):
    name_list = os.listdir(file_path)
    buildEdgeList(name_list, threshold)
    dwFeature(training_portion)  # 0.9 then round down
    features = open('implicit_entity/graph.embeddings').readlines()[1:]
    (docEmb, entEmb) = embedding_process(features)
    ranking(docEmb, entEmb)

if __name__ == "__main__":
    path = sys.argv[1] # '/Users/dreamysx/Documents/USC-DTIN/isi/reliefWebProcessed/'  #
    threshold = eval(sys.argv[2]) # 3 #
    training_portion = eval(sys.argv[3]) # 0.9 #
    exe(path, threshold, training_portion)