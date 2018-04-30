import json
import operator
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

training = json.loads(open('implicit_entity/trainingDocEnt.json').read())
print len(training)
testing = json.loads(open('implicit_entity/testingDocEnt.json').read())
print len(testing)
docEmb = json.loads(open('implicit_entity/docDeepwalk.json').read())
entEmb = json.loads(open('implicit_entity/entDeepwalk.json').read())

def get_occur_list():
    ent_cooccur = {}
    ent_cooccur_dup = {}
    for doc in training:
        for ent in training[doc]:
            if ent not in ent_cooccur_dup:
                ent_cooccur_dup[ent] = []
            ent_cooccur_dup[ent].extend(training[doc])
    for ent in ent_cooccur_dup:
        temp = []
        for ent_temp in ent_cooccur_dup[ent]:
            if ent_temp != ent:
                temp.append(ent_temp)
        ent_cooccur[ent] = list(set(temp))
    # with open('implicit_entity/cooccured_entity.json', 'w') as f:
    #     f.write(json.dumps(ent_cooccur))
    return ent_cooccur


def get_ranking_res(ent_cooccur, topk, sign):
    cooccur_baseline = {}
    for doc in testing:
        docFeature = docEmb[doc]
        cooccur_ranking = {}
        check_ents = training[doc]
        for ent in check_ents:
            for co_ent in ent_cooccur[ent]:
                if sign == 'document count':
                    if co_ent not in cooccur_ranking:
                        cooccur_ranking[co_ent] = 0
                    cooccur_ranking[co_ent] += 1
                elif sign == 'cosine similarity':
                    if co_ent not in cooccur_ranking:
                        entFeature = entEmb[co_ent]
                        docFeature = np.array(docFeature)
                        entFeature = np.array(entFeature)
                        docFeature = docFeature.reshape(1, -1)
                        entFeature = entFeature.reshape(1, -1)
                        cooccur_ranking[co_ent] = cosine_similarity(docFeature, entFeature)[0][0]

        ranking_res = sorted(cooccur_ranking.items(), key=operator.itemgetter(1))
        ranking_res.reverse()
        print ranking_res
        detected = []
        for ent in ranking_res:
            detected.append(ent[0])
            if len(detected) == topk:
                break
        cooccur_baseline[doc] = detected

    with open('implicit_entity/ranking_top100_cooccur_baseline_cosine.json', 'w') as f:
        f.write(json.dumps(cooccur_baseline))
    # with open('implicit_entity/ranking_top100_cooccur_baseline.json', 'w') as f:
    #     f.write(json.dumps(cooccur_baseline))
    return cooccur_baseline

if __name__=='__main__':
    topk = 100 # any number
    method = 'cosine similarity' # 'document count' #
    ent_cooccur = json.loads(open('implicit_entity/cooccured_entity.json').read())
    get_ranking_res(ent_cooccur, topk, method)