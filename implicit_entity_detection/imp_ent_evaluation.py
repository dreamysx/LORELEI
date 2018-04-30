import json
import numpy as np
import math

groundtruth = json.loads(open('implicit_entity/groundTruth.json').read())
ranking_mean = json.loads(open('implicit_entity/ranking_top_dot_product_mean.json').read())
ranking_doc = json.loads(open('implicit_entity/ranking_top_dot_product_doc.json').read())
ranking_cooccur = json.loads(open('implicit_entity/ranking_cooccur_baseline.json').read())
ranking_cooccur_cos = json.loads(open('implicit_entity/ranking_top100_cooccur_baseline_cosine.json').read())
training = json.loads(open('implicit_entity/trainingDocEnt.json').read())

def _compute_ndcg(groundTruth, ranking_res):
    dcg = 0.0
    for c in groundTruth:
        if c in ranking_res:
            correct_ind_rank = ranking_res.index(c)
            dcg += 1.0 / math.log(correct_ind_rank + 2, 2)
        else:
            continue
    idcg = 0.0
    for i in range(0, len(groundTruth)):
        idcg += 1.0 / math.log(i + 2, 2)
    if int(dcg / idcg) > 1 or int(dcg / idcg) < 0:
        # print 'ndcg for current doc: %f' % (dcg / idcg)
        raise Exception
    return dcg / idcg

def ndcg_score(ranking_res, groundtruth):
    doc_len = len(ranking_res)
    overall_ndcg = 0.0
    for doc in ranking_res:
        overall_ndcg += _compute_ndcg(groundtruth[doc], ranking_res[doc])
    avg_ndcg = overall_ndcg / doc_len
    return avg_ndcg

if __name__ == "__main__":
    print 'Using cooccurred entities, ndcg score is %f' % ndcg_score(ranking_cooccur, groundtruth)
    print 'Using cooccurred entities cosine, ndcg score is %f' % ndcg_score(ranking_cooccur_cos, groundtruth)
    print 'Using mean entities reference, ndcg score is %f' % ndcg_score(ranking_mean, groundtruth)
    print 'Using document reference, ndcg score is %f' % ndcg_score(ranking_doc, groundtruth)

