import json

groundTruth = json.loads(open('implicit_entity/groundTruth.json').read())
ranking_cooccur = json.loads(open('implicit_entity/ranking_top100_cooccur_baseline.json').read())
ranking_cooccur_cos = json.loads(open('implicit_entity/ranking_top100_cooccur_baseline_cosine.json').read())
ranking_mean = json.loads(open('implicit_entity/ranking_top100_dot_product_mean.json').read())
ranking_doc = json.loads(open('implicit_entity/ranking_top100_dot_product_doc.json').read())
# print len(groundtruth)
# print len(ranking_cooccur)

def overlap(gt, ranking):
    for ent in ranking:
        if ent in gt:
            return True
    return False

def hit_at_k(groundtruth, ranking_res):
    total_doc = len(ranking_res)
    rel_doc = 0
    for doc in ranking_res:
        if overlap(ranking_res[doc], groundtruth[doc]):
            rel_doc += 1
    return float(rel_doc) / total_doc

if __name__=='__main__':
    print 'Using cooccurred entities, %.2f%% documents have at least one entity correct.' % (hit_at_k(groundTruth, ranking_cooccur)*100)
    print 'Using cooccurred entities cosine, %.2f%% documents have at least one entity correct.' % (hit_at_k(groundTruth, ranking_cooccur_cos)*100)
    print 'Using mean entities reference, %.2f%% documents have at least one entity correct.' % (hit_at_k(groundTruth, ranking_mean)*100)
    print 'Using document reference, %.2f%% documents have at least one entity correct.' % (hit_at_k(groundTruth, ranking_doc)*100)
