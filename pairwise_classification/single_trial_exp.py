from __future__ import division
import json
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import time
import random
from sklearn.linear_model import LogisticRegressionCV
import pandas as pd
from sklearn.metrics import precision_recall_curve

print 'IMPORTANT: experiment can be modified by changing parameter combinations in main function!'
print 'loading data...'
part1_pos_walk_200 = json.loads(open("dedup_part1_pos_200_walk.json").read())
part2_pos_walk_200 = json.loads(open("dedup_part2_pos_200_walk.json").read())
part3_pos_walk_200 = json.loads(open("dedup_part3_pos_200_walk.json").read())
part4_pos_walk_200 = json.loads(open("dedup_part4_pos_200_walk.json").read())
part5_pos_walk_200 = json.loads(open("dedup_part5_pos_200_walk.json").read())
part1_pos_10 = json.loads(open("dedup_part1_pos_10.json").read())
part2_pos_10 = json.loads(open("dedup_part2_pos_10.json").read())
part3_pos_10 = json.loads(open("dedup_part3_pos_10.json").read())
part4_pos_10 = json.loads(open("dedup_part4_pos_10.json").read())
part5_pos_10 = json.loads(open("dedup_part5_pos_10.json").read())

global_neg_walk = json.loads(open("dedup_global_neg_200_walk.json").read())
global_neg_10 = json.loads(open("dedup_global_neg_10.json").read())
global_pos_walk = json.loads(open("dedup_global_pos_200_walk.json").read())
global_pos_10 = json.loads(open("dedup_global_pos_10.json").read())

print 'defining function...'
def takingSamples(alist, num=0, portion=0):
    assert ((num > 0 and portion == 0) or (num == 0 and portion > 0)), "should offer only one method, num or portion"
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

def oneTrial(train_ratio=1,
             pos_training_dataset=None,
             pos_testing_dataset=None,
             neg_dataset=None,
             scoring="f1"):

    num_pos_sample = len(pos_training_dataset) * train_ratio
    num_neg_sample = num_pos_sample

    # take sample of num_pos_sample number of positive examples
    (posPicked, posNotPicked) = takingSamples(pos_training_dataset, num=num_pos_sample)
    (negPicked, negNotPicked) = takingSamples(neg_dataset, num=num_neg_sample)

    # create train_X, train_y
    train_X = pd.DataFrame(posPicked + negPicked)
    train_y = np.array([1 for i in range(len(posPicked))] + [0 for i in range(len(negPicked))])

    # create test_X and test_y
    if (pos_testing_dataset == None):
        test_X = pd.DataFrame(posNotPicked + negNotPicked)
        test_y = np.array([1 for i in range(len(posNotPicked))] + [0 for i in range(len(negNotPicked))])
    else:
        test_X = pd.DataFrame(pos_testing_dataset + negNotPicked)
        test_y = np.array([1 for i in range(len(pos_testing_dataset))] + [0 for i in range(len(negNotPicked))])


    # train and test the model
    reg = LogisticRegressionCV(scoring=scoring)
    LogModel = reg.fit(train_X, train_y)
    # return LogModel
    y_predlog = LogModel.predict_proba(test_X)
    y_predlog_1 = y_predlog[:, 1]

    prec, rec, thresholds = precision_recall_curve(test_y, y_predlog_1)
    fone = []
    for i in range(len(prec)):
        fone.append(2*prec[i]*rec[i]/(prec[i]+rec[i]))

    return (fone, LogModel)

if __name__ == "__main__":
    print 'Start training...'
    print 'part2_walk'
    (result_walk, model_walk) = oneTrial(pos_training_dataset=part2_pos_walk_200,
                                         pos_testing_dataset=part1_pos_walk_200 + part5_pos_walk_200 + part3_pos_walk_200 + part4_pos_walk_200,
                                         neg_dataset=global_neg_walk)

    print 'Best f1: %f' % max(result_walk)