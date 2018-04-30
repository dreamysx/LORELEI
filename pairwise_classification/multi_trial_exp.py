from __future__ import division
import json
import numpy as np
import time
import random
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.linear_model import LogisticRegressionCV
import pandas as pd
import operator

# load basic global data
print 'IMPORTANT: experiment can be modified by changing parameter combinations in main function!'
print 'preparing data...'
# global data
global_pos_10 = json.loads(open('new_dedup_global_pos_10.json').read()) # 10-feat
global_neg_10 = json.loads(open('new_dedup_global_neg_10.json').read())
global_pos_200_embed = json.loads(open('new_dedup_global_pos_200_embed.json').read()) # doc vec
global_neg_200_embed = json.loads(open('new_dedup_global_neg_200_embed.json').read())
global_neg_200_walk = json.loads(open('new_dedup_global_neg_200_walk.json').read()) # random walk (name diff)
global_pos_200_walk = json.loads(open('new_dedup_global_pos_200_walk.json').read())

def combineData(source1_pos=None,
                source1_neg=None,
                source2_pos=None,
                source2_neg=None,
                source3_pos=None,
                source3_neg=None):

    # assert (len(source1_pos) == len(source2_pos) == len(source3_pos)), "pos should be equal length"
    # assert (len(source1_neg) == len(source2_neg) == len(source3_neg)), "neg should be equal length"

    comb_pos = []
    comb_neg = []

    if source3_pos == None: # only combine two datasets
        for i in range(len(source1_pos)):
            comb_pos.append(source1_pos[i] + source2_pos[i])

        if source1_neg != None:
            for i in range(len(source1_neg)):
                comb_neg.append(source1_neg[i] + source2_neg[i])
    else:
        for i in range(len(source1_pos)):
            comb_pos.append(source1_pos[i] + source2_pos[i] + source3_pos[i])

        if source1_neg != None:
            for i in range(len(source1_neg)):
                comb_neg.append(source1_neg[i] + source2_neg[i] + source3_neg[i])

    if len(comb_neg) == 0:
        return comb_pos
    else:
        return (comb_pos, comb_neg)


# combination of global data
(combPos_10_walk, combNeg_10_walk) = combineData(source1_pos=global_pos_10,
                                 source1_neg=global_neg_10,
                                 source2_pos=global_pos_200_walk,
                                 source2_neg=global_neg_200_walk,
                                 source3_pos=None,
                                 source3_neg=None)
(combPos_10_walk_dv, combNeg_10_walk_dv) = combineData(source1_pos=global_pos_10,
                                 source1_neg=global_neg_10,
                                 source2_pos=global_pos_200_walk,
                                 source2_neg=global_neg_200_walk,
                                 source3_pos=global_pos_200_embed,
                                 source3_neg=global_neg_200_embed)
# print len(combPos_10_walk), len(combPos_10_walk_dv), len(combNeg_10_walk),  len(combNeg_10_walk_dv)

print 'done preparing data'

##################################################### experiment phase #####################################################
print 'preparing function...'
# functions
# general function for taking samples from a list
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

    # usage e.g.
    # (listPicked, listNotPicked) = takingSamples([1,2,3,4,5,6], num=4)
    # (listPicked, listNotPicked) = takingSamples([[1,2],[2,5],[3,7],[4,6],[5,5],[6,1]], num=4)
    # print listPicked
    # print listNotPicked

# averaging the results from trials
def avgProcess(trialsAns):
    trialsAns_np = np.array(trialsAns)
    num_trial = len(trialsAns_np) # 10

    # place holder for average threshold, precision, recall, f1
    avg_thres = np.array([0.0 for i in range(len(trialsAns_np[0]))])
    avg_prec = np.array([0.0 for i in range(len(trialsAns_np[0]))])
    avg_rec = np.array([0.0 for i in range(len(trialsAns_np[0]))])
    avg_f1 = np.array([0.0 for i in range(len(trialsAns_np[0]))])

    for i in range(num_trial):
        tmp = np.array(trialsAns_np[i])
        avg_thres += tmp[:, 0] # the 0th column
        avg_prec += tmp[:, 1]
        avg_rec += tmp[:, 2]
        avg_f1 += tmp[:, 3]

    avg_thres = avg_thres / float(num_trial)
    avg_prec = avg_prec / float(num_trial)
    avg_rec = avg_rec / float(num_trial)
    avg_f1 = avg_f1 / float(num_trial)

    avg_thres = list(avg_thres)
    avg_prec = list(avg_prec)
    avg_rec = list(avg_rec)
    avg_f1 = list(avg_f1)

    return (avg_thres, avg_prec, avg_rec, avg_f1)

# input should be lists of 10 or 210 dimensions
def oneTrialWithCertainTrainSize(num_pos_sample=50,
                                 neg_pos_ratio=1,
                                 pos_dataset=None,
                                 neg_dataset=None):

    assert(type(pos_dataset) == list and type(neg_dataset) == list), "input datasets should be lists"

    num_neg_sample = int(num_pos_sample * neg_pos_ratio)

    # take sample of num_pos_sample number of positive examples
    (posPicked, posNotPicked) = takingSamples(pos_dataset, num=num_pos_sample)
    (negPicked, negNotPicked) = takingSamples(neg_dataset, num=num_neg_sample)

    # create train_X, train_y
    train_X = pd.DataFrame(posPicked + negPicked)
    train_y = np.array([1 for i in range(len(posPicked))] + [0 for i in range(len(negPicked))])

    # create test_X and test_y
    test_X = pd.DataFrame(posNotPicked + negNotPicked)
    test_y = np.array([1 for i in range(len(posNotPicked))] + [0 for i in range(len(negNotPicked))])


    # train and test the model
    # reg = RFC()
    reg = RFC(n_estimators=200, max_features='log2')
    # reg = LogisticRegressionCV(scoring='f1')
    model = reg.fit(train_X, train_y)
    y_predlog = model.predict_proba(test_X)
    y_predlog_1 = y_predlog[:, 1]
    # print "coef:", reg.coef_
    # print "intercept:", reg.intercept_

    # pred_combine sorted
    pred_combine = []
    for i in range(len(test_y)):
        pred_combine.append((y_predlog_1[i], test_y[i]))

    pred_combine = sorted(pred_combine, key=operator.itemgetter(0))

    # create an array of 0.1:0.01:0.99
    thres_new = []
    initial = 0.1
    while initial <= 0.99:
        thres_new.append(initial)
        initial += 0.01
        initial = round(initial, 2)

    # generate "threshold, prec, rec, f1" list
    # test_y is truth, y_predlog_1 is prob of being 1
    result = []
    item_index = 0

    FN_accu = 0
    TN_accu = 0
    TP_accu = list(test_y).count(1)
    FP_accu = list(test_y).count(0)

    for i in thres_new:  # i is [0.1:0.01:0.99]
        if (item_index < len(pred_combine)):
            while pred_combine[item_index][0] < i:
                if pred_combine[item_index][1] == 1:  # this item actually 1, predict as 0
                    FN_accu += 1
                    TP_accu -= 1
                else:  # this item is actually 0, predict as 0, pred_combine[item_index][1] == 0
                    TN_accu += 1
                    FP_accu -= 1
                item_index += 1
                if (item_index == len(pred_combine)): break

        # print "th: " + str(i) + ", TP: " + str(TP_accu) + ", FP: " + str(FP_accu) + ", FN: " + str(FN_accu) + ", TN: " + str(TN_accu)

        if (TP_accu == 0):
            preci = 0
        else:
            preci = float(TP_accu) / (TP_accu + FP_accu)

        if (TP_accu == 0):
            recal = 0
        else:
            recal = float(TP_accu) / (FN_accu + TP_accu)

        if (2 * preci * recal == 0):
            fone = 0
        else:
            fone = 2 * preci * recal / (preci + recal)

        result.append([i, preci, recal, fone])

    return result # 90

    # outArr = oneTrialWithCertainTrainSize(num_pos_sample=60, pos_neg_ratio=1, pos_dataset=global_pos_10, neg_dataset=global_neg_10)
    # print "finish"

# trialsWithVariedTrainSize
def trialsWithVariedTrainSize(num_pos_sample=50,
                              num_pos_sample_cap=1500,
                              neg_pos_ratio=1,
                              pos_dataset=None,
                              neg_dataset=None,
                              num_trial=10,
                              save=False,
                              saveName="0"):

    generalResults = []
    generalResultsPosNumRef = []
    generalStdDev = []

    while num_pos_sample <= num_pos_sample_cap:
        trialsAns = []

        # for each num_pos_sample, perform 10 trials
        for trialsCount in range(num_trial):
            # one single trial
            outArr = oneTrialWithCertainTrainSize(num_pos_sample=num_pos_sample, neg_pos_ratio=neg_pos_ratio, pos_dataset=pos_dataset, neg_dataset=neg_dataset)
            # put outArr together
            trialsAns.append(outArr) # outArr = [threshold, prec, rec, f1tmp]

            print "trial #" + str(trialsCount + 1)  + " results collection finished!"

        # with open('trialsAns.json', 'w') as f:
        #     json.dump(trialsAns, f)

        print str(num_pos_sample) + " all trials finished!"

        # calc std dev of max f1 based on trialsAns
        # stdArray = []
        # for e in range(len(trialsAns[0])):
        #     tmpArr = []
        #     for k in trialsAns:
        #         tmpArr.append(k[e][3])
        #     stdArray.append(np.std(np.array(tmpArr)))
        #
        # stddev = np.average(stdArray)
        # generalStdDev.append(stddev)
        #
        if save == True:
            fileName = "rawResults_" + saveName + ".json"
            with open(fileName, 'w') as f: json.dump(trialsAns, f)

        (avg_thres, avg_prec, avg_rec, avg_f1) = avgProcess(trialsAns)

        #
        generalResults.append([avg_thres, avg_prec, avg_rec, avg_f1])
        generalResultsPosNumRef.append(num_pos_sample)

        # print results for each trial
        targ = generalResults
        index = targ[0][3].index(max(targ[0][3]))
        for ntrial in range(len(trialsAns)):
            fone = trialsAns[ntrial][index][3]
            prec = trialsAns[ntrial][index][1]
            rec = trialsAns[ntrial][index][2]
            print "For trial#" + str(ntrial)
            print "f1: %.4f" %fone + ", prec: %.4f" %prec + ", rec: %.4f" %rec


        #
        print str(num_pos_sample) + " positive finished!"

        num_pos_sample += 50

        # if num_pos_sample < 200: num_pos_sample += 10
        # elif num_pos_sample < 500: num_pos_sample += 50
        # else: num_pos_sample += 100

    # return (generalResults, generalStdDev, generalResultsPosNumRef)
    return (generalResults, generalResultsPosNumRef)
    # return None

    # usage e.g. (generalResults, generalStdDev, generalResultsPosNumRef) = trialsWithVariedTrainSize(num_pos_sample=50, num_pos_sample_cap=1500, pos_neg_ratio=1, pos_dataset=global_pos_10, neg_dataset=global_neg_10)
print 'done preparing function'

if __name__ == "__main__":
    print 'starting training...'
    print 'global sample 30% train with 10_walk_dv...'
    # sample 6039, 10_walk
    (generalResults_10_walk, generalResultsPosNumRef) = trialsWithVariedTrainSize(num_pos_sample=6039,
                                                                                  num_pos_sample_cap=6039,
                                                                                  neg_pos_ratio=1,
                                                                                  pos_dataset=combPos_10_walk_dv,
                                                                                  neg_dataset=combNeg_10_walk_dv)

    targ = generalResults_10_walk
    max_f1 = max(targ[0][3])  # 0.5885
    index_max_f1 = targ[0][3].index(max(targ[0][3]))  # 73
    prec_at_max_f1 = targ[0][1][index_max_f1]  # 0.5536
    rec_at_max_f1 = targ[0][2][index_max_f1]  # 0.6204

    print "index: %d, f1: %f, prec: %f, rec: %f" % (index_max_f1, round(max_f1, 4), round(prec_at_max_f1, 4), round(rec_at_max_f1, 4))
    print 'done!'