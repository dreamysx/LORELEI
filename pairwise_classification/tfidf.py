import json
import math
from scipy.sparse import csc_matrix
import numpy as np

# input file should be like {doc id: word id list} in json

def gen_tfidf(file):
    doc_num = len(file)
    term_num = 0
    tf_idf = {}
    tf = {}
    idf = {}
    tfidf_in_sparse_vector = {}
    pair_wise_sim = {}
    # print doc_num

    # compute normalized tf, get term list
    for idx in file:
        term_num_in_doc = float(len(file[idx]))
        doc_tf = {}
        for term in file[idx]:
            doc_tf[term] = doc_tf.get(term, 0.0) + 1.0
            idf[term] = 0.0  # get initialized idf list
        for temp in doc_tf:
            doc_tf[temp] = doc_tf[temp] / term_num_in_doc
        tf[idx] = doc_tf
    term_num = len(idf)
    # print term_num

    # compute idf
    for term in idf:
        for idx in file:
            if term in file[idx]:
                idf[term] += 1
    for term in idf:
        idf[term] = math.log(float(doc_num) / idf[term])

    for idx in file:
        for term in file[idx]:
            tf[idx][term] *= idf[term]
        tf_idf[idx] = tf[idx]
    # print tf_idf
    # print len(tf_idf)

    docList = json.loads(open('gt_docIdList.json').read())
    wordList = json.loads(open('gt_wordIdList.json').read())

    idTfidf = {}
    for file in tf_idf:
        value = {}
        for word in tf_idf[file]:
            value[wordList[word]] = tf_idf[file][word]
        idTfidf[docList[file]] = value
    # print idTfidf

    vecTfidf = {}
    for file in idTfidf:
        row = np.zeros(len(idTfidf[file]))
        col = idTfidf[file].keys()
        val = idTfidf[file].values()
        vec = csc_matrix((np.array(val), (np.array(row), np.array(col))), shape=(1, term_num))
        vecTfidf[file] = vec.todense()
    # print vecTfidf

    return (idTfidf, vecTfidf)

if __name__ == "__main__":
    file = json.loads(open('gt_wordList.json').read())  # 272 docs
    (tfidf, tfidf_in_sparse_mat) = gen_tfidf(file)