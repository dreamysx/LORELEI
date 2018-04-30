# used to transfer raw data into {doc id: word list} with uuidAndIntMapping.json
import json
import os
import MTokenizer

def lower(list):
    if list != None:
        res = []
        for temp in list:
            res.append(temp.lower())
        return res

def prep(ref, files):
    docE = {}
    docRE = {}
    docW = {}
    docRWC = {}
    for file_name in files:
        file = json.loads(open('/Users/dreamysx/Documents/USC-DTIN/isi/reliefWebProcessed/' + file_name).read())
        if file['loreleiJSONMapping']['wordcloud'] != None:
            docW[ref[file['uuid']]] = lower(file['loreleiJSONMapping']['wordcloud'])
            refinedWC = []
            for worcloud in lower(file['loreleiJSONMapping']['wordcloud']):
                refinedWC += MTokenizer.MTokenizer.tokenize_string(worcloud)
            docRWC[ref[file['uuid']]] = refinedWC
        if file['situationFrame']['entities'] != None:
            docE[ref[file['uuid']]] = lower(file['situationFrame']['entities'])
            refinedE = []
            for entity in lower(file['situationFrame']['entities']):
                refinedE += MTokenizer.MTokenizer.tokenize_string(entity)
            docRE[ref[file['uuid']]] = refinedE

    return (docE, docRE, docW, docRWC)

if __name__ == "__main__":
    ref = json.loads(open('uuidAndIntMapping.json').read())
    name_list = os.listdir('/Users/dreamysx/Documents/USC-DTIN/isi/reliefWebProcessed/')
    (docE, docRE, docW, docRWC) = prep(ref, name_list)
    with open('idxAndEntities.json', 'w') as f:
        f.write(json.dumps(docE))
    with open('idxAndRefinedEntities.json', 'w') as f:
        f.write(json.dumps(docRE))
    with open('idxAndWordcloud.json', 'w') as f:
        f.write(json.dumps(docW))
    with open('idxAndRefinedWordcloud.json', 'w') as f:
        f.write(json.dumps(docRWC))