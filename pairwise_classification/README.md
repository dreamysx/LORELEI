# Pairwise Classification
Pairwise Classification of text documents as part of experiment results of Project **LORELEI**.

## Usage
* **Five** callable scripts are in this directory, each can be called simply by `$ python script_name.py`.
* `data_prep.py` can be called to preprocess raw data. It will transfer multiple raw json files into single one json file, of which the key is document file id, the value is useful text content like entities or word cloud of one document.
* `tfidf.py` is for calculating tf-idf of each document in document corpus. It takes output of `data_prep.py` as input, and return tfidf of each document, in both dense and sparse formats.
* `multi_trial_exp.py`, `single_trial_exp.py` and `one_part_exp.py` can be called to run classification on text documents, then evaluate performance of selected classification algorithm using f measure. Algorithm can be changed by modifying code in certain part. Our default algorithm is Random Forest Classifier, since it gave best f one score during our past experiments. Text documents are represented as feature vectors. Difference between these three scripts are:
	1. `multi_trial_exp.py` must run the whole Sampling-Training-Testing procedure more than one time in one experiment;
	2. `single_trial_exp.py` must run the whole procedure only once in one experiment;
	3. `one_part_exp.py` doesn't sample. Instead, it run Trainging on one part of data, while Tesing on all the other parts of data.
* `MTokenizer.py` is used in refining input text content, credit to [Mayank Kejriwal](http://usc-isi-i2.github.io/kejriwal/).
