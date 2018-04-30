# Clustering
Text document clustring part of Project **LORELEI**.

## Usage
* `clustering.py` can be called to do text document clustring based on [deepwalk](https://github.com/phanein/deepwalk) embedding algorithm generating feature vector.
* `clustering.py` takes four arguments, in the order of:
	1. File path of documents
	2. Type of model for clustering (supervised/ unsupervised)
	3. Number of clusters
	4. Plot clustering using [Tsne](https://lvdmaaten.github.io/tsne/) or not

	For example: `$ python clustering.py file_path unsupervised 30 False`
* `MTokenizer.py` is used in refining input text content, credit to [Mayank Kejriwal](http://usc-isi-i2.github.io/kejriwal/).
* `tfidf_clustering.py`, `unsupervised_clustering.py` and `supervised_clustering.py` can be called to run clustering on text documents, then evaluate performance of selected clustering algorithm using purity and f measure. Algorithm can be changed by modifying code in certain part. Text documents are represented as feature vectors. Difference between these three scripts are:
	1. `tfidf_clustering.py` runs unsupervised clustering on dataset based on document tfidf, affinity of clustering algorithm is consine;
	2. `unsupervised_clustering.py` runs unsupervised clustering on dataset based on other typical document representations, affinity of clustering algorithm is consine;
	3. `supervised_clustering.py` runs supervised clustering on dataset by using precomputed affinity matrix.