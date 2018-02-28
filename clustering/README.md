# Clustering
Text document clustring pipeline built on part of experiment results of Project **LORELEI**.

## Usage
* `clustering.py` can be called to do text document clustring based on [deepwalk](https://github.com/phanein/deepwalk) embedding algorithm generating feature vector.
* `clustering.py` takes four arguments, in the order of:
	1. File path of documents
	2. Type of model for clustering (supervised/ unsupervised)
	3. Number of clusters
	4. Plot clustering using [Tsne](https://lvdmaaten.github.io/tsne/) or not

	For example: `$ python clustering.py file_path unsupervised 30 False`
* `MTokenizer.py` is used in refining input text content, credit to [Mayank Kejriwal](http://usc-isi-i2.github.io/kejriwal/).
