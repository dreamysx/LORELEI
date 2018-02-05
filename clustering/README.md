# Clustering
Text document clustring pipeline built on part of experiment results of Project **LORELEI**.

## Usage
* `clustering.py` could be called to do text document clustring based on [deepwalk](https://github.com/phanein/deepwalk) embedding algorithm generated feature vector.
* `clustering.py` takes three arguments, in the order of:
	1. File path of documents
	2. Type of model for clustering (supervised/ unsupervised)
	3. Number of clusters

	`$ python clustering.py file_path unsupervised 30`
* `MTokenizer.py` is used in refining input text content, all credit to [Mayank Kejriwal](http://usc-isi-i2.github.io/kejriwal/).
