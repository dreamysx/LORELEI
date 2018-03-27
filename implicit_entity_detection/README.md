# Implicit Entity Detection
Implicit entity detection built on part of experiment results of Project **LORELEI**.

## Usage
* `imp_ent_detection.py` can be called to evaluate possibility ranking of all entities in dataset being the entity of a certain document. This evaluation is done based on known entities of the document, as well as the feature vectors of documents and entities in dataset.

* `imp_ent_detection.py` takes four arguments, in the order of:
	1. File path of documents
	2. Threshold for pruning entities(3 by default)
	3. Portion of edges for training(0.9 by default)

	For example: `$ python imp_ent_detection.py file_path 3 0.9`
* `MTokenizer.py` is used in refining input text content, credit to [Mayank Kejriwal](http://usc-isi-i2.github.io/kejriwal/).

## Result
* All files generated and result will be put into directory 'implicit_entity/', you can refer to it checking my result for reliefWebProcessed dataset.