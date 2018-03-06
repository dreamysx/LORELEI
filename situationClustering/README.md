# Situation Clustering
Situation clustering aims at clustering documents based on real world information it contains.
This first version is developed for dataset **ll_Nepal** only, and clustering is based on location and topic information.

### Explain ll_Nepal
* **Topic information** comes from field `topics` of sample data, 9498 out of 29946 documents have this field, totally 11 kinds of topics.
* **Location information** comes from field `geoLocations`/`geohash`, 24834 out of 29946 documents have this field, totally 346 kinds of locations. Sample data also have a field `LOC`, basically each `LOC` value is corresponding to a `geohash` value. But, only 5384 out of 29946 documents have `LOC`. So, we choose `geohash` over `LOC` to represent location information.
* All the documents are in **json** format.

### Usage
Git clone package with the following command:
```
$ git clone https://github.com/dreamysx/lorelei.git
```
Change directory to where code is:
```
$ cd situationClustering
```
Execute the following command to run the code:
```
python [code-file-name] [input-path] [output-path]
```
For example:
```
$ python situationClustering.py sampleInput/ sampleOutput/
```