# Situation Clustering
Situation clustering aims at clustering documents based on real world information it contains.
This first version is developed for dataset **ll_Nepal** only, and clustering is based on location and topic information.

### Explainations of ll_Nepal
**Topic information** comes from field `topics` of sample data, 9498 out of 29946 documents have this field.
**Location information** comes from field `geoLocations`/`geohash`, 24834 out of 29946 documents have this field. Sample data also have a field `LOC`, basically each `LOC` value is corresponding to a `geohash` value. But, only 5384 out of 29946 documents have `LOC`. So, we choose `geohash` over `LOC` to represent location information.

### EntityLinkage Download
Git clone the EntityLinkage package with the following command.
```
$ git clone https://github.com/ZihaoZhai/EntityLinkage.git
```

### rltk Download & Configuration
This code use similarity and distance fucntions from rltk package, so that we should download and configure rltk first. Here is the command.
```
$ git clone https://github.com/usc-isi-i2/rltk
```
You should add rely first. Run the following commands to go into the rltk folder where the file `requirements.txt` exist and add rely.
```
$ cd rltk
$ pip install -r requirements.txt
```
### Run code
Now you should be in the place where both EntityLinkage and rltk are, and then use the command to go into the EntityLinkage folder and run the code.
```
$ cd ../EntityLinkage
```
```
python [code-file-name] [input-path] [output-path]
```
If your EntityLinkage and rltk are in the same outer folder, the default rltk path should work. If not, please add the rltk path after your command and use space to separate like the following.

```
python [code-file-name] [input-path] [output-path] [rltk-path]
```
Because of the different format of the Haiti and Nepal json elements, we have to run different codes to get the result. Here, we have created sample dataset similar to each of them to test, so that we can run them with comands as follows.
```
$ python HaitiClustering.py sampleHaitiJsonInput/ sampleHaitiJsonOutput/
$ python NepalClustering.py sampleNepalJsonInput/ sampleNepalJsonOutput/
```
or if your rltk is not in the default path
```
$ python HaitiClustering.py sampleHaitiJsonInput/ sampleHaitiJsonOutput/ ../rltk
$ python NepalClustering.py sampleNepalJsonInput/ sampleNepalJsonOutput/ ../rltk
```
