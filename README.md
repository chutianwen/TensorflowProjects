# TensorflowProjects
The repo includes multiple projects using tensorflow, including CNN with image classification, RNN for stock analysis.

## Prerequisite
Please use `pip install` or `conda install` to prepare dependencies shown as below:
- python 3.5
- sklearn
- pandas
- numpy
- pickle

## CNN Image Classification
In this project, the program classifies images from the CIFAR-10 dataset. The dataset consists of airplanes, dogs, cats,and other objects.
After preprocessing, the features are normalized and labels are one-hot encoded. 

### To run the program
```
python CNN.py
```

## Embedding Word2Vec
In this project, the program use skip-gram model to process the data "text8" which can be downloaded from remote server. After two steps of de-noise, word are represented as tokens. By using embedding_look_up, huge compute cost is saved. Finally the words are translated into embedding matrix. Each word corresponds to one row in the embedding matrix. One visualization function can show the relationship between each words by projecting word in higher dimension(#embedding column/feature) into 2D space.

### To run the program
```
python app.py
```
## Authors

* **Tianwen Chu** - *Initial work*
