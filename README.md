# TensorflowProjects
This never-ending repo includes multiple projects using tensorflow, including CNN, RNN applications. Deep learning is one of my great passions and I cannot stop learning these cutting-edge knowledge. In all the code, I put some important tips and notices as comment. It does take some time to have a deeper grasp of tensorflow. Understanding scope, graph, session, tensors, import_meta_graph, restore is very useful points to study tensorflow. 

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

## RNN Write Robot Article 
In this project, the program trained a RNN neural network to learn writing article. Given training data like text files, books and articles, the program will orgnaize the text into a number of long sequences. Each long sequence will then be evenly splitted into shorter sequences with specified length of characters (also called number of steps). Thus training data will be processed into batches, and each batch size will be #number of long sequence * #number of steps. Input and output pair will be the current character and the following character, ex. "I am good" will have input pairs as ('I', ' '), (' ', 'a'), ('a', 'm')... To avoid gradient vanishing or exploding, lstm cell is used with extral help of gradienct threshold cutting. 

The important idea of using RNN to process such problem is the state transportation. The determination of next character depends on all previous characters rather than the current character only. The program can learn a context of given word, sentence or even paragraph because of such feature. 
### To run the program
```
python app.py
```
## Authors

* **Tianwen Chu** - *Initial work*
