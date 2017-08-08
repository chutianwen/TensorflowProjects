from helper import logger
import numpy as np
import string
import os


class DataCenter:
    def __init__(self, data_path='data', batch_size=500, feature_length=300):
        """
        :param data_path: path storing text data
        :param batch_size:
        :param feature_length:
        """
        self.data_path = data_path
        self.batch_size = batch_size
        self.feature_length = feature_length

    def process_data(self):

        with open("{}/{}".format(self.data_path, "labels.txt")) as f:
            lables = f.read()
        with open("{}/{}".format(self.data_path, "reviews.txt")) as f:
            reviews = f.read()

        # remove all the punctuations in the reviews
        translator = str.maketrans('', '', string.punctuation)
        reviews_rinsed = reviews.translate(translator)
        reviews_separated = reviews_rinsed.split("\n")
        # remove any empty review
        reviews_separated = list(filter(lambda x: x, reviews_separated))
        path_int_to_vocab = "{}/int_to_vocab.npy".format(self.data_path)
        path_vocab_to_int = "{}/vocab_to_int.npy".format(self.data_path)
        if os.path.exists(path_vocab_to_int) and os.path.exists(path_int_to_vocab):
            logger.info("vocab_to_int already exist, just reload")
            vocab_to_int = np.load(path_vocab_to_int).item()
        else:
            # split by word
            vocabulary = set(reviews_rinsed.split())
            vocab_to_int = {vocab: id for id, vocab in enumerate(vocabulary, 1)}
            int_to_vocab = dict(enumerate(vocabulary, 1))
            np.save(path_vocab_to_int, vocab_to_int)
            np.save(path_int_to_vocab, int_to_vocab)

        features = np.zeros([len(reviews_separated), self.feature_length], dtype=np.int32)
        for id in range(features.shape[0]):
            review_words = reviews_separated[id].split()
            review_size = len(review_words)
            review_size = review_size if review_size <= self.feature_length else self.feature_length
            features[id, -review_size:] = [vocab_to_int[word] for word in review_words[:review_size]]

        labels_separated = lables.split()
        targets = np.array(list(map(lambda x: 1 if x == 'positive' else 0, labels_separated)))[:, None]
        logger.info("Number of reviews:{}\tNumber of labels:{}".format(len(features), len(targets)))
        return features, targets

    def run(self):
        return self.process_data()

