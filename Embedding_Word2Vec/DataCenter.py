import os
from helper import logger, DLProgress, TaskReporter
from urllib.request import urlretrieve
import zipfile
from collections import Counter
import numpy as np
import random

class DataCenter:

    def __init__(self, data_path='data', batch_size=128):
        """

        :param data_path: the path on disk, storing the raw data
        :param batch_size: tunnable batch_size of training
        """
        self.data_path = data_path
        self.batch_size = batch_size

    @TaskReporter("Download and extract data")
    def download_and_extract_data(self):
        """
        Download and extract the raw data.
        :param data_path: local disk path which stores the raw data, if not exist, then decide if necessary to download
        the data from server and extract.
        """
        if os.path.exists(self.data_path):
            logger.info("Data is ready, good to go")
        else:
            os.mkdir(self.data_path)
            dataset_filename_compressed = 'text8.zip'
            if not os.path.isfile(dataset_filename_compressed):
                data_remote_url = 'http://mattmahoney.net/dc/text8.zip'
                logger.info("Downloading the data from remote server...")
                with DLProgress(unit="B", unit_scale=True, miniters=1, desc="Text8 data") as pbar:
                    urlretrieve(
                        data_remote_url,
                        dataset_filename_compressed,
                        pbar.hook
                    )
                logger.info("Downloading finished")

            with zipfile.ZipFile(dataset_filename_compressed) as zip_ref:
                zip_ref.extractall(self.data_path)

    @TaskReporter("Preprocess data")
    def preprocess_data(self):
        """
        Read the data, tokenize the words. Apply two noise reduction method.
        1). Using word frequency threshold cut to trim out the ones with very low frequency
        2). Using Mikolov subsampling to trim out the words have too high frequency like "the, a ..."
        Save the int_to_word and word_to_int look-up tables as .npy.
        :return:
        """
        with open("{}/{}".format(self.data_path, 'text8')) as f:
            text = f.read()
            # Replace punctuation with tokens so we can use them in our model
            # important to tokenize these signs
            text = text.lower()
            text = text.replace('.', ' <PERIOD> ')
            text = text.replace(',', ' <COMMA> ')
            text = text.replace('"', ' <QUOTATION_MARK> ')
            text = text.replace(';', ' <SEMICOLON> ')
            text = text.replace('!', ' <EXCLAMATION_MARK> ')
            text = text.replace('?', ' <QUESTION_MARK> ')
            text = text.replace('(', ' <LEFT_PAREN> ')
            text = text.replace(')', ' <RIGHT_PAREN> ')
            text = text.replace('--', ' <HYPHENS> ')
            text = text.replace('?', ' <QUESTION_MARK> ')
            # text = text.replace('\n', ' <NEW_LINE> ')
            text = text.replace(':', ' <COLON> ')
            words = text.split()
            logger.info("Total origin words:{}".format(len(words)))

            # Remove all words with 5 or fewer occurrences.
            word_counts = Counter(words)
            lookup_table_trimmed = {word: cnt for word, cnt in word_counts.items() if cnt > 5}
            # get unique words from lookup_table_trimmed.keys() with sorted order. Words up front are with higher
            # frequency.
            unique_words = sorted(lookup_table_trimmed, key=lookup_table_trimmed.get, reverse=True)
            unique_words_set = set(unique_words)
            logger.info("Total unique words:{}".format(len(unique_words)))

            trimmed_words = [word for word in words if word in unique_words_set]
            logger.info("Total trimmed words:{}".format(len(trimmed_words)))

            int_to_word = {i: word for i, word in enumerate(unique_words)}
            word_to_int = {word: i for i, word in int_to_word.items()}
            path_int_to_word = "{}/int_to_word.npy".format(self.data_path)
            path_word_to_int = "{}/word_to_int.npy".format(self.data_path)

            if not os.path.exists(path_int_to_word):
                np.save(path_int_to_word, int_to_word)
            if not os.path.exists(path_word_to_int):
                np.save(path_word_to_int, word_to_int)
            total_cnt_words_trimmed = len(trimmed_words)

            # Apply Mikolov sub-sampling
            threshold = 1e-5
            prob_discard_lookup_table = {word: 1 - np.sqrt(threshold/(cnt/total_cnt_words_trimmed))
                                         for word, cnt in lookup_table_trimmed.items()}
            trimmed_words_subsampled = [word_to_int[word] for word in trimmed_words if random.random() <
                                        1 - prob_discard_lookup_table[word]]
            logger.info("Total trimmed and sampled words:{}".format(len(trimmed_words_subsampled)))
            return trimmed_words_subsampled


    def get_target(self, words, idx, window_size=5):
        """
        Get a list of words in a window around an index.
        :param words: List[int] All words in the given batch of text
        :param idx: int The index of current word
        :param window_size: int This is the sampling range for the selection range.
        :return:
        """
        R = np.random.randint(1, window_size+1)
        # handling the two-end corner cases
        start = idx - R if (idx - R) > 0 else 0
        stop = idx + R
        # Here has a confusion of removing duplicate words within R. Is not the duplicate target a fact?
        target_words = set(words[start:idx] + words[idx+1:stop+1])

        return list(target_words)

    def get_batches(self, words, batch_size, window_size=5):
        """
        Create a generator of word batches as a tuple (inputs, targets)
        :param words: All words in the text
        :param batch_size:
        :param window_size:
        :return: generator of List[(List[int], List[int])]
        """
        n_batches = len(words) // batch_size

        # only full batches, trim out the rest. In the real practice, this may be too careless.
        words = words[:n_batches*batch_size]

        for idx in range(0, len(words), batch_size):
            x, y = [], []
            batch = words[idx: idx + batch_size]
            for ii in range(len(batch)):
                batch_x = batch[ii]
                batch_y = self.get_target(batch, ii, window_size)
                y.extend(batch_y)
                x.extend([batch_x]*len(batch_y))
            yield x, y


    def run(self):
        """
        Application Interface.
        :return:
        """
        self.download_and_extract_data()
        processed_data = self.preprocess_data()
        data = self.get_batches(processed_data, self.batch_size)
        return data