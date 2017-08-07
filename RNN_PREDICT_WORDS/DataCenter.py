import numpy as np
import os
from helper import logger


class DataCenter:

    def __init__(self, data_path='data', n_seqs=50, n_steps=100):
        """

        :param data_path: the path on disk, storing the raw data
        :param batch_size: tunnable batch_size of training
        """
        self.data_path = data_path
        self.n_seqs = n_seqs
        self.n_steps = n_steps

    def process_data(self):
        with open("{}/{}".format(self.data_path, "anna.txt")) as f:
            text = f.read()

        path_int_to_vocab = "{}/int_to_vocab.npy".format(self.data_path)
        path_vocab_to_int = "{}/vocab_to_int.npy".format(self.data_path)

        if os.path.exists(path_vocab_to_int) and os.path.exists(path_int_to_vocab):
            logger.info("vocab_to_int already exist, just reload")
            vocab_to_int = np.load(path_vocab_to_int).item()
        else:
            logger.info("Create two look-up tables, vocab_to_int, int_to_vocab")
            vocab = set(text)
            logger.info("There are {} unique words in the data.".format(len(vocab)))
            vocab_to_int = {c: i for i, c in enumerate(vocab)}
            int_to_vocab = dict(enumerate(vocab))
            np.save(path_int_to_vocab, int_to_vocab)
            np.save(path_vocab_to_int, vocab_to_int)

        meta_data = {"len_vocabulary": len(vocab_to_int), "num_seq": self.n_seqs, "num_step": self.n_steps}
        path_meta = "{}/meta.npy".format(self.data_path)
        np.save(path_meta, meta_data)

        encoded = np.array([vocab_to_int[c] for c in text], dtype=np.int32)
        return encoded

    def get_batches(self, words):
        """

        :param words:
        :param n_seqs: number of sequences in one batch
        :param n_steps:  number of chars in one sequence
        :return:
        """
        chars_per_batch = self.n_seqs * self.n_steps
        num_batch = words.size // chars_per_batch

        words = words[: num_batch * chars_per_batch]
        words = np.reshape(words, [self.n_seqs, -1])

        for start in range(0, words.shape[-1], self.n_steps):
            x = words[:, start: start + self.n_steps]
            y = np.zeros_like(x)
            y[:, :-1], y[:, -1] = x[:, 1:], x[:, 0]
            yield x, y

    def run(self):
        data = self.process_data()
        return list(self.get_batches(data))
