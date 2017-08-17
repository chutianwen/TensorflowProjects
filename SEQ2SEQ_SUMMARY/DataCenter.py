import helper
import numpy as np
import string
import os
from helper import logger

class DataCenter:
    def __init__(self, data_path='data'):
        self.data_path = data_path

    def process_data(self):
        source_path = '{}/letters_source.txt'.format(self.data_path)
        target_path = '{}/letters_target.txt'.format(self.data_path)

        source_sentences = helper.load_data(source_path)
        target_sentences = helper.load_data(target_path)

        path_source_int_to_vocab = "{}/source_int_to_vocab.npy".format(self.data_path)
        path_source_vocab_to_int = "{}/source_vocab_to_int.npy".format(self.data_path)
        path_target_int_to_vocab = "{}/target_int_to_vocab.npy".format(self.data_path)
        path_target_vocab_to_int = "{}/target_vocab_to_int.npy".format(self.data_path)
        if os.path.exists(path_source_int_to_vocab) and os.path.exists(path_source_vocab_to_int) and \
            os.path.exists(path_target_int_to_vocab) and os.path.exists(path_target_vocab_to_int):
            logger.info("vocab_to_int already exist, just reload")
            source_vocab_to_int = np.load(path_source_vocab_to_int).item()
            target_vocab_to_int = np.load(path_target_vocab_to_int).item()
        else:
            source_int_to_vocab, source_vocab_to_int = self.extract_character_vocab(source_sentences)
            target_int_to_vocab, target_vocab_to_int = self.extract_character_vocab(target_sentences)
            np.save(path_source_int_to_vocab, source_int_to_vocab)
            np.save(path_source_vocab_to_int, source_vocab_to_int)
            np.save(path_target_int_to_vocab, target_int_to_vocab)
            np.save(path_target_vocab_to_int, target_vocab_to_int)

        # Convert characters to ids
        source_letter_ids = [[source_vocab_to_int.get(letter, source_vocab_to_int['<UNK>']) for letter in line]
                             for line in source_sentences.split('\n')]
        target_letter_ids = [[target_vocab_to_int.get(letter, target_vocab_to_int['<UNK>']) for letter in line] +
                             [target_vocab_to_int['<EOS>']]
                             for line in target_sentences.split('\n')]
        return source_letter_ids, target_letter_ids

    def extract_character_vocab(self, data):
        special_vocab = ['<PAD>', '<UNK>', '<GO>',  '<EOS>']

        set_vocab = set([character for line in data.split('\n') for character in line])
        int_to_vocab = {word_i: word for word_i, word in enumerate(special_vocab + list(set_vocab))}
        vocab_to_int = {word: word_i for word_i, word in int_to_vocab.items()}

        return int_to_vocab, vocab_to_int

    def run(self):
        source_letter_ids, target_letter_ids = self.process_data()
        return source_letter_ids, target_letter_ids