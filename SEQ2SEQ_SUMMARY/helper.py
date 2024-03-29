import os
import logging
from logging.config import dictConfig

logging_config = dict(
    version=1,
    formatters={
        'f': {'format': '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'},
        'f1': {'format': '%(asctime)s %(levelname)-8s %(message)s'}
    },
    handlers={
        'sh': {'class': 'logging.StreamHandler',
               'formatter': 'f',
               'level': logging.DEBUG
               },
        'fh': {'class': 'logging.FileHandler',
               'filename': 'seq2seq.log',
               'formatter': 'f1',
               'level': logging.DEBUG
               }
    },
    root={
        'handlers': ['sh', 'fh'],
        'level': logging.DEBUG,
    },
)
dictConfig(logging_config)
logger = logging.getLogger()


class TaskReporter(object):
    def __init__(self, task):
        self.task = task

    def __call__(self, original_func):
        decorator_self = self

        def wrapper(*args, **kwargs):
            logger.info("Processing task: {}...".format(decorator_self.task))
            res = original_func(*args, **kwargs)
            logger.info("Task: {} is done".format(decorator_self.task))
            return res
        return wrapper


def load_data(path):
    input_file = os.path.join(path)
    with open(input_file, "r", encoding='utf-8', errors='ignore') as f:
        data = f.read()

    return data


def extract_vocab(data):
    special_words = ['<pad>', '<unk>', '<s>',  '<\s>']

    set_words = set([word for line in data.split('\n') for word in line.split()])
    int_to_vocab = {word_i: word for word_i, word in enumerate(special_words + list(set_words))}
    vocab_to_int = {word: word_i for word_i, word in int_to_vocab.items()}

    return int_to_vocab, vocab_to_int


def pad_id_sequences(source_ids, source_vocab_to_int, target_ids, target_vocab_to_int, sequence_length):
    new_source_ids = [list(reversed(sentence + [source_vocab_to_int['<pad>']] * (sequence_length - len(sentence)))) \
                      for sentence in source_ids]
    new_target_ids = [sentence + [target_vocab_to_int['<pad>']] * (sequence_length - len(sentence)) \
                      for sentence in target_ids]

    return new_source_ids, new_target_ids


def batch_data(source, target, batch_size):
    """
    Batch source and target together
    """
    for batch_i in range(0, len(source)//batch_size):
        start_i = batch_i * batch_size
        source_batch = source[start_i:start_i + batch_size]
        target_batch = target[start_i:start_i + batch_size]
        yield source_batch, target_batch
