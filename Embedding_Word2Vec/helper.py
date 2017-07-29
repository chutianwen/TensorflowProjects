import logging
from logging.config import dictConfig
from tqdm import tqdm

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
               'filename': 'word2vec.log',
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


class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        """
        :param block_num: int, optional
                            Number of blocks transferred so far [default: 1].
        :param block_size: int, optional
                            Size of each block (in tqdm units) [default: 1].
        :param total_size: int, optional
                            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num

