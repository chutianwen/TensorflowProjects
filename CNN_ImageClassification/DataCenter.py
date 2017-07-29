import helper
from helper import logger, TaskReporter, DLProgress
from os.path import isdir, isfile
import os
from urllib.request import urlretrieve
import tarfile
import numpy as np


class DataCenter:

    cifar10_dataset_folder_path = 'cifar-10-batches-py'

    def __init__(self):
        pass

    @staticmethod
    def download_and_extract_data(cifar10_dataset_folder_path='cifar-10-batches-py'):
        """
        Detect if cifar data is ready on disk or download the cifar data from internet and extract it.
        :param cifar10_dataset_folder_path: Name of extracted data folder
        :return:
        """
        # check if it has already been unzipped
        if isdir(cifar10_dataset_folder_path):
            logger.info("Data is ready, good to go!")
        else:
            # Check if the compressed data has already been downloaded
            tar_gz_path = 'cifar-10-python.tar.gz'

            # Check if compressed data is not on disk, download by urllib
            data_remoteURL = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
            if not isfile(tar_gz_path):
                logger.info("Downloading data from remote server...")
                with DLProgress(unit='B', unit_scale=True, miniters=1, desc='CIFAR-10 Dataset') as pbar:
                    urlretrieve(
                        data_remoteURL,
                        tar_gz_path,
                        pbar.hook
                    )
                logger.info("Downloading finished")

            with tarfile.open(tar_gz_path) as tar:
                logger.info("Extracting the data...")
                tar.extractall()
                tar.close()
            logger.info("Data is fully extracted, good to go!")

    @staticmethod
    @TaskReporter(task="explore_dataset")
    def explore_dataset(batch_id, sample_id):
        """
        Explore what is inside the data by showing specific image.
        :param batch_id:
        :param sample_id:
        :return:
        """
        helper.display_stats(DataCenter.cifar10_dataset_folder_path, batch_id, sample_id)

    @staticmethod
    def normalize(images):
        """
        Normalize a list of sample image data in the range of 0 to 1
        : images: List of image data.(numpy.ndarray)  The image shape is (32, 32, 3)
        : return: Numpy array of normalize data
        """
        # if image has already been
        if images.max() <= 1:
            return images
        else:
            return images/255

    @staticmethod
    def one_hot_encode(labels):
        """
        One hot encode a list of sample labels. Return a one-hot encoded vector for each label.
        : labels: List of sample Labels
        : return: Numpy array of one-hot encoded labels
        """
        labels_ont_hot = np.zeros((len(labels), 10), dtype=np.int16)
        for idx, label in enumerate(labels):
            labels_ont_hot[idx][label] = 1
        return labels_ont_hot

    @staticmethod
    @TaskReporter(task="Preprocess data")
    def process_data():
        # Preprocess Training, Validation, and Testing Data
        target_dir = "./ProcessedData"
        if not os.path.exists(target_dir):
            os.mkdir(target_dir)
            helper.preprocess_and_save_data(DataCenter.cifar10_dataset_folder_path, DataCenter.normalize,
                                            DataCenter.one_hot_encode, target_dir=target_dir)
        else:
            logger.info("Data has already been processed, good to go!")
            return
