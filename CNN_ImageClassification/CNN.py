from DataCenter import DataCenter
from NeuralNetwork import NeuralNetwork

if __name__ == "__main__":
    DataCenter.download_and_extract_data()
    DataCenter.process_data()
    # cifar10_dataset_folder_path = 'cifar-10-batches-py'
    # Explore the dataset
    # DataCenter.exploreDataset(2,44)
    # NeuralNetwork().train_on_one_batch()
    NeuralNetwork().train_on_whole_data()
    # NeuralNetwork().test_model()
    # x = NeuralNetwork().neural_net_image_input((32, 32, 3))