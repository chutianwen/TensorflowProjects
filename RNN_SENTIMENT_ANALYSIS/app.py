from DataCenter import DataCenter
from NeuralNetworks import NeuralNetworks
from helper import logger

if __name__ == "__main__":

    logger.info("Start Job...")
    inputs, targets = DataCenter().run()
    neural_network = NeuralNetworks(inputs, targets, split_fraction=0.8, embed_size=300, lstm_size=256)
    neural_network.train()
    neural_network.test()
    logger.info("Job is done!")