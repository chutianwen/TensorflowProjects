from DataCenter import DataCenter
from helper import logger
from NeuralNetwork import NeuralNetwork

if __name__ == "__main__":
    logger.info("Job started!")
    data = DataCenter().run()
    neuralNetwork = NeuralNetwork(data)
    neuralNetwork.train()
    neuralNetwork.visualizing_words()
    logger.info("Job finished!")
