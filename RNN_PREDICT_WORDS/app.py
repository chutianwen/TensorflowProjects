from DataCenter import DataCenter
from helper import logger
from NeuralNetworks import NeuralNetworks

if __name__ == "__main__":
    logger.info("Job started!")
    data = DataCenter().run()

    neural_network = NeuralNetworks(data)
    neural_network.train()
    # new_text = neural_network.sample(1000, prime='Far')
    # print(new_text)
    logger.info("Job finished!")
