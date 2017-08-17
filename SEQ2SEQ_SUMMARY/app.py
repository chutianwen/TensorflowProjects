from DataCenter import DataCenter
from NeuralNetworks import NeuralNetworks
from helper import logger


if __name__ == "__main__":

    logger.info("Start Job...")
    data = DataCenter().run()
    para_dict = {
        'lstm_size': 128,
        'lstm_layers': 2,
        'embedding_size': 25,
        'batch_size': 256,
        'epochs': 2,
        'keep_prob': 0.5,
        'learning_rate': 0.001
    }
    neuralNetworks = NeuralNetworks(data, para_dict)
    neuralNetworks.train()
    text = ["BigBig", 'tianwen']
    neuralNetworks.seq_to_seq(text, 20)
    logger.info("Job is done!")