from helper import logger, TaskReporter
import tensorflow as tf
import os
import numpy as np
from tensorflow.python.util.nest import flatten


class NeuralNetworks:
    def __init__(self, features, targets, split_fraction=0.8,
                 embed_size=300, lstm_size=256, lstm_layers=1, batch_size=500, learning_rate=0.001,
                 keep_prob=0.5, epochs=15):

        split_idx = int(len(features) * split_fraction)
        self.train_x, self.val_x = features[:split_idx], features[split_idx:]
        self.train_y, self.val_y = targets[:split_idx], targets[split_idx:]

        test_idx = int(len(self.val_x) * 0.5)
        self.val_x, self.test_x = self.val_x[:test_idx], self.val_x[test_idx:]
        self.val_y, self.test_y = self.val_y[:test_idx], self.val_y[test_idx:]

        self.embed_size = embed_size
        self.lstm_size = lstm_size
        self.lstm_layers = lstm_layers
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.keep_prob = keep_prob
        self.epochs = epochs

        path_vocab_to_int = "{}/vocab_to_int.npy".format("./data")
        if os.path.exists(path_vocab_to_int):
            self.vocab_to_int = np.load(path_vocab_to_int).item()
        else:
            logger.error("No vocab_to_int data found, please re-run DataCenter.")
            exit()

    def build_inputs(self):
        """
        Define placeholders for inputs, targets, and dropout
        :param batch_size: Batch size, number of sequences per batch
        :param num_steps: Number of sequence steps in a batch
        :return: tensorflow placeholders
        """
        # Declare placeholders we'll feed into the graph
        inputs = tf.placeholder(tf.int32, [None, None], name='inputs')
        targets = tf.placeholder(tf.int32, [None, None], name='targets')

        # Keep probability placeholder for drop out layers
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        return inputs, targets, keep_prob

    def build_embedding(self, inputs):

        assert isinstance(self.vocab_to_int, dict), "vocab_to_int should be python dict!"
        vocab_size = len(self.vocab_to_int)
        # initialize weights based on random_uniform between [-1/sqrt(n), 1/sqrt(n)]
        # n is size of input to a neuron, rather than the batch size
        limit = 1.0 / np.sqrt(vocab_size)
        embedding_table = tf.Variable(tf.random_uniform([vocab_size, self.embed_size], minval=-limit, maxval=limit))
        embed = tf.nn.embedding_lookup(embedding_table, inputs)
        return embed

    def build_lstm(self, keep_prob):
        """
        Build tensorflow lstm layer.
        :param num_layers: number of lstm layers stacked on each other
        :param batch_size:
        :param keep_prob: Drop out probablity
        :return:
        """

        def build_cell(lstm_size, keep_prob):
            """
            Build multi-lstm layers
            :param lstm_size:
            :param keep_prob:
            :return:
            """
            # Use a basic LSTM cell
            lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)

            # Add dropout to the cell
            drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)
            return drop

        # Stack up multiple LSTM layers, consider multi layer of lstm as one cell
        cell = tf.contrib.rnn.MultiRNNCell([build_cell(self.lstm_size, keep_prob) for _ in range(self.lstm_layers)])
        initial_state = cell.zero_state(self.batch_size, tf.float32)

        return cell, initial_state

    def build_output(self, lstm_output, keep_prob):
        inputs = lstm_output[:, -1]
        weights = tf.Variable(tf.truncated_normal([self.lstm_size, 1], mean=0, stddev=0.1))
        bias = tf.Variable(tf.zeros([1, 1]))
        logits = tf.add(tf.matmul(inputs, weights), bias)
        predict = tf.nn.sigmoid(logits, name='prediction')
        # predict = tf.nn.dropout(predict, keep_prob=keep_prob, name='predict')
        return predict

    def build_loss(self, predict, labels):
        cost = tf.losses.mean_squared_error(labels=labels, predictions=predict)
        cost = tf.reduce_mean(cost, name='cost')
        return cost

    def build_optimizer(self, cost):
        optimizer = tf.train.AdamOptimizer(self.learning_rate, name='optimizer').minimize(cost)
        return optimizer

    def build_graph(self, save_model_path):
        if os.path.exists("{}.meta".format(save_model_path)):
            logger.info("Graph existed, ready to be reloaded...")
        else:
            tf.reset_default_graph()

            inputs, targets, keep_prob = self.build_inputs()
            embed = self.build_embedding(inputs=inputs)
            cell, initial_state = self.build_lstm(keep_prob=keep_prob)
            # Add all tensors in inital_state into saved graph.
            for tensor in flatten(initial_state):
                tf.add_to_collection('initial_state', tensor)

            # with tf.device("cpu:0"):
            outputs, final_state = tf.nn.dynamic_rnn(cell, embed, initial_state=initial_state)
            # Add all tensors in final_state into saved graph.
            for tensor in flatten(final_state):
                tf.add_to_collection('final_state', tensor)

            predict = self.build_output(outputs, keep_prob)
            cost = self.build_loss(predict=predict, labels=targets)
            optimizer = self.build_optimizer(cost=cost)
            correct_prediction = tf.equal(tf.cast(tf.round(predict), tf.int32), targets)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

            saver = tf.train.Saver()
            with tf.Session(graph=tf.get_default_graph()) as sess:
                sess.run(tf.global_variables_initializer())
                saver.save(sess, save_model_path)

    def load_graph(self, sess, save_model_path):

        graph = {}
        print("Saved_model_path:{}".format(save_model_path))
        loader = tf.train.import_meta_graph(save_model_path + '.meta')

        graph['inputs'] = sess.graph.get_tensor_by_name("inputs:0")
        graph['targets'] = sess.graph.get_tensor_by_name("targets:0")
        graph['keep_prob'] = sess.graph.get_tensor_by_name("keep_prob:0")
        graph['cost'] = sess.graph.get_tensor_by_name('cost:0')
        graph['prediction'] = sess.graph.get_tensor_by_name("prediction:0")
        graph['optimizer'] = sess.graph.get_operation_by_name("optimizer")
        graph['accuracy'] = sess.graph.get_tensor_by_name("accuracy:0")

        initial_state_new = tf.get_collection('initial_state')
        initial_state_tuple = ()
        for id in range(0, len(initial_state_new), 2):
            initial_state_tuple += tf.contrib.rnn.LSTMStateTuple(initial_state_new[id], initial_state_new[id + 1]),
        graph['initial_state'] = initial_state_tuple

        final_state_new = tf.get_collection('final_state')
        final_state_tuple = ()
        for id in range(0, len(final_state_new), 2):
            final_state_tuple += tf.contrib.rnn.LSTMStateTuple(final_state_new[id], final_state_new[id + 1]),
        graph['final_state'] = final_state_tuple

        logger.info("model is ready, good to go!")

        check_point = tf.train.latest_checkpoint('checkpoints')
        # if no check_point found, means we need to start training from scratch, just initialize the variables.
        if not check_point:
            # Initializing the variables
            logger.info("Initializing the variables")
            sess.run(tf.global_variables_initializer())
        else:
            logger.info("check point path:{}".format(check_point))
            loader.restore(sess, check_point)
        return graph

    def get_batch(self, x, y):

        num_batch = len(x) // self.batch_size
        x, y = x[:num_batch * self.batch_size], y[:num_batch * self.batch_size]
        for start in range(0, len(x), self.batch_size):
            yield x[start:start + self.batch_size], y[start:start + self.batch_size]

    def validate_model(self, sess, inputs, targets, keep_prob, initial_state, state, accuracy):
        res = []
        for val_x, val_y in self.get_batch(self.val_x, self.val_y):
            accuracy_cur = sess.run(accuracy, feed_dict={
                inputs: val_x,
                targets: val_y,
                keep_prob: self.keep_prob,
                initial_state: state
            })
            res.append(accuracy_cur)
        logger.info("Validation accuracy is:{}".format(np.mean(res)))

    @TaskReporter("Train graph")
    def train(self):
        save_model_path = './savedModel/rnn-model'
        self.build_graph(save_model_path)
        tf.reset_default_graph()

        validate_time = 50
        checkpoint_time = 200
        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        with tf.Session(graph=tf.get_default_graph(), config=config) as sess:
            graph = self.load_graph(sess, save_model_path)
            logger.info("Start training...")
            iteration = 1
            for epoch in range(self.epochs):
                state = sess.run(graph['initial_state'])
                state_validation = state
                saver = tf.train.Saver()
                for x, y in self.get_batch(self.train_x, self.train_y):
                    feed = {
                        graph['inputs']: x,
                        graph['targets']: y,
                        graph['keep_prob']: self.keep_prob,
                        graph['initial_state']: state
                    }
                    state, cost, _, prediction = sess.run([graph['final_state'], graph['cost'], graph['optimizer'],
                                                        graph['prediction']], feed_dict=feed)
                    # print(prediction)
                    if iteration % 10 == 0:
                        logger.info("Epoch: {}/{}\t".format(epoch + 1, self.epochs) +
                                    "Iteration: {}\t".format(iteration) +
                                    "Train loss: {:.3f}\t".format(cost))

                    if iteration % validate_time == 0:
                        self.validate_model(sess, graph['inputs'], graph['targets'],
                                            graph['keep_prob'], graph['initial_state'],
                                            state_validation, graph['accuracy'])
                    iteration += 1

                    if iteration % checkpoint_time == 0:
                        saver.save(sess, "checkpoints/i{}_l{}.ckpt".format(iteration, self.lstm_size))
            saver.save(sess, "checkpoints/i{}_l{}.ckpt".format(iteration, self.lstm_size))

    @TaskReporter("Test graph")
    def test(self):
        save_model_path = './savedModel/rnn-model'
        self.build_graph(save_model_path)
        tf.reset_default_graph()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        res = []
        with tf.Session(graph=tf.get_default_graph(), config=config) as sess:
            graph = self.load_graph(sess, save_model_path)
            logger.info("Start test...")
            state = sess.run(graph['initial_state'])

            for x, y in self.get_batch(self.test_x, self.test_y):
                state, accuracy_cur = sess.run([graph['final_state'], graph['accuracy']], feed_dict={
                    graph['inputs']: x,
                    graph['targets']: y,
                    graph['keep_prob']: self.keep_prob,
                    graph['initial_state']: state
                })
                res.append(accuracy_cur)
        logger.info("Test accuracy is:{}".format(np.mean(res)))