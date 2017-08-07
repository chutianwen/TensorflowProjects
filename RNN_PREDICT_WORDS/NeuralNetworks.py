import tensorflow as tf
import os
from helper import logger, TaskReporter
import numpy as np
from time import time
from tensorflow.python.util.nest import flatten

class NeuralNetworks:

    def __init__(self, data):
        path_meta = "./data/meta.npy"
        if os.path.exists(path_meta):
            meta_data = np.load(path_meta).item()
            print(meta_data)
            assert isinstance(meta_data, dict), "meta data is not python dict type, something wrong"
            self.num_seq = meta_data['num_seq']
            self.num_step = meta_data['num_step']
            self.len_vocabulary = meta_data['len_vocabulary']

            self.data = data

            self.lstm_size = 512         # Size of hidden layers in LSTMs
            self.num_layers = 2          # Number of LSTM layers
            self.learning_rate = 0.001   # Learning rate
            self.keep_prob = 0.5         # Dropout keep probability
        else:
            logger.error("Meta data is missing, please re-run DataCenter first!")
            exit()

    def build_inputs(self, batch_size, num_steps):
        """
        Define placeholders for inputs, targets, and dropout
        :param batch_size: Batch size, number of sequences per batch
        :param num_steps: Number of sequence steps in a batch
        :return: tensorflow placeholders
        """
        # Declare placeholders we'll feed into the graph
        inputs = tf.placeholder(tf.int32, [batch_size, num_steps], name='inputs')
        targets = tf.placeholder(tf.int32, [batch_size, num_steps], name='targets')

        # Keep probability placeholder for drop out layers
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        return inputs, targets, keep_prob

    def build_lstm(self, lstm_size, num_layers, batch_size, keep_prob):
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
        cell = tf.contrib.rnn.MultiRNNCell([build_cell(lstm_size, keep_prob) for _ in range(num_layers)])
        initial_state = cell.zero_state(batch_size, tf.float32)
        # print(initial_state)

        return cell, initial_state

    def build_output(self, lstm_output, in_size, out_size):
        """
        Build output layer
        :param in_size: number of LSTM output units
        :param out_size:
        :return:
        """
        # Reshape output so it's a bunch of rows, one row for each step for each sequence.
        # That is, the shape should be num_seqs*num_steps rows by lstm_size columns
        # this will become a flattened long tensor with M*N*L
        seq_output = tf.concat(lstm_output, axis=1)
        x = tf.reshape(seq_output, [-1, in_size])

        # Connect the RNN outputs to a softmax layer
        with tf.variable_scope('softmax'):
            softmax_w = tf.Variable(tf.truncated_normal((in_size, out_size), mean=0.0, stddev=0.1))
            softmax_b = tf.Variable(tf.zeros(out_size))

        # Since output is a bunch of rows of RNN cell outputs, logits will be a bunch
        # of rows of logit outputs, one for each step and sequence
        logits = tf.matmul(x, softmax_w) + softmax_b

        # Use softmax to get the probabilities for predicted characters
        out = tf.nn.softmax(logits, name='predictions')

        return out, logits

    def build_loss(self, logits, targets):
        """
        Calculate loss term during training. Using softmax_cross_entropy as cost function.
        :param logits:
        :param targets:
        :return:
        """
        targets_one_hot = tf.one_hot(targets, self.len_vocabulary)
        targets_one_hot = tf.reshape(targets_one_hot, logits.shape)
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=targets_one_hot)
        loss = tf.reduce_mean(loss, name="loss")
        return loss

    def build_optimizer(self, loss, learning_rate, grad_clip):
        """
        Build optmizer for training, using gradient clipping. Even though using lstm to solve gradience explode/vanish
        issues, it may still appear to be too large, so using a threashold to avoid such situation.
        :param loss:
        :param learning_rate:
        :param grad_clip:
        :return:
        """
        train_vars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, train_vars), grad_clip)
        train_op = tf.train.AdamOptimizer(learning_rate)
        optimizer = train_op.apply_gradients(zip(grads, train_vars), name='optimizer')

        return optimizer

    def build_graph(self, save_model_path, lstm_size=128, num_layers=2, learning_rate=0.001, grad_clip=5,
                    sampling=False):
        """
        Build the graph for RNN Model and save them as meta data for future reload. Notice that the initial state from
        lstm layer method cannot be saved directly, initial state is a python tuple of lstm table. We have to flatten
        first and save them as a collection by tensorflow tf.add_to_collection
        :param save_model_path: path of stored model
        :param num_classes: number of unique words
        :param batch_size: number of sequences in one batch
        :param num_steps: number of steps in one sequence
        :param lstm_size: number of output units from lstm layer
        :param num_layers: number of stacked lstm layers
        :param learning_rate: coe during training.
        :param grad_clip: threshold of clipping large gradient during training.
        :param sampling: choose different size of input
        """
        print("creating into " + save_model_path)
        if os.path.exists("{}.meta".format(save_model_path)):
            logger.info("Graph existed, ready to be reloaded...")
        else:
            logger.info("Creating a new meta graph...")
            # When we're using this network for sampling later, we'll be passing in
            # one character at a time, so providing an option for that
            if sampling:
                batch_size, num_steps = 1, 1
            else:
                batch_size, num_steps = self.num_seq, self.num_step

            tf.reset_default_graph()
            inputs, targets, keep_prob = self.build_inputs(batch_size, num_steps)

            cell, initial_state = self.build_lstm(lstm_size, num_layers, batch_size, keep_prob)
            # Add all tensors in inital_state into saved graph.
            for tensor in flatten(initial_state):
                tf.add_to_collection('rnn_state_input', tensor)

            inputs_one_hot = tf.one_hot(inputs, self.len_vocabulary)

            # !!!To-do some ops cannot be run on gpu. Need to figure out!!
            with tf.device('/cpu:0'):
                outputs, final_state = tf.nn.dynamic_rnn(cell, inputs_one_hot, initial_state=initial_state)

                # Add all tensors in final_state into saved graph.
                for tensor in flatten(final_state):
                    tf.add_to_collection('rnn_final_input', tensor)
                out_prob, logits = self.build_output(lstm_output=outputs, in_size=lstm_size, out_size=self.len_vocabulary)
                logits = tf.identity(logits, "logits")
                loss = self.build_loss(logits, targets)
                optimizer = self.build_optimizer(loss, learning_rate, grad_clip)

            saver = tf.train.Saver()
            # if not os.path.exists('./savedModel'):
            #     os.mkdir('./savedModel')
            logger.info("Save to {}".format(save_model_path))
            with tf.Session(graph=tf.get_default_graph()) as sess:
                sess.run(tf.global_variables_initializer())
                saver.save(sess, save_model_path)

            tf.reset_default_graph()

    def __load_graph(self, sess, save_model_path, sampling=False):
        """
        Reload data into a graph dictionary containing tensors and ops.
        Notice that for initial states and final states, we should iterate by each two steps(cell and hidden tensors)
        in the value(). These two tensors will form a LSTMStateTuple.
        !!!!!!!!!!
        Never call build_graph() here, since _load_graph is called inside a session, build_graph() has another session,
        nested session() will cause serious PROBLEM.
        :param sess: current session
        :param save_model_path: path storing meta data
        :return: dictionary
        """
        graph = {}
        print("Saved_model_path:{}".format(save_model_path))
        loader = tf.train.import_meta_graph(save_model_path + '.meta')

        graph['inputs'] = sess.graph.get_tensor_by_name("inputs:0")
        graph['targets'] = sess.graph.get_tensor_by_name("targets:0")
        graph['keep_prob'] = sess.graph.get_tensor_by_name("keep_prob:0")
        graph['loss'] = sess.graph.get_tensor_by_name('loss:0')
        graph['logits'] = sess.graph.get_tensor_by_name("logits:0")
        graph['optimizer'] = sess.graph.get_operation_by_name("optimizer")

        graph['predictions'] = sess.graph.get_tensor_by_name("predictions:0")

        initial_state_new = tf.get_collection('rnn_state_input')
        initial_state_tuple = ()
        for id in range(0, len(initial_state_new), 2):
            initial_state_tuple += tf.contrib.rnn.LSTMStateTuple(initial_state_new[id], initial_state_new[id + 1]),
        graph['initial_state'] = initial_state_tuple

        final_state_new = tf.get_collection('rnn_final_input')
        final_state_tuple = ()
        for id in range(0, len(final_state_new), 2):
            final_state_tuple += tf.contrib.rnn.LSTMStateTuple(final_state_new[id], final_state_new[id + 1]),
        graph['final_state'] = final_state_tuple

        logger.info("model is ready, good to go!")

        # If doing sampling later, than restore the variable data from latest checkpoint.

        check_point = tf.train.latest_checkpoint('checkpoints')
        # if no check_point found, means we need to start training from scratch, just initialize the variables.
        if not check_point:
            # Initializing the variables
            sess.run(tf.global_variables_initializer())
        else:
            logger.info("check point path:{}".format(check_point))
            loader.restore(sess, check_point)
        return graph

    @TaskReporter("Training graph")
    def train(self):
        """
        Traing the data and save the variables to backend.
        """
        epochs = 50
        save_every_n = 200
        save_model_path = './savedModel/rnn-model'
        self.build_graph(save_model_path, lstm_size=self.lstm_size)

        tf.reset_default_graph()
        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)

        with tf.Session(graph=tf.get_default_graph(),
                        config=tf.ConfigProto(allow_soft_placement=True) #, log_device_placement=True
                        ) as sess:
            graph = self.__load_graph(sess, save_model_path)

            counter = 0
            saver = tf.train.Saver()
            for epoch in range(epochs):
                new_state = sess.run(graph['initial_state'])
                # print(graph['initial_state'])
                loss = 0
                for x, y in self.data:
                    counter += 1
                    start = time()
                    feed = {
                        graph['inputs']: x,
                        graph['targets']: y,
                        graph['keep_prob']: self.keep_prob,
                        graph['initial_state']: new_state
                    }
                    batch_loss, new_state, _ = sess.run([graph['loss'], graph['final_state'], graph['optimizer']],
                                                        feed_dict=feed)
                    end = time()
                    logger.info('Epoch: {}/{}... '.format(epoch+1, epochs) +
                                'Training Step: {}... '.format(counter) +
                                'Training loss: {:.4f}... '.format(batch_loss) +
                                '{:.4f} sec/batch'.format((end-start)))
                    if counter % save_every_n == 0:
                        saver.save(sess, "checkpoints/i{}_l{}.ckpt".format(counter, self.lstm_size))

    def pick_top_n(self, preds, vocab_size, top_n=5):
        """
        Pick one character with high prob
        :param preds:
        :param vocab_size:
        :param top_n:
        :return:
        """
        p = np.squeeze(preds)
        p[np.argsort(p)[:-top_n]] = 0
        p /= np.sum(p)
        c = np.random.choice(vocab_size, 1, p=p)[0]
        return c

    def sample(self, n_samples, prime="The "):
        """
        Writing robot article by given a prime as a start.
        :param n_samples:
        :param prime:
        :return:
        """
        save_model_path = './savedModel/rnn-model-sample'

        self.build_graph(save_model_path=save_model_path, lstm_size=self.lstm_size, sampling=True)
        path_vocab_to_int = "{}/vocab_to_int.npy".format("./data")
        vocab_to_int = None
        if os.path.exists(path_vocab_to_int):
            vocab_to_int = np.load(path_vocab_to_int).item()
        else:
            logger.error("No vocab_to_int data found, please re-run DataCenter.")
            exit()
        assert isinstance(vocab_to_int, dict), "vocab_to_int should be python dict!"

        path_int_to_vocab = "{}/int_to_vocab.npy".format("./data")
        int_to_vocab = None
        if os.path.exists(path_int_to_vocab):
            int_to_vocab = np.load(path_int_to_vocab).item()
        else:
            logger.error("No int_to_vocab data found, please re-run DataCenter.")
            exit()
        assert isinstance(int_to_vocab, dict), "int_to_vocab should be python dict!"

        samples = [c for c in prime]
        tf.reset_default_graph()
        with tf.Session(graph=tf.get_default_graph()) as sess:
            graph = self.__load_graph(sess, save_model_path, sampling=True)
            preds = None
            new_state = sess.run(graph['initial_state'])
            # print(graph['inputs'])
            # print(graph['keep_prob'])
            # print(graph['initial_state'])
            for c in prime:
                x = np.zeros((1, 1))
                x[0, 0] = vocab_to_int[c]
                feed = {
                    graph['inputs']: x,
                    graph['keep_prob']: 1.0,
                    graph['initial_state']: new_state
                }
                preds, new_state = sess.run([graph['predictions'], graph['final_state']], feed_dict=feed)
            c = self.pick_top_n(preds, self.len_vocabulary)
            samples.append(int_to_vocab[c])

            for i in range(n_samples):
                x[0, 0] = c
                feed = {
                    graph['inputs']: x,
                    graph['keep_prob']: 1.,
                    graph['initial_state']: new_state
                }
                preds, new_state = sess.run([graph['predictions'], graph['final_state']], feed_dict=feed)
                c = self.pick_top_n(preds, self.len_vocabulary)
                if c not in int_to_vocab:
                    print("!!!!!!!!!!!!!!!!!!!!!!!!!!")
                samples.append(int_to_vocab[c])

        return ''.join(samples)
