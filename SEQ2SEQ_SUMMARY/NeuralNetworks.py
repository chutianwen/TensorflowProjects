import tensorflow as tf
import numpy as np
import os
from helper import logger
from tensorflow.python.layers.core import Dense


class NeuralNetworks:
    def __init__(self, data, para_dict):

        self.data_path = "data"
        self.lstm_size = para_dict['lstm_size']
        self.lstm_layers = para_dict['lstm_layers']
        self.embedding_size = para_dict['embedding_size']
        self.batch_size = para_dict['batch_size']
        self.epochs = para_dict['epochs']
        self.keep_prob = para_dict['keep_prob']
        self.learning_rate = para_dict['learning_rate']

        self.source_train, self.target_train = data[0][self.batch_size:], data[1][self.batch_size:]
        self.source_valid, self.target_valid = data[0][:self.batch_size], data[1][:self.batch_size]
        path_target_int_to_vocab = "{}/target_int_to_vocab.npy".format(self.data_path)
        path_target_vocab_to_int = "{}/target_vocab_to_int.npy".format(self.data_path)
        path_source_int_to_vocab = "{}/source_int_to_vocab.npy".format(self.data_path)
        path_source_vocab_to_int = "{}/source_vocab_to_int.npy".format(self.data_path)


        if os.path.exists(path_target_int_to_vocab) and os.path.exists(path_target_vocab_to_int) and \
                os.path.exists(path_source_vocab_to_int) and os.path.exists(path_source_int_to_vocab):
            self.target_int_to_vocab = np.load(path_target_int_to_vocab).item()
            self.target_vocab_to_int = np.load(path_target_vocab_to_int).item()
            self.source_vocab_to_int = np.load(path_source_vocab_to_int).item()
            self.source_int_to_vocab = np.load(path_source_int_to_vocab).item()
            assert isinstance(self.source_vocab_to_int, dict)
            self.source_vocab_size = len(self.source_vocab_to_int)
            assert isinstance(self.target_vocab_to_int, dict)
            self.target_vocab_size = len(self.target_vocab_to_int)
        else:
            logger.error("No vocab_to_int data found, please re-run DataCenter.")
            exit()

    def build_inputs(self):
        inputs = tf.placeholder(tf.int32, [None, None], name='inputs')
        targets = tf.placeholder(tf.int32, [None, None], name='targets')
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')

        target_sequence_length = tf.placeholder(tf.int32, (None,), name='target_sequence_length')
        max_target_sequence_length = tf.reduce_max(target_sequence_length, name='max_target_length')
        source_sequence_length = tf.placeholder(tf.int32, (None,), name='source_sequence_length')

        return inputs, targets, keep_prob, learning_rate, target_sequence_length, max_target_sequence_length, \
               source_sequence_length

    def build_encode_layer(self, source, source_sequence_length, keep_prob):

        limit = 1 / np.sqrt(self.source_vocab_size)
        embed = tf.contrib.layers.embed_sequence(source, self.source_vocab_size, self.embedding_size,
                                                 initializer=tf.random_uniform_initializer(-limit, limit, seed=2))

        def build_cell():
            lstm_cell = tf.contrib.rnn.LSTMCell(self.lstm_size,
                                                initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
            # Add dropout to the cell
            drop = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob)
            return drop

        cell = tf.contrib.rnn.MultiRNNCell([build_cell() for _ in range(self.lstm_layers)])
        outputs, final_state = tf.nn.dynamic_rnn(cell, embed, sequence_length=source_sequence_length, dtype=tf.float32)
        return final_state

    def process_decoder_input(self, targets):
        ending = tf.strided_slice(targets, [0, 0], [self.batch_size, -1], [1, 1])
        assert isinstance(self.target_vocab_to_int, dict), 'target_vocab_to_int is not dict'
        dec_input = tf.concat([tf.fill([self.batch_size, 1], self.target_vocab_to_int['<GO>']), ending], 1)
        return dec_input

    def build_decoding_layer(self, input, target_sequence_length, encode_state, max_target_sequence_length, keep_prob):

        limit = 1 / np.sqrt(self.target_vocab_size)
        embedding = tf.Variable(
            tf.random_uniform([self.target_vocab_size, self.embedding_size], minval=-limit, maxval=limit)
        )
        embed = tf.nn.embedding_lookup(embedding, input)

        def build_cell():
            lstm_cell = tf.contrib.rnn.LSTMCell(self.lstm_size,
                                                initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
            # Add dropout to the cell
            drop = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob)
            return drop

        cell = tf.contrib.rnn.MultiRNNCell([build_cell() for _ in range(self.lstm_layers)])

        output_layer = Dense(units=self.target_vocab_size,
                             kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

        with tf.variable_scope("decode"):
            training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=embed,
                                                                sequence_length=target_sequence_length,
                                                                time_major=False)
            training_decoder = tf.contrib.seq2seq.BasicDecoder(cell,
                                                               training_helper,
                                                               encode_state,
                                                               output_layer
                                                               )
            training_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=training_decoder,
                                                                              impute_finished=True,
                                                                              maximum_iterations=max_target_sequence_length)
        with tf.variable_scope('decode', reuse=True):
            start_tokens = tf.tile(tf.constant([self.target_vocab_to_int['<GO>']], dtype=tf.int32), [self.batch_size],
                                   name='start_tokens')
            inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding,
                                                                        start_tokens,
                                                                        self.target_vocab_to_int['<EOS>'])
            inference_decoder = tf.contrib.seq2seq.BasicDecoder(cell,
                                                                inference_helper,
                                                                encode_state,
                                                                output_layer)
            inference_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(inference_decoder,
                                                                               impute_finished=True,
                                                                               maximum_iterations=max_target_sequence_length)

        return training_decoder_output, inference_decoder_output

    def build_graph(self, save_model_path):
        if os.path.exists("{}.meta".format(save_model_path)):
            logger.info("Graph existed, ready to be reloaded...")
        else:
            tf.reset_default_graph()
            train_graph = tf.Graph()
            with train_graph.as_default():
                inputs, targets, keep_prob, learning_rate, target_sequence_length, max_target_sequence_length, \
                source_sequence_length = self.build_inputs()

                encode_state = self.build_encode_layer(inputs, source_sequence_length, keep_prob)
                decoder_inputs = self.process_decoder_input(targets)
                training_decoder_output, inference_decoder_output = self.build_decoding_layer(decoder_inputs,
                                                                                              target_sequence_length,
                                                                                              encode_state,
                                                                                              max_target_sequence_length,
                                                                                              keep_prob)
                training_logits = tf.identity(training_decoder_output.rnn_output, 'logits')
                inference_predictions = tf.identity(inference_decoder_output.sample_id, name='predictions')

                masks = tf.sequence_mask(target_sequence_length, max_target_sequence_length, dtype=tf.float32,
                                         name='masks')
                with tf.name_scope("optimization"):
                    cost = tf.contrib.seq2seq.sequence_loss(training_logits, targets, masks, name='cost')
                    # loss = tf.identity(cost, name='cost')

                    optimizer = tf.train.AdamOptimizer(learning_rate)
                    gradients = optimizer.compute_gradients(cost)
                    clipped_gradients = [(tf.clip_by_value(gra, -5., 5.), var) for gra, var in gradients if
                                         gra is not None]
                    train_op = optimizer.apply_gradients(clipped_gradients, name='optimizer')

                saver = tf.train.Saver()
            with tf.Session(graph=train_graph) as sess:
                sess.run(tf.global_variables_initializer())
                saver.save(sess, save_model_path)

    def get_batches(self, source, target, source_pad_int, target_pad_int):
        def pad_sentence_batch(sentence_batch, pad_int):
            max_sentence_length = max([len(sentence) for sentence in sentence_batch])
            sentence_batch_pad = [sentence + [pad_int] * (max_sentence_length - len(sentence))
                                  for sentence in sentence_batch]
            return sentence_batch_pad

        num_batches = len(source) // self.batch_size
        for id in range(0, num_batches):
            start = id * self.batch_size
            x, y = source[start: start + self.batch_size], target[start: start + self.batch_size]
            x_pad, y_pad = pad_sentence_batch(x, source_pad_int), pad_sentence_batch(y, target_pad_int)
            inputs = np.array(x_pad)
            targets = np.array(y_pad)

            source_sequence_length = list(map(len, inputs))
            target_sequence_length = list(map(len, targets))

            yield inputs, targets, source_sequence_length, target_sequence_length

    def load_graph(self, sess, save_model_path):

        graph = {}
        print("Saved_model_path:{}".format(save_model_path))
        loader = tf.train.import_meta_graph(save_model_path + '.meta')

        graph['inputs'] = sess.graph.get_tensor_by_name("inputs:0")
        graph['targets'] = sess.graph.get_tensor_by_name("targets:0")
        graph['keep_prob'] = sess.graph.get_tensor_by_name("keep_prob:0")
        graph['learning_rate'] = sess.graph.get_tensor_by_name("learning_rate:0")
        graph['source_sequence_length'] = sess.graph.get_tensor_by_name("source_sequence_length:0")
        graph['target_sequence_length'] = sess.graph.get_tensor_by_name("target_sequence_length:0")
        graph['prediction'] = sess.graph.get_tensor_by_name("predictions:0")

        graph['cost'] = sess.graph.get_tensor_by_name('optimization/cost/truediv:0')
        graph['optimizer'] = sess.graph.get_operation_by_name("optimization/optimizer")

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

    def valid_cost(self, sess, graph, epoch, batch_i, train_cost):
        for inputs, targets, source_sequence_length, target_sequence_length in \
                self.get_batches(self.source_valid, self.target_valid, self.source_vocab_to_int['<PAD>'],
                                 self.target_vocab_to_int['<PAD>']):
            validation_loss = sess.run(graph['cost'], feed_dict={
                graph['inputs']: inputs,
                graph['targets']: targets,
                graph['keep_prob']: self.keep_prob,
                graph['source_sequence_length']: source_sequence_length,
                graph['target_sequence_length']: target_sequence_length
            })
            logger.info('Epoch {:>3}/{} Batch {:>4}/{} - Loss: {:>6.3f}  - Validation loss: {:>6.3f}'
                        .format(epoch,
                                self.epochs,
                                batch_i,
                                len(self.source_train) // self.batch_size,
                                train_cost,
                                validation_loss))

    def train(self):
        display_iterations = 10
        save_model_path = './savedModel/seq2seq-model'
        self.build_graph(save_model_path)
        tf.reset_default_graph()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        with tf.Session(graph=tf.get_default_graph(), config=config) as sess:
            graph = self.load_graph(sess, save_model_path)
            saver = tf.train.Saver()
            logger.info("Start training...")
            for epoch in range(self.epochs):
                for batch_i, (inputs, targets, source_sequence_length, target_sequence_length) in \
                        enumerate(self.get_batches(self.source_train, self.target_train,
                                                   self.source_vocab_to_int['<PAD>'],
                                                   self.target_vocab_to_int['<PAD>']
                                                   ), 1):
                    # print(graph['inputs'])
                    # print(inputs)
                    # print(graph['targets'])
                    # print(targets)
                    # print(graph['source_sequence_length'])
                    # print(source_sequence_length)
                    # print(graph['target_sequence_length'])
                    # print(target_sequence_length)

                    _, cost = sess.run([graph['optimizer'], graph['cost']], feed_dict={
                        graph['inputs']: inputs,
                        graph['targets']: targets,
                        graph['keep_prob']: self.keep_prob,
                        graph['learning_rate']: self.learning_rate,
                        graph['source_sequence_length']: source_sequence_length,
                        graph['target_sequence_length']: target_sequence_length
                    })
                    if batch_i % display_iterations == 0:
                        self.valid_cost(sess, graph, epoch, batch_i, cost)

                # Save Model
                saver.save(sess, "checkpoints/e{}_l{}.ckpt".format(epoch, self.lstm_size))

    def source_to_seq(self, text):
        sequence_length = 7
        res = [self.source_vocab_to_int.get(word, self.source_vocab_to_int['<UNK>']) for word in text] + \
               [self.source_vocab_to_int['<PAD>']] * (sequence_length - len(text))
        return np.array(res)

    def seq_to_seq(self, text):
        save_model_path = './savedModel/seq2seq-model'
        self.build_graph(save_model_path)
        tf.reset_default_graph()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        text = self.source_to_seq(text)
        with tf.Session(graph=tf.get_default_graph(), config=config) as sess:
            graph = self.load_graph(sess, save_model_path)
            logger.info("Start seq to seq...")
            answer_logits = sess.run(graph['prediction'], feed_dict={
                graph['inputs']: [text]*self.batch_size,
                graph['target_sequence_length']: [len(text)]*self.batch_size,
                graph['source_sequence_length']: [len(text)]*self.batch_size,
                graph['keep_prob']: self.keep_prob
            })[0]

        pad = self.source_vocab_to_int["<PAD>"]

        print('Original Text:', text)

        print('\nSource')
        print('  Word Ids:    {}'.format([i for i in text]))
        print('  Input Words: {}'.format(" ".join([self.source_int_to_vocab[i] for i in text])))

        print('\nTarget')
        print('  Word Ids:       {}'.format([i for i in answer_logits if i != pad]))
        print('  Response Words: {}'.format(" ".join([self.target_int_to_vocab[i] for i in answer_logits if i != pad])))

# import tensorflow.contrib as contrib
# #
# contrib.seq2seq.sequence_loss()
