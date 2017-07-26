import tensorflow as tf
import pandas as pd
from helper import TaskReporter, logger
import helper
import pickle
import os
import shutil
import random

class NeuralNetwork:

    # Training parameters
    epochs = 20
    batch_size = 128
    keep_probability = 0.75

    def neural_net_image_input(self, image_shape):
        """
        Return a Tensor for a batch of image input
        : image_shape: Shape of the images
        : return: Tensor for image input.
        """
        assert len(image_shape) == 3, 'image shape is not correct'
        features = tf.placeholder(tf.float32, [None, image_shape[0], image_shape[1], image_shape[2]], 'x')
        return features

    def neural_net_label_input(self, n_classes):
        """
        Return a Tensor for a batch of label input
        : n_classes: Number of classes
        : return: Tensor for label input.
        """
        labels = tf.placeholder(tf.int32, [None, n_classes], 'y')
        return labels

    def neural_net_keep_prob_input(self):
        """
        Return a Tensor for keep probability
        : return: Tensor for keep probability.
        """
        dropout = tf.placeholder(tf.float32, name="keep_prob")
        return dropout

    def conv2d_maxpool(self, x_tensor, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides):
        """
        Apply convolution then max pooling to x_tensor
        :param x_tensor: TensorFlow Tensor
        :param conv_num_outputs: Number of outputs for the convolutional layer
        :param conv_ksize: kernal size 2-D Tuple for the convolutional layer
        :param conv_strides: Stride 2-D Tuple for convolution
        :param pool_ksize: kernal size 2-D Tuple for pool
        :param pool_strides: Stride 2-D Tuple for pool
        : return: A tensor that represents convolution and max pooling of x_tensor
        """
        image_shape = x_tensor.get_shape().as_list()
        num_channel = image_shape[-1]
        weights = tf.Variable(tf.random_normal([conv_ksize[0], conv_ksize[1], num_channel, conv_num_outputs]))
        bias = tf.Variable(tf.random_normal([conv_num_outputs]))
        conv = tf.nn.conv2d(x_tensor, weights, [1, conv_strides[0], conv_strides[1], 1], padding="SAME")
        conv = tf.nn.bias_add(conv, bias)
        conv = tf.nn.max_pool(conv, [1, pool_ksize[0], pool_ksize[1], 1], [1, pool_strides[0], pool_strides[1], 1],
                              padding='SAME')
        return conv

    def flatten(self, x_tensor):
        """
        Flatten x_tensor to (Batch Size, Flattened Image Size)
        : x_tensor: A tensor of size (Batch Size, ...), where ... are the image dimensions.
        : return: A tensor of size (Batch Size, Flattened Image Size).
        """
        shape = x_tensor.get_shape().as_list()
        assert len(shape) >= 4, 'shape of image is not correct'
        single_image_dimension = shape[1] * shape[2] * shape[3]
        x_tensor = tf.reshape(x_tensor, [-1, single_image_dimension])
        return x_tensor

    def fully_conn(self, x_tensor, num_outputs):
        """
        Apply a fully connected layer to x_tensor using weight and bias
        : x_tensor: A 2-D tensor where the first dimension is batch size.
        : num_outputs: The number of output that the new tensor should be.
        : return: A 2-D tensor where the second dimension is num_outputs.
        """
        shape = x_tensor.get_shape().as_list()
        weights = tf.Variable(tf.random_normal([shape[-1], num_outputs]))
        biases = tf.Variable(tf.random_normal([num_outputs]))
        logits = tf.add(tf.matmul(x_tensor, weights), biases)
        logits = tf.nn.relu(logits)
        return logits

    def output(self, x_tensor, num_outputs):
        """
        Apply a output layer to x_tensor using weight and bias
        : x_tensor: A 2-D tensor where the first dimension is batch size.
        : num_outputs: The number of output that the new tensor should be.
        : return: A 2-D tensor where the second dimension is num_outputs.
        """
        shape = x_tensor.get_shape().as_list()
        weights = tf.Variable(tf.random_normal([shape[-1], num_outputs]))
        biases = tf.Variable(tf.random_normal([num_outputs]))
        logits = tf.add(tf.matmul(x_tensor, weights), biases)
        return logits

    def conv_net(self, x, keep_prob):
        """
        Create a convolutional neural network model
        : x: Placeholder tensor that holds image data.
        : keep_prob: Placeholder tensor that hold dropout keep probability.
        : return: Tensor that represents logits
        """
        conv_paras = {
            'conv_num_outputs': [32, 24, 16],
            'conv_ksize': [[5, 5], [4, 4], [3, 3]],
            'conv_strides': [[1, 1], [1, 1], [1, 1]],
            'pool_ksize': [[2, 2], [2, 2], [2, 2]],
            'pool_strides': [[2, 2], [2, 2], [2, 2]]
        }
        conv_paras = pd.DataFrame(conv_paras)
        conv = None
        for idx, row in conv_paras.iterrows():
            logger.info("{},{},{},{},{}".format(row['conv_num_outputs'], row['conv_ksize'], row['conv_strides'],
                                                row['pool_ksize'], row['pool_strides']))
            conv_num_outputs = row['conv_num_outputs']
            conv_ksize = row['conv_ksize']
            conv_strides = row['conv_strides']
            pool_ksize = row['pool_ksize']
            pool_strides = row['pool_strides']
            conv = self.conv2d_maxpool(x, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides)
            x = conv

        fully_conn0 = self.flatten(conv)

        fully_connected_paras = {
            'outputs': [512, 1024]
        }
        fully_connected_paras = pd.DataFrame(fully_connected_paras)
        fully_conn = None
        for idx, row in fully_connected_paras.iterrows():
            fully_conn = self.fully_conn(fully_conn0, row['outputs'])
            fully_conn0 = fully_conn
        fully_conn = tf.nn.dropout(fully_conn, keep_prob)

        out = self.output(fully_conn, 10)
        return out

    def build_graph(self):
        """
        Build a CNN training graph and save to disk for reloading
        """
        tf.reset_default_graph()

        # placeholders
        x = self.neural_net_image_input((32, 32, 3))
        y = self.neural_net_label_input(10)
        keep_prob = self.neural_net_keep_prob_input()

        # model
        logits_out = self.conv_net(x, keep_prob)

        # Name logits_out
        logits_out = tf.identity(logits_out, name='logits')

        # loss and optimizer
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_out, labels=y), name='cost')
        optimzer = tf.train.AdamOptimizer(name='optimizer').minimize(loss)

        # Accuracy
        correct_pred = tf.equal(tf.argmax(y, axis=1), tf.argmax(logits_out, axis=1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

        # print(type(tf.Variable(1)))
        saver = tf.train.Saver()
        if not os.path.exists('./savedModel'):
            os.mkdir('./savedModel')
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.save(sess, './savedModel/cnn-model')

    def train_neural_network(self, session, x, y, keep_prob, optimizer, keep_probability, feature_batch, label_batch):
        """
        Optimize the session on a batch of images and labels
        : session: Current TensorFlow session
        : optimizer: TensorFlow optimizer function
        : keep_probability: keep probability
        : feature_batch: Batch of Numpy image data
        : label_batch: Batch of Numpy label data
        """
        session.run(optimizer, feed_dict={x: feature_batch, y: label_batch, keep_prob: keep_probability})
        pass

    def print_stats(self, session, x, y, keep_prob, feature_batch, label_batch, valid_features, valid_labels,
                    cost, accuracy):
        """
        Print information about loss and validation accuracy
        : session: Current TensorFlow session
        : feature_batch: Batch of Numpy image data
        : label_batch: Batch of Numpy label data
        : cost: TensorFlow cost function
        : accuracy: TensorFlow accuracy function
        """
        cost = session.run(cost, feed_dict={x: feature_batch, y: label_batch, keep_prob: 1.0})
        train_accuracy = session.run(accuracy, feed_dict={x: feature_batch, y: label_batch, keep_prob: 1.0})
        valid_accuracy = session.run(accuracy, feed_dict={x: valid_features, y: valid_labels, keep_prob: 1.0})
        logger.info("loss:{}".format(cost))
        logger.info("Train Accuracy:{}".format(train_accuracy))
        logger.info("Validation Accuracy:{}".format(valid_accuracy))
        pass

    @TaskReporter("Train on single batch")
    def train_on_one_batch(self):
        """
        Train the model on one batch and save the session on disk
        """
        save_model_path = './savedModel/cnn-model'
        if not os.path.exists("{}.meta".format(save_model_path)):
            logger.info("No graph can be loaded, so create a new graph...")
            self.build_graph()

        logger.info("model is ready, good to go!")
        dir_processed_data = "./ProcessedData"
        valid_features, valid_labels = pickle.load(open('{}/preprocess_validation.p'.format(dir_processed_data),
                                                        mode='rb'))

        loaded_graph = tf.Graph()
        with tf.device('/gpu:0'):
            with tf.Session(graph=loaded_graph) as sess: #config=tf.ConfigProto(log_device_placement=True)

                try:
                    loader = tf.train.import_meta_graph(save_model_path + '.meta')
                    loader.restore(sess, save_model_path)

                    x = loaded_graph.get_tensor_by_name("x:0")
                    y = loaded_graph.get_tensor_by_name("y:0")
                    keep_prob = loaded_graph.get_tensor_by_name("keep_prob:0")
                    cost = loaded_graph.get_tensor_by_name('cost:0')
                    accuracy = loaded_graph.get_tensor_by_name('accuracy:0')
                    optimizer = loaded_graph.get_operation_by_name("optimizer")
                except Exception as e:
                    logger.error("Something is missing from the previous saved graph, remove it and regenerate graph")
                    shutil.rmtree("./savedModel")
                    exit()

                # Initializing the variables
                sess.run(tf.global_variables_initializer())

                for epoch in range(NeuralNetwork.epochs):
                    batch_i = 1
                    for batch_features, batch_labels in \
                            helper.load_preprocess_training_batch(batch_i, NeuralNetwork.batch_size):
                        self.train_neural_network(sess, x, y, keep_prob, optimizer, NeuralNetwork.keep_probability,
                                                  batch_features, batch_labels)
                    logger.info('\nEpoch {:>2}, CIFAR-10 Batch {}:  '.format(epoch + 1, batch_i))
                    self.print_stats(sess, x, y, keep_prob, batch_features, batch_labels, valid_features, valid_labels,
                                         cost, accuracy)

                # Save Model
                saver = tf.train.Saver()
                saver.save(sess, save_model_path)

    @TaskReporter("Train on whole batches")
    def train_on_whole_data(self):
        save_model_path = './savedModel/cnn-model'
        valid_features, valid_labels = pickle.load(open('{}/preprocess_validation.p'.format(dir_processed_data),
                                                        mode='rb'))
        loaded_graph = tf.Graph()
        with tf.Session(graph=loaded_graph) as sess:
            try:
                loader = tf.train.import_meta_graph(save_model_path + '.meta')
                loader.restore(sess, save_model_path)

                x = loaded_graph.get_tensor_by_name("x:0")
                y = loaded_graph.get_tensor_by_name("y:0")
                keep_prob = loaded_graph.get_tensor_by_name("keep_prob:0")
                cost = loaded_graph.get_tensor_by_name('cost:0')
                accuracy = loaded_graph.get_tensor_by_name('accuracy:0')
                optimizer = loaded_graph.get_operation_by_name("optimizer")
            except Exception as e:
                logger.error("Something is missing from the previous saved graph, remove it and regenerate graph")
                shutil.rmtree("./savedModel")
                exit()

            # Initializing the variables
            sess.run(tf.global_variables_initializer())

            # Training cycle
            for epoch in range(NeuralNetwork.epochs):
                # Loop over all batches
                n_batches = 5
                for batch_i in range(1, n_batches + 1):
                    for batch_features, batch_labels in helper.load_preprocess_training_batch(batch_i,
                                                                                              NeuralNetwork.batch_size):
                        self.train_neural_network(sess, x, y, keep_prob, optimizer, NeuralNetwork.keep_probability,
                                                  batch_features, batch_labels)
                    logger.info('Epoch {:>2}, CIFAR-10 Batch {}:  \n'.format(epoch + 1, batch_i))
                    self.print_stats(sess, x, y, keep_prob, batch_features, batch_labels, valid_features,
                                         valid_labels, cost, accuracy)

            # Save Model
            saver = tf.train.Saver()
            save_path = saver.save(sess, save_model_path)

    @TaskReporter("Test model")
    def test_model(self):

        save_model_path = './savedModel/cnn-model'
        n_samples = 4
        top_n_predictions = 3
        test_features, test_labels = pickle.load(open('./ProcessedData/preprocess_test.p', mode='rb'))
        loaded_graph = tf.Graph()

        with tf.Session(graph=loaded_graph) as sess:

            try:
                loader = tf.train.import_meta_graph(save_model_path + '.meta')
                loader.restore(sess, save_model_path)

                x = loaded_graph.get_tensor_by_name("x:0")
                y = loaded_graph.get_tensor_by_name("y:0")
                keep_prob = loaded_graph.get_tensor_by_name("keep_prob:0")
                logits = loaded_graph.get_tensor_by_name('logits:0')
                cost = loaded_graph.get_tensor_by_name('cost:0')
                accuracy = loaded_graph.get_tensor_by_name('accuracy:0')
                optimizer = loaded_graph.get_operation_by_name("optimizer")
            except Exception as e:
                logger.error("Something is missing from the previous saved graph, remove it and regenerate graph")
                shutil.rmtree("./savedModel")
                exit()

            # Get accuracy in batches for memory limitations
            test_batch_acc_total = 0
            test_batch_count = 0

            for test_feature_batch, test_label_batch in helper.batch_features_labels(test_features, test_labels,
                                                                                     NeuralNetwork.batch_size):
                test_batch_acc_total += sess.run(
                    accuracy,
                    feed_dict={x: test_feature_batch, y: test_label_batch, keep_prob: 1.0})
                test_batch_count += 1

            logger.info('Testing Accuracy: {}\n'.format(test_batch_acc_total/test_batch_count))

            # Print Random Samples
            random_test_features, random_test_labels = tuple(zip(*random.sample(list(zip(test_features, test_labels)),
                                                                                n_samples)))
            random_test_predictions = sess.run(
                tf.nn.top_k(tf.nn.softmax(logits), top_n_predictions),
                feed_dict={x: random_test_features, y: random_test_labels, keep_prob: 1.0})

            helper.display_image_predictions(random_test_features, random_test_labels, random_test_predictions)


