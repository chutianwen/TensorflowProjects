import tensorflow as tf
import os
import numpy as np
import random
from helper import logger, TaskReporter
import shutil
import time
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


class NeuralNetwork:

    # train parameters
    epochs = 10
    window_size = 10

    def __init__(self, data, num_embedding_features=200):
        """
        :param data: input data, already in batches with batch_size defined in DataCenter. This is a generator
        :param num_embedding_features:
        """
        self.data = data
        self.num_embedding_features = num_embedding_features
        try:
            self.int_to_word = np.load("data/int_to_word.npy").item()
            self.num_words = len(self.int_to_word)
        except:
            logger.error("Cannot find int_to_word dict, run DataCenter again...")
            exit()

    def build_graph(self):
        """
        Building the entire training graph including train and validation.
        """
        train_graph = tf.Graph()

        with train_graph.as_default():

            inputs = tf.placeholder(tf.int32, [None], name='inputs')
            labels = tf.placeholder(tf.int32, [None, None], name='labels')

            embedding = tf.Variable(
                tf.random_uniform([self.num_words, self.num_embedding_features], minval=-1, maxval=1),
                name='embedding')

            embed = tf.nn.embedding_lookup(embedding, inputs)

            negative_sampling = 100
            softmax_w = tf.Variable(tf.truncated_normal([self.num_words, self.num_embedding_features], stddev=0.1))
            softmax_bias = tf.Variable(tf.zeros(self.num_words))

            loss = tf.nn.sampled_softmax_loss(softmax_w, softmax_bias, labels, embed, negative_sampling, self.num_words)
            cost = tf.reduce_mean(loss, name='cost')
            optimzer = tf.train.AdamOptimizer(name='optimizer').minimize(cost)

            ############# validation steps #############
            # Random set of words to evaluate similarity on.
            valid_size = 16
            valid_window = 100
            # pick 8 samples from (0,100) and (1000,1100) each ranges. lower id implies more frequent
            valid_examples = np.array(random.sample(range(valid_window), valid_size // 2))
            valid_examples = np.append(valid_examples,
                                       random.sample(range(1000, 1000 + valid_window), valid_size // 2))
            # fixed words for comparing to the updated embedding matrix
            valid_dataset = tf.constant(valid_examples, dtype=tf.int32, name='valid_dataset')

            # Use the cosine distance:
            norm = tf.sqrt(tf.reduce_sum(tf.square(embedding), 1, keep_dims=True))
            normalized_embedding = embedding / norm
            normalized_embedding = tf.identity(normalized_embedding, name='normalized_embedding')

            valid_embedding = tf.nn.embedding_lookup(normalized_embedding, valid_dataset)
            similarity = tf.matmul(valid_embedding, tf.transpose(normalized_embedding), name='similarity')

            if not os.path.exists("./savedModel"):
                os.mkdir("./savedModel")

            saver = tf.train.Saver()
            with tf.Session(graph=train_graph) as sess:
                sess.run(tf.global_variables_initializer())
                saver.save(sess, "./savedModel/word2vec-model")

    def validation(self, similarity, valid_dataset, valid_size=16):
        """
        Printing the k-closet words to validation set.
        :param similarity: tensor cosine similarities between validation words and all other words
        :param valid_dataset:
        :param valid_size:
        :return:
        """
        valid_examples = valid_dataset.eval()
        sim = similarity.eval()
        for i in range(valid_size):
            # get the actual string name of the word from token
            valid_word = self.int_to_word[valid_examples[i]]
            top_k = 8  # number of nearest neighbors
            # 0 is itself, so starting from 1, get the top k's index
            nearest = (-sim[i, :]).argsort()[1:top_k + 1]
            log = 'Nearest to %s:' % valid_word
            for k in range(top_k):
                close_word = self.int_to_word[nearest[k]]
                log = '%s %s,' % (log, close_word)
            print(log)

    @TaskReporter("Train dataset")
    def train(self):
        """
        Training step. Printout the stats every 1000 iterations.
        """
        save_model_path = './savedModel/word2vec-model'
        if not os.path.exists("{}.meta".format(save_model_path)):
            logger.info("No graph can be loaded, so create a new graph...")
            self.build_graph()

        check_point_path = './checkpoints/text8.ckpt'
        if not os.path.exists('./checkpoints'):
            os.mkdir('./checkpoints')

        loaded_graph = tf.Graph()

        inputs, labels, cost, optimizer, similarity, valid_dataset, normalized_embedding = [None]*7
        with tf.device('/gpu:0'):
            with tf.Session(graph=loaded_graph) as sess:
                try:
                    loader = tf.train.import_meta_graph(save_model_path + '.meta')
                    loader.restore(sess, save_model_path)
                    # saver cannot be called before reloading graph, otherwise, no tensors in such session
                    # claimed or reloaded. Nothing to save.
                    saver = tf.train.Saver()
                    inputs = loaded_graph.get_tensor_by_name("inputs:0")
                    labels = loaded_graph.get_tensor_by_name("labels:0")
                    similarity = loaded_graph.get_tensor_by_name("similarity:0")
                    cost = loaded_graph.get_tensor_by_name('cost:0')
                    valid_dataset = loaded_graph.get_tensor_by_name("valid_dataset:0")
                    normalized_embedding = loaded_graph.get_tensor_by_name("normalized_embedding:0")
                    embedding = loaded_graph.get_tensor_by_name("embedding:0")
                    optimizer = loaded_graph.get_operation_by_name("optimizer")
                except Exception as e:
                    logger.error("Something is missing from the previous saved graph, remove it and regenerate graph")
                    shutil.rmtree("./savedModel")
                    exit()

                iteration = 1
                loss = 0
                sess.run(tf.global_variables_initializer())
                for e in range(1, NeuralNetwork.epochs + 1):
                    start = time.time()
                    for x, y in self.data:

                        feed = {
                            inputs: x,
                            labels: np.array(y)[:, None]
                        }
                        train_loss, _ = sess.run([cost, optimizer], feed_dict=feed)
                        loss += train_loss

                        if iteration % 100 == 0:
                            end = time.time()
                            logger.info("Epoch {}/{}\t".format(e, NeuralNetwork.epochs) +
                                        "Iteration: {}\t".format(iteration) +
                                        "Avg. Training loss: {:.4f}\t".format(loss / 100) +
                                        "{:.4f} sec/batch".format((end - start) / 100))
                            loss = 0
                            start = time.time()

                        if iteration % 1000 == 0:
                            # note that this is expensive (~20% slowdown if computed every 500 steps)
                            self.validation(similarity, valid_dataset)
                        iteration += 1

                # Save the checkpoint
                saver.save(sess, check_point_path)

    def visualizing_words(self):
        """
        Project the words into 2D coordinate by sklearn. Show relationship between words
        :return:
        """
        save_model_path = './savedModel/word2vec-model'
        # check_point_path = './checkpoints/text8.ckpt'
        loaded_graph = tf.Graph()
        with tf.Session(graph=loaded_graph) as sess:
            loader = tf.train.import_meta_graph(save_model_path + '.meta')
            loader.restore(sess, save_model_path)
            loader.restore(sess, tf.train.latest_checkpoint('checkpoints'))

            embedding = loaded_graph.get_tensor_by_name("embedding:0")
            embed_mat = sess.run(embedding)

        viz_words = 500
        tsne = TSNE()
        embed_tsne = tsne.fit_transform(embed_mat[:viz_words, :])
        fig, ax = plt.subplots(figsize=(14, 14))
        for idx in range(viz_words):
            plt.scatter(*embed_tsne[idx, :], color='steelblue')
            plt.annotate(self.int_to_word[idx], (embed_tsne[idx, 0], embed_tsne[idx, 1]), alpha=0.7)
            plt.show()
