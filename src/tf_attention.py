from __future__ import print_function, division
import os
import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn
from tqdm import tqdm
from utils import batch_generator, get_max_len_info
from data_prep_for_visualization import prepare_data


# Define some parameters
path = os.getcwd()[:os.getcwd().rfind('/')]
MODEL_PATH = path + '/models/tf_attention/'
BATCH_SIZE = 50
EPOCHS = 2
EMBEDDING_DIM = 100
HIDDEN_UNITS = 150
ATTENTION_UNITS = 50
KEEP_PROB = 0.8
DELTA = 0.5
SHUFFLE = False

# Get the data
X_train, y_train, X_test, y_test, vocabulary_size, tokenizer, max_tweet_length \
    = prepare_data(shuffle=SHUFFLE, labels_to_categorical=False)

# Get the word to index and the index to word mappings
word_index = tokenizer.word_index
index_to_word = {index: word for word, index in word_index.items()}

# Set the sequence length
SEQUENCE_LENGTH = max_tweet_length


# This is piece of code is Copyright (c) 2017 to Ilya Ivanov and grants permission under MIT Licence
# https://github.com/ilivans/tf-rnn-attention/blob/master/attention.py
# Implementation as proposed by Yang et al. in "Hierarchical Attention Networks for Document Classification" (2016)
def attention(inputs, attention_size, time_major=False, return_alphas=False):
    if isinstance(inputs, tuple):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        inputs = tf.concat(inputs, 2)

    if time_major:
        # (T,B,D) => (B,T,D)
        inputs = tf.array_ops.transpose(inputs, [1, 0, 2])

    hidden_size = inputs.shape[2].value  # D value - hidden size of the RNN layer

    # Trainable parameters
    w_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
    b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
    u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

    with tf.name_scope('v'):
        # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
        #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
        v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)

    # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
    vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
    alphas = tf.nn.softmax(vu, name='alphas')  # (B,T) shape

    # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
    output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)

    if not return_alphas:
        return output
    else:
        return output, alphas


def build_attention_model():
    # Different placeholders
    with tf.name_scope('Inputs'):
        batch_ph = tf.placeholder(tf.int32, [None, SEQUENCE_LENGTH], name='batch_ph')
        target_ph = tf.placeholder(tf.float32, [None], name='target_ph')
        seq_len_ph = tf.placeholder(tf.int32, [None], name='seq_len_ph')
        keep_prob_ph = tf.placeholder(tf.float32, name='keep_prob_ph')

    # Embedding layer
    with tf.name_scope('Embedding_layer'):
        embeddings_var = tf.Variable(tf.random_uniform([vocabulary_size, EMBEDDING_DIM], -1.0, 1.0), trainable=True)
        tf.summary.histogram('embeddings_var', embeddings_var)
        batch_embedded = tf.nn.embedding_lookup(embeddings_var, batch_ph)

    # (Bi-)RNN layer(-s)
    rnn_outputs, _ = bi_rnn(GRUCell(HIDDEN_UNITS), GRUCell(HIDDEN_UNITS),
                            inputs=batch_embedded, sequence_length=seq_len_ph, dtype=tf.float32)
    tf.summary.histogram('RNN_outputs', rnn_outputs)

    # Attention layer
    with tf.name_scope('Attention_layer'):
        attention_output, alphas = attention(rnn_outputs, ATTENTION_UNITS, return_alphas=True)
        tf.summary.histogram('alphas', alphas)

    # Dropout
    drop = tf.nn.dropout(attention_output, keep_prob_ph)

    # Fully connected layer
    with tf.name_scope('Fully_connected_layer'):
        W = tf.Variable(
            tf.truncated_normal([HIDDEN_UNITS * 2, 1], stddev=0.1))  # Hidden size is multiplied by 2 for Bi-RNN
        b = tf.Variable(tf.constant(0., shape=[1]))
        y_hat = tf.nn.xw_plus_b(drop, W, b)
        y_hat = tf.squeeze(y_hat)
        tf.summary.histogram('W', W)

    with tf.name_scope('Metrics'):
        # Cross-entropy loss and optimizer initialization
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_hat, labels=target_ph))
        tf.summary.scalar('loss', loss)
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)

        # Accuracy metric
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(tf.sigmoid(y_hat)), target_ph), tf.float32))
        tf.summary.scalar('accuracy', accuracy)

    merged = tf.summary.merge_all()

    # Batch generators
    train_batch_generator = batch_generator(X_train, y_train, BATCH_SIZE)
    test_batch_generator = batch_generator(X_test, y_test, BATCH_SIZE)
    session_conf = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    saver = tf.train.Saver()
    return batch_ph, target_ph, seq_len_ph, keep_prob_ph, alphas, loss, accuracy, optimizer, merged, \
           train_batch_generator, test_batch_generator, session_conf, saver


if __name__ == '__main__':
    batch_ph, target_ph, seq_len_ph, keep_prob_ph, alphas, loss, accuracy, optimizer, merged, \
    train_batch_generator, test_batch_generator, session_conf, saver = build_attention_model()

    with tf.Session(config=session_conf) as sess:
        sess.run(tf.global_variables_initializer())
        print("Start learning...")
        for epoch in range(EPOCHS):
            loss_train = 0
            loss_test = 0
            accuracy_train = 0
            accuracy_test = 0

            # Training
            num_batches = X_train.shape[0] // BATCH_SIZE
            for b in tqdm(range(num_batches)):
                x_batch, y_batch = next(train_batch_generator)
                seq_lists = []
                for x in x_batch:
                    if 0 not in list(x):
                        seq_lists.append(SEQUENCE_LENGTH)
                    else:
                        seq_lists.append(list(x).index(0) + 1)
                seq_len = np.array(seq_lists)
                loss_tr, acc, _, summary = sess.run([loss, accuracy, optimizer, merged],
                                                    feed_dict={batch_ph: x_batch,
                                                               target_ph: y_batch,
                                                               seq_len_ph: seq_len,
                                                               keep_prob_ph: KEEP_PROB})
                accuracy_train += acc
                loss_train = loss_tr * DELTA + loss_train * (1 - DELTA)
            accuracy_train /= num_batches

            # Testing
            num_batches = X_test.shape[0] // BATCH_SIZE
            for batch in tqdm(range(num_batches)):
                x_batch, y_batch = next(test_batch_generator)
                seq_lists = []
                for x in x_batch:
                    if 0 not in list(x):
                        seq_lists.append(SEQUENCE_LENGTH)
                    else:
                        seq_lists.append(list(x).index(0) + 1)
                seq_len = np.array(seq_lists)
                loss_test_batch, acc, summary = sess.run([loss, accuracy, merged],
                                                         feed_dict={batch_ph: x_batch,
                                                                    target_ph: y_batch,
                                                                    seq_len_ph: seq_len,
                                                                    keep_prob_ph: 1.0})
                accuracy_test += acc
                loss_test += loss_test_batch
            accuracy_test /= num_batches
            loss_test /= num_batches

            print("loss: {:.3f}, val_loss: {:.3f}, acc: {:.3f}, val_acc: {:.3f}".format(
                loss_train, loss_test, accuracy_train, accuracy_test))
        saver.save(sess, MODEL_PATH)
