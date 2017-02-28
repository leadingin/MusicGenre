# -*- coding:utf-8 -*-

"""
@author: Songgx
@file: 1103_mutilayerRNN.py
@time: 2017/1/15 18:37
"""
import tensorflow as tf
import time
import cells
import numpy as np

# data info
x_height = 96
x_width = 1366
n_tags = 50

# Network Parameters
n_input = x_width # data input (shape: 96*1366)
n_steps = x_height # timesteps
n_hidden = 256# hidden layer num of features
batch_size = 100
training_num = 14600
validation_num = 1629
test_num = 6499

def read_and_decode(filename):
    filename_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'features_mel': tf.FixedLenFeature([], tf.string),
                                           'label': tf.FixedLenFeature([n_tags], tf.float32),
                                       })

    x = tf.decode_raw(features['features_mel'], tf.float32)
    x = tf.reshape(x, [x_height, x_width])
    y = tf.cast(features['label'], tf.float32)
    return x, y

def get_top_50_tags(top_50_tags_index, tags_batch_val):
    result=[]
    for row in tags_batch_val:
        result_row=[]
        for index in range(len(row)): # 189
            if index in top_50_tags_index:
                result_row.append(row[index])
        result.append(result_row)
    return np.array(result)

def reset_graph():
    if 'sess' in globals() and sess:
        sess.close()
    tf.reset_default_graph()

def build_multilayer_graph_with_custom_cell(
    cell_type = None,
    num_weights_for_custom_cell = 5,
    state_size = n_hidden,
    num_classes = n_tags,
    batch_size = batch_size,
    num_steps = x_width,
    num_layers = 3,
    learning_rate = 1e-4):

    reset_graph()

    x = tf.placeholder(tf.int32, [batch_size, num_steps], name='input_placeholder')
    y = tf.placeholder(tf.int32, [batch_size, n_tags], name='labels_placeholder')

    embeddings = tf.get_variable('embedding_matrix', [num_classes, state_size])

    rnn_inputs = tf.nn.embedding_lookup(embeddings, x)

    if cell_type == 'PGRUCell':
        cell = cells.PGRUCell(state_size)
    elif cell_type == 'PhasedLSTMCell':
        cell = cells.PhasedLSTMCell(state_size, use_peepholes=True, state_is_tuple=True)
    elif cell_type == 'GRU':
        cell = tf.nn.rnn_cell.GRUCell(state_size)
    elif cell_type == 'LSTM':
        cell = tf.nn.rnn_cell.LSTMCell(state_size, state_is_tuple=True)
    else:
        cell = tf.nn.rnn_cell.BasicRNNCell(state_size)

    if cell_type == 'LSTM':
        cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
    elif cell_type == 'PLSTM':
        cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
    else:
        cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers)

    init_state = cell.zero_state(batch_size, tf.float32)
    rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, rnn_inputs, initial_state=init_state)

    with tf.variable_scope('softmax'):
        W = tf.get_variable('W', [state_size, num_classes])
        b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))

    #reshape rnn_outputs and y
    rnn_outputs = tf.reshape(rnn_outputs, [-1, state_size])
    y_reshaped = tf.reshape(y, [-1])

    logits = tf.matmul(rnn_outputs, W) + b

    total_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, y_reshaped))
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)

    return dict(
        x = x,
        y = y,
        init_state = init_state,
        final_state = final_state,
        total_loss = total_loss,
        train_step = train_step
    )


# mtt_mel_training
mel_features_training, tags_training = read_and_decode("data/merge/mtt_mel_training.tfrecords")
mel_features_batch_training, tags_batch_training = tf.train.shuffle_batch([mel_features_training, tags_training],
                                                        batch_size=batch_size, capacity=2000,
                                                        min_after_dequeue=1000)
# mtt_mel_validation
mel_features_validation, tags_validation = read_and_decode("data/merge/mtt_mel_validation.tfrecords")
mel_features_batch_validation, tags_batch_validation = tf.train.shuffle_batch([mel_features_validation, tags_validation],
                                                        batch_size=batch_size, capacity=2000,
                                                        min_after_dequeue=1000)
# mtt_mel_test
mel_features_test, tags_test = read_and_decode("data/merge/mtt_mel_test.tfrecords")
mel_features_batch_test, tags_batch_test = tf.train.shuffle_batch([mel_features_test, tags_test],
                                                        batch_size=batch_size, capacity=2000,
                                                        min_after_dequeue=1000)

t = time.time()

g = build_multilayer_graph_with_custom_cell(cell_type='LSTM')
t = time.time()
num_training_epochs = int(training_num/batch_size)
tf.set_random_seed(2345)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    training_losses = []
    steps = 0
    for idx in range(num_training_epochs):
        training_loss = 0
        training_state = None
        steps += 1
        audio_batch_vals, label_batch_vals = sess.run([mel_features_batch_training, tags_batch_training])
        feed_dict = {g['x']: audio_batch_vals, g['y']: label_batch_vals}
        if training_state is not None:
            feed_dict[g['init_state']] = training_state
        training_loss_, training_state, _ = sess.run([g['total_loss'],
                                                      g['final_state'],
                                                      g['train_step']],
                                                     feed_dict)
        training_loss += training_loss_
        print("Average training loss for Epoch", idx, ":", training_loss / steps)
        training_losses.append(training_loss / steps)

   # if isinstance(save, str):
    #    g['saver'].save(sess, save)

print("It took", time.time() - t, "seconds to train for 3 epochs.")



'''

g = build_multilayer_graph_with_custom_cell(cell_type='Custom', num_steps=30)
t = time.time()
train_network(g, 5, num_steps=30)
print("It took", time.time() - t, "seconds to train for 5 epochs.")
'''


def multiPLSTM(input, batch_size, lens, n_layers, units_p_layer, n_input, cell, initial_states=None):
    """
    Function to build multilayer PLSTM
    :param input: 3D tensor, where the time input is appended and represents the last feature of the tensor
    :param batch_size: integer, batch size
    :param lens: 2D tensor, length of the sequences in the batch (for synamic rnn use)
    :param n_layers: integer, number of layers
    :param units_p_layer: integer, number of units per layer
    :param n_input: integer, number of features in the input (without time feature)
    :param initial_states: list of tuples of initial states
    :return: 3D tensor, output of the multilayer PLSTM
    """
    if initial_states is None:
        initial_states = [tf.nn.rnn_cell.LSTMStateTuple(tf.zeros([batch_size, units_p_layer], tf.float32),
                                                        tf.zeros([batch_size, units_p_layer], tf.float32))
                          for _ in range(n_layers)]

    assert (len(initial_states) == n_layers)
    times = tf.slice(input, [0, 0, n_input], [-1, -1, 1])
    newX = tf.slice(input, [0, 0, 0], [-1, -1, n_input])

    for k in range(n_layers):
        newX = tf.concat(2, [newX, times])
        with tf.variable_scope("{}".format(k)):
            cell = cell
            outputs, initial_states[k] = tf.nn.dynamic_rnn(cell, newX, dtype=tf.float32,
                                                           sequence_length=lens,
                                                           initial_state=initial_states[k])
            newX = outputs
    return newX, initial_states


