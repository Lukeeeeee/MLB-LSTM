'''
LSTM-MLB 1.0
One layer width 10 lstm
'''


from __future__ import print_function
import config
import tensorflow as tf
import numpy as np
import random

raw_x = []
raw_y = []



def prepare_dataset(dataset_file, neighbor_set, sequence_length, offset):
    file_in = open(dataset_file, "r")
    data = file_in.readlines()
    global raw_x
    global raw_y
    for index, line in enumerate(data):
        item = line.split(" ")
        if ((index == 0) or len(item) == 1):
            total_sum = int(item[0])
        else:
            sample = []
            for i in range(len(item)):
                bs_id = int(item[i])
                bs_id_index = neighbor_set.index(bs_id)
                sample.append(bs_id_index)
            raw_x.append(sample[0 : sequence_length])
            raw_y.append(sample[offset : sequence_length + offset])
    return raw_x, raw_y

def return_batch_dataset(batch_size, batch_count, index):
    start = index * batch_size
    end = start + batch_size
    x = raw_x[start : end]
    y = raw_y[start : end]
    return x, y


def construct_and_train(epoch):
    x = tf.placeholder(tf.int64, [config.batch_size, config.sequence_length], name='input_placeholder')
    y = tf.placeholder(tf.int64, [config.batch_size, config.sequence_length], name ='output_placehodler')

    x_one_hot = tf.one_hot(x, config.neighbor_count)
    rnn_inputs = tf.unpack(x_one_hot, axis = 1)

    init_state = tf.zeros([config.batch_size, config.state_size])

    cell = tf.nn.rnn_cell.BasicRNNCell(config.state_size)
    rnn_outputs, final_state = tf.nn.rnn(cell, rnn_inputs, initial_state = init_state)

    with tf.variable_scope('softmax'):
        W = tf.get_variable('W', [config.state_size, config.neighbor_count])
        b = tf.get_variable('b', [config.neighbor_count], initializer=tf.constant_initializer(0.0))

    logits = [tf.matmul(rnn_output, W) + b for rnn_output in rnn_outputs]
    #print("logits", logits)
    predication = [tf.nn.softmax(logit) for logit in logits]
    #print("predication", predication)

    results = tf.transpose([tf.argmax(predication_, 1) for predication_ in predication])
    correct_pred = tf.equal(results, y)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    y_as_list = [tf.squeeze(i, squeeze_dims = [1]) for i in tf.split(1, config.sequence_length, y)]

    loss_weights = [tf.ones([config.batch_size]) for i in range(config.sequence_length)]
    losses = tf.nn.seq2seq.sequence_loss_by_example(logits, y_as_list, loss_weights)

    total_loss = tf.reduce_mean(losses)
    train_step = tf.train.AdagradOptimizer(config.learning_rate).minimize(total_loss)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        print ("Train start, total epoch = ", epoch, "batch_count = ", config.batch_count)
        sess.run(init)
        training_state = np.zeros((config.batch_size, config.state_size))
        for i in range(epoch):
            print("Epoch = ", i)
            avg_loss = 0.
            avg_acc = 0.
            for j in range(config.batch_count):
                train_x, train_y = return_batch_dataset(config.batch_size, config.batch_count, j)
                acc, res, loss, training_state, _ = sess.run([accuracy, results, total_loss, final_state, train_step], feed_dict = {x:train_x, y:train_y, init_state:training_state})
                avg_loss += loss / (config.batch_count)
                avg_acc = acc
                #print("train_y", train_y)
                #print("logits", logits)
                #print("results", res)
                #print("correct_pred", correct_pred)
            #print ("Loss = ", avg_loss, "prediction = ", pred, "Answer = ", test_y)
                if ((j + 1) % 100 == 0):
                    print("Iter ", '%04d' % (j+1), ", Average Loss= " + "{:.6f}".format(avg_loss), ", Training Accuracy= ", "{:.5f}".format(avg_acc))
                    avg_loss = 0
                    avg_acc = 0                    
            batch_index = random.randint(0, config.batch_count-1)
            test_x, test_y = return_batch_dataset(config.batch_size, config.batch_count, batch_index)
            res, correct, acc = sess.run([results, correct_pred, accuracy], feed_dict = {x:test_x, y:test_y})
            print("res", res)
            print("ans_y")
            for ans_y in test_y:
                print(ans_y)
            print("correct", correct)
            print("acc", acc)
        print("Optimization Finished!")
   










