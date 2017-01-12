'''
LSTM-MLB 2.0
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

    cell = tf.nn.rnn_cell.LSTMCell(config.state_size, state_is_tuple = True)
    cell = tf.nn.rnn_cell.MultiRNNCell([cell] * config.layer_size, state_is_tuple = True)

    init_state = cell.zero_state(config.batch_size, tf.float32)


    print("state\n", init_state)
    rnn_outputs, final_state = tf.nn.rnn(cell, rnn_inputs, initial_state = init_state)

    with tf.variable_scope('softmax'):
        W1 = tf.get_variable('W1', [config.state_size, config.neighbor_count])
        b1 = tf.get_variable('b1', [config.neighbor_count], initializer=tf.constant_initializer(0.0))
    logits = [tf.matmul(rnn_output, W1) + b1 for rnn_output in rnn_outputs]

    #logits = [tf.matmul(logit, W2) + b2 for logit in logits]
    #logit = [tf.slice(logit, [config.sequence_length-1, 0, 0], [1, config.batch_size, config.neighbor_count]) for logit in logits]
    logit = tf.slice(logits, [config.sequence_length-1, 0, 0], [1, config.batch_size, config.neighbor_count])
    logit = tf.reshape(logit, [config.batch_size, config.neighbor_count])
    #logit = logits[config.sequence_length-1]
    
    y_last = tf.slice(y, [0, config.sequence_length-1], [config.batch_size, 1])
    y_last_one_hot = tf.one_hot(y_last, config.neighbor_count)
    y_last_one_hot = tf.reshape(y_last_one_hot, [config.batch_size, config.neighbor_count])

    #Compute acc
    predication = tf.nn.softmax(logit)
    results = tf.argmax(predication, axis=1)
    correct_pred = tf.equal(results, tf.reshape(y_last, [config.batch_size]))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    #Compute loss
    
    loss = tf.nn.softmax_cross_entropy_with_logits(predication, y_last_one_hot)

    #y_as_list = [tf.squeeze(i, squeeze_dims = [1]) for i in tf.split(1, config.sequence_length, y)]
    #loss_weights = [tf.ones([config.batch_size]) for i in range(config.sequence_length)]
    #losses = tf.nn.seq2seq.sequence_loss_by_example(logits, y_as_list, loss_weights)

    total_loss = tf.reduce_mean(loss)
    train_step = tf.train.AdagradOptimizer(config.learning_rate).minimize(total_loss)

    init = tf.global_variables_initializer()


    with tf.Session() as sess:
        print ("Train start, total epoch = ", epoch, "batch_count = ", config.batch_count)
        sess.run(init)
        for i in range(epoch):
            print("Epoch = ", i)
            epoch_loss = 0.
            last_bacht_acc = 0.
            training_state = sess.run(init_state)
            for j in range(config.batch_count):
                batch_index = random.randint(0, config.batch_count-1)
                train_x, train_y = return_batch_dataset(config.batch_size, config.batch_count, j)
                train_acc, train_loss, training_state, _ = sess.run([accuracy, total_loss, final_state, train_step], feed_dict = {x:train_x, y:train_y, init_state:training_state})
                epoch_loss += train_loss / config.batch_count
                last_iter_acc = train_acc
                if ((j + 1) % config.display_step == 0):
                    batch_index = random.randint(0, config.batch_count-1)
                    test_x, test_y = return_batch_dataset(config.batch_size, config.batch_count, batch_index)
                    res, correct, test_acc = sess.run([results, correct_pred, accuracy], feed_dict = {x:test_x, y:test_y})
                    log_output(i, j, epoch_loss, last_iter_acc, test_acc)
                    print("Iter ", '%04d' % (j+1), ", Total Loss= " + "{:.6f}".format(epoch_loss), ", Training Accuracy= ", "{:.5f}".format(last_iter_acc), ", Test Accuracy= ", "{:.5f}".format(test_acc))                    
                    #print("x:\n",train_x)
                    #print("y:\n",train_y)
                    print("logit", sess.run(logit, feed_dict = {x:train_x}))
                    print("predication", sess.run(predication, feed_dict = {x:train_x}))
                    #print("y_last", sess.run(y_last, feed_dict = {y:train_y}))
                    #print("y_last_one_hot", sess.run(y_last_one_hot, feed_dict = {y:train_y}))
                    #print("results\n", sess.run(results, feed_dict = {x:train_x}))
                    avg_loss = 0.
                    avg_acc = 0.
            batch_index = random.randint(0, config.batch_count-1)
            test_x, test_y = return_batch_dataset(config.batch_size, config.batch_count, batch_index)
            res, correct, acc = sess.run([results, correct_pred, accuracy], feed_dict = {x:test_x, y:test_y})
            print("predication = ", res)
            #print(sess.run(y_last, feed_dict = {x:test_x, y:test_y}))
            print("correct = ", correct)
            print("test acc = ", acc)
        print("Optimization Finished!")
   


def log_output(epoch, iter, loss, train_acc, test_acc):
    log_out = open("train.log", "a")
    print("Epoch= ", epoch, "iter= ", iter, "loss = ", loss, "Train acc = ", train_acc, "Test acc = ", test_acc, file = log_out)
    log_out.close()






