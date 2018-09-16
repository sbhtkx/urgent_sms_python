import os
import os.path as path

import tensorflow as tf
from DataManager import DataManager

from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib

MODEL_NAME = 'one_hidden_layer'
NUM_STEPS = 6000
NUM_HIDDEN_LAYER_NODES = 5
NUM_URGENT_MSGS = 67
NUM_CALM_MSGS = 1043


def model_input(input_node_name, n_features):
    x = tf.placeholder(tf.float32, [None, n_features], name=input_node_name)
    y_ = tf.placeholder(tf.float32, [None, 1])
    return x, y_


def build_model(x,
                y_,
                output_node_name,
                num_features,
                num_hidden_layer_nodes=NUM_HIDDEN_LAYER_NODES,
                num_calm_msgs=NUM_CALM_MSGS,
                num_urgent_msgs=NUM_URGENT_MSGS,
                training_rate=0.1):
    
    # hidden layer
    w1 = tf.Variable(tf.truncated_normal([num_features, num_hidden_layer_nodes], stddev=0.1))
    b1 = tf.Variable(tf.constant(0.1, shape=[num_hidden_layer_nodes]))
    z1 = tf.matmul(x, w1) + b1
    activation1 = tf.nn.relu(z1)
    
    # output layer
    w2 = tf.Variable(tf.truncated_normal([num_hidden_layer_nodes, 1], stddev=0.1))
    b2 = tf.Variable(0.)
    z2 = tf.matmul(activation1, w2) + b2
    outputs = tf.divide(tf.constant(1.0), tf.add(tf.constant(1.0), tf.exp(-z2)), name=output_node_name)

    # loss
    loss = tf.reduce_mean(-((num_calm_msgs / (num_calm_msgs + num_urgent_msgs)) * y_ * tf.log(outputs) +
                            (num_urgent_msgs / (num_calm_msgs + num_urgent_msgs)) * (1 - y_) * tf.log(1 - outputs)))

    # train step
    train_step = tf.train.GradientDescentOptimizer(training_rate).minimize(loss)

    # accuracy
    correct_prediction = tf.equal(outputs, y_)      # tf.argmax(outputs, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    tf.summary.scalar("loss", loss)
    tf.summary.scalar("accuracy", accuracy)
    merged_summary_op = tf.summary.merge_all()

    return train_step, loss, accuracy, merged_summary_op


def train(x, y_, train_step, loss, accuracy, merged_summary_op, saver, data_manager):
    print("training start...")

    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init_op)

        tf.train.write_graph(sess.graph_def, 'out', MODEL_NAME + '.pbtxt', True)

        # op to write logs to Tensorboard
        summary_writer = tf.summary.FileWriter('logs/', graph=tf.get_default_graph())

        for step in range(NUM_STEPS):
            training_data = data_manager.get_training_data()  # batch = mnist.train.next_batch(BATCH_SIZE)
            if step % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict={x: training_data[0], y_: training_data[1]})
                print('step %d, training accuracy %f' % (step, train_accuracy))
                loss_f = loss.eval(feed_dict={x: training_data[0], y_: training_data[1]})
                print('step %d, loss %f' % (step, loss_f))
            _, summary = sess.run([train_step, merged_summary_op], feed_dict={x: training_data[0], y_: training_data[1]}
                                  )
            summary_writer.add_summary(summary, step)

        saver.save(sess, 'out/' + MODEL_NAME + '.chkp')

        test_data = data_manager.get_test_data()
        test_accuracy = accuracy.eval(feed_dict={x: test_data[0], y_: test_data[1]})
        print('test accuracy %g' % test_accuracy)


    print("training finished!")


def export_model(input_node_names, output_node_name):
    print("exporting started...")

    freeze_graph.freeze_graph('out/' + MODEL_NAME + '.pbtxt',
                              "",
                              False,
                              'out/' + MODEL_NAME + '.chkp',
                              output_node_name,
                              "save/restore_all",
                              "save/Const:0",
                              'out/frozen_' + MODEL_NAME + '.pb',
                              True,
                              "")

    input_graph_def = tf.GraphDef()
    with tf.gfile.Open('out/frozen_' + MODEL_NAME + '.pb', "rb") as f:
        input_graph_def.ParseFromString(f.read())

    output_graph_def = optimize_for_inference_lib.optimize_for_inference(input_graph_def,
                                                                         input_node_names,
                                                                         [output_node_name],
                                                                         tf.float32.as_datatype_enum)

    with tf.gfile.FastGFile('out/opt_' + MODEL_NAME + '.pb', "wb") as f:
        f.write(output_graph_def.SerializeToString())

    print("graph saved!")


def main():
    if not path.exists('out'):
        os.mkdir('out')

    input_node_name = 'input'
    output_node_name = 'output'

    data_manager = DataManager()
    num_features = data_manager.get_voc_size()

    x, y_ = model_input(input_node_name, num_features)

    train_step, loss, accuracy, merged_summary_op = build_model(x, y_, output_node_name, num_features)
    saver = tf.train.Saver()

    train(x, y_, train_step, loss, accuracy, merged_summary_op, saver, data_manager)
    print("training finished")
    export_model([input_node_name], output_node_name)


if __name__ == '__main__':
    main()
