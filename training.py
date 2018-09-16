import tensorflow as tf
from nltk.tokenize import word_tokenize
from FileManager import FileManager
from WordsManager import WordsManager
from DetailedResult import DetailedResult
import timeit
import pickle


# parameters
l_urgent, l_calm = 1, 0
threshold_1 = 0.8  # if the prediction is above threshold then we consider the prediction as 1 otherwise: 0
n_iterations = 6000
training_rate = 0.1     # for the optimizer
display_iter = 500
voc_max_size = 9999     # if voc_max_size is smaller than the number of distinguished words in the data-set - only the
                        # most common words will be considered
path_train_data = 'data/long.txt'
path_test_data = 'data/test.txt'
path_records = 'results/records.txt'
directory_checkpoints = 'restore_files'
n_urgent = 67   # for the loss function(number of calm and urgent messages)
n_calm = 1043   # and for the labeling
num_hidden_layer_nodes = 5

save_path = 'restore_files/save_net.ckpt'
path_voc = 'exported_data/voc.txt'


# prints each message from msgs alongside the prediction and the real label
def detailed_results(msgs, labels, words_m, sess, in_tensor, out_tensor, threshold, to_print=True):
    detailed_result = DetailedResult()
    for i in range(len(msgs)):
        pred = out_tensor.eval(session=sess, feed_dict={in_tensor: [words_m.string_to_vector(msgs[i])]})[0][0]
        if to_print:
            print('Prediction for: "' + msgs[i] + '"', pred, ', ', labels[i])
        result = detailed_result.update(label=labels[i], prediction=pred, threshold=threshold)
        if not result and to_print:
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~^ mistake ^~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    return detailed_result


def main(save_checkpoint=True, save_results=True):
    start = timeit.default_timer()

    # load data and create vocabulary
    msgs_train = FileManager.get_messages_from_txt(path_train_data)
    words_manager = WordsManager(tokenize=word_tokenize, strings=msgs_train, voc_size=voc_max_size)

    for e in words_manager.voc:
        print('e:', str(e))
        FileManager.append_to_file(path_voc, str(e))

    print(':)')

    x_data = [words_manager.string_to_vector(s) for s in msgs_train]
    y_data = [[l_urgent] for t2 in range(len(msgs_train))]
    for i in range(n_calm):
        y_data[i][0] = l_calm

    # build neural network
    n_features = words_manager.voc_size

    x = tf.placeholder(tf.float32, [None, n_features], name='x')
    W1 = tf.Variable(tf.truncated_normal([n_features, num_hidden_layer_nodes], stddev=0.1), name='W1')
    b1 = tf.Variable(tf.constant(0.1, shape=[num_hidden_layer_nodes]), name='b1')
    z1 = tf.matmul(x, W1) + b1
    activation1 = tf.nn.relu(z1)
    W2 = tf.Variable(tf.truncated_normal([num_hidden_layer_nodes, 1], stddev=0.1), name='W2')
    b2 = tf.Variable(0., name='b2')
    z2 = tf.matmul(activation1, W2) + b2
    # y = 1 / (1.0 + tf.exp(-z2))
    y = tf.divide(tf.constant(1.0), tf.add(tf.constant(1.0), tf.exp(-z2)))
    y_ = tf.placeholder(tf.float32, [None, 1], name='y_')
    loss = tf.reduce_mean(-((n_calm / (n_calm + n_urgent)) * y_ * tf.log(y) +
                            (n_urgent / (n_calm + n_urgent)) * (1 - y_) * tf.log(1 - y)))
    update = tf.train.GradientDescentOptimizer(training_rate).minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    test1 = "I need you now! Please answer ASAP"  # test
    test2 = "I wanted to hear your thoughts about my plans"  # test
    test3 = "HI ITS KATE CAN U GIVE ME A RING ASAP"  # train
    test4 = "Mum say we wan to go then go... Then she can shun bian watch da glass exhibition..."  # train
#######################################################################################################################
    tf.summary.scalar("loss", loss)
    #tf.summary.scalar("accuracy", accuracy)
    merged_summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter('logs/', graph=tf.get_default_graph())
#######################################################################################################################
    for i in range(n_iterations):
        sess.run(update, feed_dict={x: x_data, y_: y_data})
        if i % display_iter == 0:
            print(i, "loss: ", sess.run(loss, feed_dict={x: x_data, y_: y_data}))
            print('Prediction for: "' + test1 + '"',
                  y.eval(session=sess, feed_dict={x: [words_manager.string_to_vector(test1)]})[0][0])
            print('Prediction for: "' + test2 + '"',
                  y.eval(session=sess, feed_dict={x: [words_manager.string_to_vector(test2)]})[0][0])
            print()
    print('training complete!')

    # some samples:
    print(test1, ' prediction:', y.eval(session=sess, feed_dict={x: [words_manager.string_to_vector(test1)]})[0][0])
    print(test2, ' prediction:', y.eval(session=sess, feed_dict={x: [words_manager.string_to_vector(test2)]})[0][0])
    print(test3, ' prediction:', y.eval(session=sess, feed_dict={x: [words_manager.string_to_vector(test3)]})[0][0])
    print(test4, ' prediction:', y.eval(session=sess, feed_dict={x: [words_manager.string_to_vector(test4)]})[0][0])

    print("\n-----------------------------------------------------------------T R A I N--------------------------------"
          "--------------------------------------------------------\n")
    labels_train = [label[0] for label in y_data]

    details_train = detailed_results(msgs=msgs_train, labels=labels_train, words_m=words_manager, sess=sess,
                                          in_tensor=x, out_tensor=y, threshold=0.5)
    print('train results: ' + str(details_train))

    print("\n-----------------------------------------------------------------T E S T----------------------------------"
          "------------------------------------------------------\n")
    # create the test data
    msgs_test = FileManager.get_messages_from_txt(path_test_data)
    labels_test = [l_calm for k in range(len(msgs_test))]
    for k in range(11):
        labels_test[k] = l_urgent

    details_test = detailed_results(msgs=msgs_test, labels=labels_test, words_m=words_manager, sess=sess,
                                         in_tensor=x, out_tensor=y, threshold=threshold_1)
    print('test results: ' + str(details_test))

    print('train results: ' + str(details_train))

    if save_checkpoint:
        # save the network to a file
        saver = tf.train.Saver()
        saved_to = saver.save(sess, './restore_files/model_1')

        # save words_manager to a file
        with open('./restore_files/words_manager.pkl', 'wb') as output:
            pickle.dump(words_manager, output, pickle.HIGHEST_PROTOCOL)

    # write results to txt file
    if save_results:
        FileManager.append_to_file(path_records, 'vocabulary size: ' + str(words_manager.voc_size))
        FileManager.append_to_file(path_records, 'threshold: ' + str(threshold_1))
        FileManager.append_to_file(path_records, 'calm label: ' + str(l_calm) + '. urgent label: ' + str(l_urgent))
        FileManager.append_to_file(path_records, str(details_test))

        stop = timeit.default_timer()
        runtime = stop - start
        FileManager.append_to_file(path_records, 'runtime: ' + str(runtime))
        FileManager.append_to_file(path_records, '------------------------------')
        FileManager.append_to_file(path_records, 'W1:')
        for weight in sess.run(W1):
            FileManager.append_to_file(path_records, str(weight))
        FileManager.append_to_file(path_records, 'b1:')
        for weight in sess.run(b1):
            FileManager.append_to_file(path_records, str(weight))
        FileManager.append_to_file(path_records, 'W2:')
        for weight in sess.run(W2):
            FileManager.append_to_file(path_records, str(weight))
        FileManager.append_to_file(path_records, 'b2:')
        FileManager.append_to_file(path_records, str(sess.run(b2)))
        FileManager.append_to_file(path_records, '------------------------------')
        print('saved to: ', saved_to)

    # export_model(['x'], 'truediv:0')


if __name__ == "__main__":
    # main(save_checkpoint=True, save_results=True)  # (False, False)
    msg = 'explorer hello asap really'
    msgs_train = FileManager.get_messages_from_txt(path_train_data)
    wm = WordsManager(tokenize=word_tokenize, strings=msgs_train, voc_size=voc_max_size)
    v = wm.string_to_vector(msg)
    for i in range(v.__len__()):
        if v[i] == 1:
            print(i)



