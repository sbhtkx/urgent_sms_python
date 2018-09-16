import tensorflow as tf
from WordsManager import WordsManager
import pickle
import inspect
import numpy as np


def is_urgent(msg, threshold=0.8):
    return urgency(msg) > threshold


def urgency(msg):
    with tf.Session() as sess, open('./restore_files/words_manager.pkl', 'rb') as input:
        words_manager = pickle.load(input)

        new_saver = tf.train.import_meta_graph('./restore_files/model_1.meta')
        new_saver.restore(sess, tf.train.latest_checkpoint('./restore_files'))

        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name("x:0")
        # y_ = graph.get_tensor_by_name("y_:0")
        y = graph.get_tensor_by_name('truediv:0')
        print('eval:', y.eval(session=sess, feed_dict={x: [words_manager.string_to_vector(msg)]})[0][0])
        print(msg)
        print(words_manager.string_to_vector(msg))
        return y.eval(session=sess, feed_dict={x: [words_manager.string_to_vector(msg)]})[0][0]


if __name__ == "__main__":

    msg = 'explorer hello asap really'
    print(msg,':', urgency(msg))

    # print('"asap the police is here!!!" :', is_urgent('asap the police is here!!!'))
    # print('"what is to eat today?" :', is_urgent('what is to eat today?'))
    # print('"is the asap" :', is_urgent('is the asap'))
    #
    # with tf.Session() as sess, open('./restore_files/words_manager.pkl', 'rb') as input:
    #     words_manager = pickle.load(input)
    #
    #     new_saver = tf.train.import_meta_graph('./restore_files/model_1.meta')
    #     new_saver.restore(sess, tf.train.latest_checkpoint('./restore_files'))
    #
    #     graph = tf.get_default_graph()
    #     # x = graph.get_tensor_by_name("x:0")
    #     # y_ = graph.get_tensor_by_name("y_:0")
    #     y = graph.get_tensor_by_name('truediv:0')
    #
    #     for d in tf.trainable_variables():
    #         print(d)
    #     print('-----------------------------------------------------------------')
    #
    #     x = words_manager.string_to_vector("asap the police is here!!!")
    #
    #     w1 = graph.get_tensor_by_name('W1:0')
    #     b1 = graph.get_tensor_by_name('b1:0')
    #     w2 = graph.get_tensor_by_name('W2:0')
    #     b2 = graph.get_tensor_by_name('b2:0')
    #
    #     w1 = sess.run(w1)
    #     b1 = sess.run(b1)
    #     w2 = sess.run(w2)
    #     b2 = sess.run(b2)
    #
    #     # print(w1)
    #     # print(type(w1))
    #
    #     z1 = np.matmul(x, w1) + b1
    #     activation1 = np.maximum(z1, 0)
    #
    #     z2 = np.matmul(activation1, w2) + b2
    #     # y = 1 / (1.0 + np.exp(-z2))
    #     y = np.exp(-z2)
    #     print(y)
    #     print(1.0+y)
    #     print(1/(1.0+y))
    #     ans = 1/(1.0+y)
    #     is_urgent("asap the police is here!!!")
    #     # print('x', len(x))
    #     # print('w1', w1.shape)
    #     # print('b1', b1.shape)
    #     # print('w2', w2.shape)
    #     # print('b2', b2.shape)
    #     # print('z1', z1.shape)
    #     # print('z2', z2.shape)
    #     # print('y', y.shape)
    #     # print('ans', ans.shape)




