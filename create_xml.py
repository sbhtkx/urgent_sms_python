import pickle
import tensorflow as tf
import numpy as np


def append_to_file(path, txt):
    with open(path, 'a+') as myfile:
        myfile.write(txt)


def delete_file_content(path):
    with open(path, 'w') as myfile:
        myfile.write('')


if __name__ == "__main__":

    with tf.Session() as sess, open('./restore_files/words_manager.pkl', 'rb') as input:
        words_manager = pickle.load(input)

        new_saver = tf.train.import_meta_graph('./restore_files/model_1.meta')
        new_saver.restore(sess, tf.train.latest_checkpoint('./restore_files'))

        graph = tf.get_default_graph()

        w1 = graph.get_tensor_by_name('W1:0')
        b1 = graph.get_tensor_by_name('b1:0')
        w2 = graph.get_tensor_by_name('W2:0')
        b2 = graph.get_tensor_by_name('b2:0')

        w1 = sess.run(w1)
        b1 = sess.run(b1)
        w2 = sess.run(w2)
        b2 = sess.run(b2)

        print(type(str(w1[0][0])))

        print(w2.shape)

        delete_file_content('exported_data/w1_array.csv')
        delete_file_content('exported_data/w2_array.csv')
        delete_file_content('exported_data/b1_array.csv')
        delete_file_content('exported_data/b2_array.csv')

        # print(len(w1[0]))

        for j in range(len(w1[0])):
            if j <= len(w1[0]):
                print(j,end='')
                print('len:',len(w1[0]-2))


        file_name = 'w1_array.csv'
        append_to_file('exported_data/'+file_name, '{')
        for i in range(len(w1)):
            append_to_file('exported_data/'+file_name, '{')
            for j in range(len(w1[i])):
                append_to_file('exported_data/'+file_name, str(w1[i][j]))
                if j < len(w1[i])-1:
                    append_to_file('exported_data/'+file_name, ',')
            append_to_file('exported_data/'+file_name, '}')
        append_to_file('exported_data/'+file_name, '}')

        file_name = 'w2_array.csv'
        append_to_file('exported_data/' + file_name, '{')
        for i in range(len(w2)):
            append_to_file('exported_data/' + file_name, '{')
            for j in range(len(w2[i])):
                append_to_file('exported_data/' + file_name, str(w2[i][j]) + ',')
            append_to_file('exported_data/' + file_name, '}')
        append_to_file('exported_data/' + file_name, '}')

        file_name = 'b1_array.csv'
        append_to_file('exported_data/' + file_name, '{')
        for i in range(len(b1)):
            append_to_file('exported_data/' + file_name, str(w1[i][j]) + ',')
        append_to_file('exported_data/' + file_name, '}')



        append_to_file('exported_data/b2_array.csv', str(b2) + ',')
        append_to_file('exported_data/b2_array.csv', '\n')





