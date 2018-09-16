from nltk.tokenize import word_tokenize
from FileManager import FileManager
from WordsManager import WordsManager

PATH_TRAIN_DATA = 'data/long.txt'
PATH_TEST_DATA = 'data/test.txt'
PATH_RECORDS = 'results/records.txt'
DIRECTORY_CHECKPOINTS = 'restore_files'
NUM_URGENT_MSGS = 67
NUM_CALM_MSGS = 1043
VOC_MAX_SIZE = 9999


class DataManager:

    def __init__(self,
                 label_urgent=1,
                 label_calm=0,
                 num_urgent_msgs=NUM_URGENT_MSGS,
                 num_calm_msgs=NUM_CALM_MSGS,
                 path_train_data=PATH_TRAIN_DATA,
                 path_test_data=PATH_TEST_DATA,
                 voc_max_size=VOC_MAX_SIZE):

        self.label_urgent, self.label_calm = label_urgent, label_calm
        self.num_urgent_msgs = num_urgent_msgs
        self.num_calm_msgs = num_calm_msgs
        self.path_train_data = path_train_data
        self.voc_max_size = voc_max_size
        self.path_test_data = path_test_data

        self.msgs_train = FileManager.get_messages_from_txt(self.path_train_data)
        self.msgs_test = FileManager.get_messages_from_txt(self.path_test_data)
        self.words_manager = WordsManager(tokenize=word_tokenize, strings=self.msgs_train, voc_size=self.voc_max_size)

        print('voc size: ', self.get_voc_size())

    def get_training_data(self):

        x_data = [self.words_manager.string_to_vector(s) for s in self.msgs_train]
        y_data = [[self.label_urgent] for t2 in range(len(self.msgs_train))]
        for i in range(self.num_calm_msgs):
            y_data[i][0] = self.label_calm
        return [x_data, y_data]

    def get_test_data(self):
        x_data = [self.words_manager.string_to_vector(s) for s in self.msgs_test]
        y_data = [[self.label_urgent] for t2 in range(len(self.msgs_test))]
        for i in range(11):
            y_data[i][0] = self.label_calm
        return [x_data, y_data]

    def get_voc_size(self):
        return self.words_manager.voc_size

