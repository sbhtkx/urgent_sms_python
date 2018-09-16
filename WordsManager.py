from collections import Counter


class WordsManager:

    # initialize the object
    # creates a vocabulary of the voc_size most common words in strings by using the tokenize function
    def __init__(self, tokenize, strings, voc_size):
        self.tokenize = tokenize
        self.voc_size = voc_size
        self.voc = []

        words = []
        for s in strings:
            words += self.tokenize(s)
        full_voc = Counter(words)
        self.voc_size = min(self.voc_size, len(full_voc))
        for pair in full_voc.most_common(self.voc_size):
            self.voc.append(pair[0])

        print(type(self.voc))
        print(self.voc)

    def string_to_vector(self, s):
        words = set(self.tokenize(s))
        features = [0 for i in range(self.voc_size)]
        for i in range(self.voc_size):
            if self.voc[i] in words:
                features[i] = 1 # features[i] + 1
            else:
                features[i] = 0
        return features

# from nltk.tokenize import word_tokenize
#
# wm = WordsManager(tokenize=word_tokenize, strings=['hello world', 'goodbye world'], voc_size=100)
# str = 'world'
# print(wm.string_to_vector(str))


# msg = 'explorer hello asap really'
# print(wm.string_to_vector())
