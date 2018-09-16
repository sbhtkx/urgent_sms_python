
class DetailedResult:

    def __init__(self):
        self.true_positive = 0
        self.false_positive = 0
        self.true_negative = 0
        self.false_negative = 0

    def add_true_positive(self):
        self.true_positive += 1

    def add_false_positive(self):
        self.false_positive += 1

    def add_true_negative(self):
        self.true_negative += 1

    def add_false_negative(self):
        self.false_negative += 1

    def get_precision(self):
        selected = self.true_positive + self.false_positive
        # if there aren't any relevant elements
        if selected == 0 and self.false_negative == 0:
            return 1
        # if there are relevant items but 0 selected elements
        if selected == 0 and self.false_negative > 0:
            return 0
        return self.true_positive / selected

    def get_recall(self):
        relevant = self.true_positive + self.false_negative
        if relevant == 0:
            return 0
        return self.true_positive / relevant

    def get_success(self):
        sum = self.true_positive + self.false_positive + self.true_negative + self.false_negative
        right = self.true_positive + self.true_negative
        return right / sum

    def update(self, label, prediction, threshold):
        if label == 1:
            if prediction >= threshold:
                self.add_true_positive()
                return True
            else:
                self.add_false_negative()
                return False
        if label == 0:
            if prediction < threshold:
                self.add_true_negative()
                return True
            else:
                self.add_false_positive()
                return False
        else:
            print('not gonna happen...')
            exit(1)

    def __str__(self):
        return ('[success: ' + str(self.get_success()) + ', precision: ' + str(self.get_precision()) + ', recall: ' +
                str(self.get_recall()) + ']')
