""" This class is a util class to load SemiEval-2010-task8 data """


class SemiEvalDataIO(object):

    @staticmethod
    def _load_data(path):
        """
        TODO: to be implemented
        :param path: is the path to the raw data text file
        :return: the loaded data in SemiEvalData format
        """
        pass

    @staticmethod
    def load_train_data():
        """
        Load all training data
        :return: the loaded training data in SemiEvalData format
        """
        return SemiEvalDataIO._load_data("src/main/resources/semieval-2018-task8/train/TRAIN_FILE.TXT")

    @staticmethod
    def load_test_data():
        """
        Load all testing data
        :return: the loaded testing data in SemiEvalData format
        """
        return SemiEvalDataIO._load_data("src/main/resources/semieval-2018-task8/test/TEST_FILE.TXT")

