""" This class is a util class to load SemiEval-2010-task8 data """


import re

from semieval_data import SemiEvalData


class SemiEvalDataIO(object):

    @staticmethod
    def _load_data(path):
        """
        TODO: to be implemented
        :param path: is the path to the raw data text file
        :return: the loaded data in SemiEvalData format
        """
        e1_pattern = re.compile("<e1>(.*)</e1>")
        e2_pattern = re.compile("<e2>(.*)</e2>")

        data = []

        with open(path, 'r') as f:
            while True:
                row = f.readline()
                if not row:
                    break
                relation = f.readline()
                _ = f.readline()
                _ = f.readline()

                record_id, raw_sentence = row.split('\t', 1)
                raw_sentence = raw_sentence.rstrip('\n').strip('"')

                e1 = e1_pattern.findall(raw_sentence)[0]
                e2 = e2_pattern.findall(raw_sentence)[0]

                sentence = e2_pattern.sub(e2, e1_pattern.sub(e1, raw_sentence))
                bag_of_words = sentence.split(' ')

                data.append(SemiEvalData(int(record_id), raw_sentence,
                                         bag_of_words, e1, e2, relation))
        return data

    @staticmethod
    def load_train_data():
        """
        Load all training data
        :return: the loaded training data in SemiEvalData format
        """
        return SemiEvalDataIO._load_data(
            "src/main/resources/semieval-2018-task8/train/TRAIN_FILE.TXT")

    @staticmethod
    def load_test_data():
        """
        Load all testing data
        :return: the loaded testing data in SemiEvalData format
        """
        return SemiEvalDataIO._load_data(
            "src/main/resources/semieval-2018-task8/test/TEST_FILE.TXT")

