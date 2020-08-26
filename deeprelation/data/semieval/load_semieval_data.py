""" This class is a util class to load SemiEval-2010-task8 data """


import re

from deeprelation.data.semieval.semieval_data import SemiEvalData
from pathlib import Path


class SemiEvalDataIO(object):

    @staticmethod
    def _process_raw_sentences(raw_sentence: str):
        """
        Process the raw sentences, including:
        (1) extracting the two entities from the raw sentence
        (2) generating bag of words by removing the entity tagging
        :param raw_sentence: is the sentence containing two entities
        :return: the entity1, entity2, bag_of_words
        """
        e1_pattern = re.compile("<e1>(.*)</e1>")
        e2_pattern = re.compile("<e2>(.*)</e2>")
        e1 = e1_pattern.findall(raw_sentence)[0]
        e2 = e2_pattern.findall(raw_sentence)[0]
        sentence = e2_pattern.sub(e2, e1_pattern.sub(e1, raw_sentence))
        bag_of_words = sentence.split(' ')
        return e1, e2, bag_of_words

    @staticmethod
    def load_train_data():
        """
        Load all training data
        :return: the loaded data in SemiEvalData format
        """
        relative_path = "resources/semieval-2018-task8/train/TRAIN_FILE.TXT"
        abs_path = Path(__file__).parent.parent.parent.parent / relative_path

        train_data = []

        with open(abs_path, 'r') as f:
            while True:
                content_row = f.readline()  # content row of id and sentence
                if not content_row:
                    break
                relation_row = f.readline()  # relation row
                _ = f.readline()  # comment row
                _ = f.readline()  # empty line breaking row

                relation = relation_row.rstrip()
                record_id, raw_sentence = content_row.split('\t', 1)

                raw_sentence = raw_sentence.rstrip('\n').strip('"')
                e1, e2, bag_of_words = SemiEvalDataIO._process_raw_sentences(
                    raw_sentence)

                train_data.append(SemiEvalData(int(record_id), raw_sentence,
                                               bag_of_words, e1, e2, relation))
        return train_data

    @staticmethod
    def load_test_data():
        """
        TODO: to be implemented
        Load all testing data
        :return: the loaded testing data in SemiEvalData format
        """
        return None


if __name__ == "__main__":
    train_examples = SemiEvalDataIO.load_train_data()
    train_labels = set(map(lambda x: x.get_relation(), train_examples))
    print("total training examples: " + str(len(train_examples)))
    print("total training labels: " + str(len(train_labels)))
    print("example raw sentence: " + train_examples[0].get_raw_sentence())
    print("example label: " + train_examples[0].get_relation())
    print("example e1: " + train_examples[0].get_entity1())
