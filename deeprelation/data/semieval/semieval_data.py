""" This is the data model class for SemiEval-2010-task8 data """


class SemiEvalData(object):

    def __init__(self, record_id, raw_sentence, bag_of_words, entity1, entity2,
                 relation):
        """
        class initiation function
        Args:
            record_id: the id of a record
            raw_sentence: the raw sentence of the input
            bag_of_words: a list of words
            entity1: the first entity
            entity2: the second entity
            relation: the relation of the two entities
        """
        self.record_id = record_id
        self.raw_sentence = raw_sentence
        self.bag_of_words = bag_of_words
        self.entity1 = entity1
        self.entity2 = entity2
        self.relation = relation
        self.relation_type = relation.split("(")[0]

    def get_record_id(self):
        return self.record_id

    def get_raw_sentence(self):
        return self.raw_sentence

    def get_bag_of_words(self):
        return self.bag_of_words

    def get_entity1(self):
        return self.entity1

    def get_entity2(self):
        return self.entity2

    def get_relation(self):
        return self.relation

    def get_relation_type(self):
        """
        Get the type of the relation regardless of the entity order. For
        example, Product-Producer(e1,e2) and Product-Producer(e2,e1) will all
        be treated as Product-Producer
        :return: the type of the relation
        """
        return self.relation_type
