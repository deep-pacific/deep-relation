from deeprelation.model.hfmodel.base_tokenizer_model import BaseTokenzierModel
from deeprelation.model.dataset import SemiEvalDataset
from deeprelation.data.semieval.semieval_data import SemiEvalData
from typing import List
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    PreTrainedModel,
    TrainingArguments
)


class RelationTypeModel(BaseTokenzierModel):

    def __init__(self,
                 hf_model_name: str,
                 train_data: List[SemiEvalData],
                 test_data: List[SemiEvalData],
                 train_args: TrainingArguments):
        """
        :param hf_model_name:
        :param train_data:
        :param test_data:
        :param train_args:
        """
        BaseTokenzierModel.__init__(self,
                                    hf_model_name,
                                    train_data,
                                    test_data,
                                    train_args)

    def _build_dataset(self, dataset: List[SemiEvalData]) -> SemiEvalDataset:
        relation_types = [data.get_relation_type() for data in dataset]
        sentences = [" ".join(data.get_bag_of_words()) for data in dataset]
        labels = [self.relation_type_dict[t] for t in relation_types]
        input_ids, attention_masks = self.tokenize(sentences)
        return SemiEvalDataset(input_ids, attention_masks, labels)

    def build_dataset(self, dataset_name: str) -> SemiEvalDataset:
        if dataset_name == "training":
            return self._build_dataset(self.train_data)
        elif dataset_name == "testing":
            return self._build_dataset(self.test_data)
        else:
            raise RuntimeError("Dataset name invalid: " + dataset_name)

    def build_model(self) -> PreTrainedModel:
        num_labels = len(self.relation_type_dict)
        model_config = AutoConfig.from_pretrained(self.hf_model_name,
                                                  num_labels=num_labels)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.hf_model_name, config=model_config).to(self.device)
        return model
