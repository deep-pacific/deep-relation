from deeprelation.model.hfmodel.base_model import HFBaseModel
from abc import abstractmethod
from typing import List
from deeprelation.data.semieval.semieval_data import SemiEvalData
from deeprelation.model.dataset import SemiEvalDataset
from transformers import (
    AutoTokenizer,
    TrainingArguments
)


class BaseTokenzierModel(HFBaseModel):

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
        HFBaseModel.__init__(self,
                             hf_model_name,
                             train_data,
                             test_data,
                             train_args)

        self.tokenizer = AutoTokenizer.from_pretrained(self.hf_model_name)

    def tokenize(self, sentence):
        batch = self.tokenizer(sentence, padding=True, truncation=True,
                               return_tensors="pt")
        input_ids = batch["input_ids"].to(self.device)
        attention_masks = batch["attention_mask"].to(self.device)
        return input_ids, attention_masks

    @abstractmethod
    def build_dataset(self, dataset_name: str) -> SemiEvalDataset:
        pass

    @abstractmethod
    def build_model(self):
        pass
