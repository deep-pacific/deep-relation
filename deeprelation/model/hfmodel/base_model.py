import torch
import time

from sklearn.metrics import precision_recall_fscore_support
from abc import ABC, abstractmethod
from typing import List
from deeprelation.data.semieval.semieval_data import SemiEvalData
from deeprelation.model.dataset import SemiEvalDataset
from transformers import (
    PreTrainedModel,
    TrainingArguments,
    Trainer
)


class HFBaseModel(ABC):

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
        self.hf_model_name = hf_model_name
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_name)
        self.train_data = train_data
        self.test_data = test_data
        self.train_args = train_args
        relation_types = \
            set(map(lambda x: x.get_relation_type(), self.train_data))
        self.relation_type_dict = {v: k for k, v in enumerate(relation_types)}
        relations = set(map(lambda x: x.get_relation(), self.train_data))
        self.relation_dict = {v: k for k, v in enumerate(relations)}
        self.model = self.build_model()

    @staticmethod
    def compute_metrics(pred):
        """ This function defines how to evaluate hugging face Trainer """
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = \
            precision_recall_fscore_support(labels, preds, average='macro')
        return {
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

    @abstractmethod
    def build_dataset(self, dataset_name: str) -> SemiEvalDataset:
        pass

    @abstractmethod
    def build_model(self) -> PreTrainedModel:
        pass

    def build_trainer(self):
        training_dataset = self.build_dataset("training")
        testing_dataset = self.build_dataset("testing")
        trainer = Trainer(
            model=self.model,
            args=self.train_args,
            compute_metrics=HFBaseModel.compute_metrics,
            train_dataset=training_dataset,
            eval_dataset=testing_dataset
        )
        return trainer

    @staticmethod
    def train_and_evaluate(trainer: Trainer):
        time1 = time.time()
        trainer.train()
        time2 = time.time()
        eval_results = trainer.evaluate()
        return time2 - time1, eval_results
