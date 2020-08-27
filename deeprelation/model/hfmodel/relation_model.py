from deeprelation.model.hfmodel.relation_type_model import RelationTypeModel
from deeprelation.model.dataset import SemiEvalDataset
from deeprelation.data.semieval.semieval_data import SemiEvalData
from typing import List
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    PreTrainedModel
)


class RelationModel(RelationTypeModel):

    def _build_dataset(self, dataset: List[SemiEvalData]) -> SemiEvalDataset:
        relations = [data.get_relation() for data in dataset]
        sentences = [" ".join(data.get_bag_of_words()) for data in dataset]
        labels = [self.relation_dict[t] for t in relations]
        input_ids, attention_masks = self.tokenize(sentences)
        return SemiEvalDataset(input_ids, attention_masks, labels)

    def build_model(self) -> PreTrainedModel:
        num_labels = len(self.relation_dict)
        model_config = AutoConfig.from_pretrained(self.hf_model_name,
                                                  num_labels=num_labels)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.hf_model_name, config=model_config).to(self.device)
        return model
