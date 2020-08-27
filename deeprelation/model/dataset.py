import torch.utils.data as data

from transformers import InputFeatures


class SemiEvalDataset(data.Dataset):
    """ The dataset holding SemiEval data as input to models """

    def __init__(self, input_ids, attention_masks, labels):
        assert len(input_ids) == len(attention_masks) == len(labels)
        self.features = []
        for index in range(len(labels)):
            feature = InputFeatures(input_ids=input_ids[index],
                                    attention_mask=attention_masks[index],
                                    label=labels[index])
            self.features.append(feature)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index) -> InputFeatures:
        return self.features[index]
