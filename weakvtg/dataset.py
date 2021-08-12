import os
import random

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co

from weakvtg import iox, bbox
from weakvtg.padder import get_phrases_tensor, get_padded_examples, get_number_examples, get_max_length_examples, \
    get_max_length_phrases
from weakvtg.utils import pivot


class VtgDataset(Dataset):
    def __init__(self, data_filepath: str, idx_filepath: str):
        super().__init__()

        self.data_filepath = data_filepath
        self.idx_filepath = idx_filepath
        
        self.idx = self._load_index()
        self.examples = self._load_examples()
        self.n = len(self.examples)

    def __getitem__(self, index) -> T_co:
        index_negative = self._sample_negative()

        def _load_example(index):
            image = self._load_image(index)
            caption = self._load_caption(index)

            example = {**caption, **image}
            example = process_example(example)

            return example

        example = _load_example(index)
        example_negative = _load_example(index_negative)

        # we retrieve only required data from negative example
        example["id_negative"] = example_negative["id"]
        example["phrases_negative"] = example_negative["phrases"]

        return example

    def __len__(self):
        return self.n

    def _load_index(self):
        return read_index(self.idx_filepath)
    
    def _load_examples(self):
        examples = sorted(os.listdir(self.data_filepath))
        examples = filter(lambda x: "img" not in x, examples)
        examples = filter(lambda x: x.split("_")[0] in self.idx, examples)
        return list(examples)

    def _load_caption(self, index):
        caption_filename = "{}".format(self.examples[index])
        caption = iox.load_pickle(os.path.join(self.data_filepath, caption_filename))
        return caption

    def _load_image(self, index):
        image_filename = "{}_img.pickle".format(self.examples[index].split("_")[0])
        image = iox.load_pickle(os.path.join(self.data_filepath, image_filename))
        return image

    def _sample_negative(self):
        [index] = random.sample(range(self.n), 1)
        return index


def collate_fn(batch, tokenizer, vocab):
    batch = pivot(batch)

    phrases = batch["phrases"]  # [b, n_ph+, n_words+]
    phrases_negative = batch["phrases"]  # [b, n_ph-, n_words-]
    phrases_2_crd = batch["phrases_2_crd"]  # [b, n_ph, 4]

    phrases, phrases_mask = get_phrases_tensor(phrases, tokenizer=tokenizer, vocab=vocab)
    phrases_negative, phrases_mask_negative = get_phrases_tensor(phrases_negative, tokenizer=tokenizer, vocab=vocab)
    phrases_2_crd, phrases_2_crd_mask = get_padded_examples(phrases_2_crd,
                                                            padding_value=0,
                                                            padding_dim=(
                                                                get_number_examples(phrases_2_crd),
                                                                get_max_length_examples(phrases_2_crd),
                                                                get_max_length_phrases(phrases_2_crd)))

    return {
        "id": torch.tensor(batch["id"], dtype=torch.long),
        "id_negative": torch.tensor(batch["id_negative"], dtype=torch.long),
        "image_w": torch.tensor(batch["image_w"], dtype=torch.long),
        "image_h": torch.tensor(batch["image_h"], dtype=torch.long),
        "pred_n_boxes": torch.tensor(batch["pred_n_boxes"], dtype=torch.int),
        "pred_boxes": torch.tensor(batch["pred_boxes"], dtype=torch.float32),
        "pred_cls_prob": torch.tensor(batch["pred_cls_prob"], dtype=torch.float32),
        "pred_attr_prob": torch.tensor(batch["pred_attr_prob"], dtype=torch.float32),
        "pred_boxes_features": torch.tensor(batch["pred_boxes_features"], dtype=torch.float32),
        "pred_active_box_index": torch.tensor(batch["pred_active_box_index"], dtype=torch.long),
        "pred_boxes_mask": torch.tensor(batch["pred_boxes_mask"], dtype=torch.bool),
        "phrases": phrases,
        "phrases_mask": phrases_mask,
        "phrases_negative": phrases_negative,
        "phrases_mask_negative": phrases_mask_negative,
        "phrases_2_crd": phrases_2_crd,
        "phrases_2_crd_mask": phrases_2_crd_mask,
    }


def read_index(filename: str):
    with open(filename, "r") as f:
        return [line.strip("\n") for line in f.readlines()]


def process_example(example, n_boxes_to_keep: int = 100, n_active_box: int = 3):
    example["id"] = int(example["id"])
    example["phrases_2_crd"] = bbox.scale_bbox(example["phrases_2_crd"], example["image_w"], example["image_h"])
    example["pred_boxes"] = bbox.scale_bbox(example["pred_boxes"], example["image_w"], example["image_h"])

    pred_n_boxes = example["pred_n_boxes"]
    n_boxes_class = len(example["pred_attr_prob"][0])
    n_boxes_attr = len(example["pred_cls_prob"][0])
    n_boxes_features = len(example["pred_boxes_features"][0])
    n_boxes_to_gen = n_boxes_to_keep - pred_n_boxes
    example["pred_n_boxes"] = n_boxes_to_keep
    example["pred_boxes"] = example["pred_boxes"] + [[0] * 4 for i in range(n_boxes_to_gen)]
    example["pred_boxes_features"] = example["pred_boxes_features"] + [[0] * n_boxes_features for i in
                                                                       range(n_boxes_to_gen)]
    example["pred_attr_prob"] = example["pred_attr_prob"] + [[0] * n_boxes_class for i in range(n_boxes_to_gen)]
    example["pred_cls_prob"] = example["pred_cls_prob"] + [[0] * n_boxes_attr for i in range(n_boxes_to_gen)]

    example['pred_boxes_mask'] = [True] * (n_boxes_to_keep - n_boxes_to_gen) + [False] * n_boxes_to_gen

    example["pred_active_box_index"] = [random.randrange(0, pred_n_boxes) for _ in range(n_active_box)]

    return example
