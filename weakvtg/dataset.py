import os
import random

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co

from weakvtg import iox, bbox, anchors
from weakvtg.bbox import get_boxes_class
from weakvtg.padder import get_phrases_tensor, get_padded_examples, get_number_examples, get_max_length_examples, \
    get_max_length_phrases, get_indexed_phrases_per_example
from weakvtg.utils import pivot


class VtgDataset(Dataset):
    def __init__(self, image_filepath: str, data_filepath: str, idx_filepath: str, process_fn=None):
        super().__init__()

        if process_fn is None:
            process_fn = process_example

        self.image_filepath = image_filepath
        self.data_filepath = data_filepath
        self.idx_filepath = idx_filepath
        self.process_fn = process_fn
        
        self.idx = self._load_index()
        self.examples = self._load_examples()
        self.n = len(self.examples)

    def __getitem__(self, index) -> T_co:
        index_negative = self._sample_negative()

        def _load_example(index):
            image = self._load_image(index)
            caption = self._load_caption(index)

            example = {**caption, **image}
            example = self.process_fn(example)

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

    sentence = batch["sentence"]  # [b, n_words_sentence]
    phrases = batch["phrases"]  # [b, n_ph+, n_words+]
    phrases_negative = batch["phrases_negative"]  # [b, n_ph-, n_words-]
    phrases_2_crd = batch["phrases_2_crd"]  # [b, n_ph, 4]
    phrases_2_crd_index = batch["phrases_2_crd_index"]  # [b, n_ph, 1]

    def _get_padded_phrases_2_crd(phrases_2_crd):
        dim = (get_number_examples(phrases_2_crd),
               get_max_length_examples(phrases_2_crd),
               get_max_length_phrases(phrases_2_crd))

        # please note that `padding_value=0` produces an invalid mask, however, this mask is not used
        return get_padded_examples(phrases_2_crd, padding_value=0, dtype=float, padding_dim=dim)

    def _get_padded_phrases_2_crd_index(phrases_2_crd_index):
        dim = (get_number_examples(phrases_2_crd_index),
               get_max_length_examples(phrases_2_crd_index),
               get_max_length_phrases(phrases_2_crd_index))

        # please note that `padding_value=0` is meaningless, however, padding on index do not matter, the user
        # is responsible to mask results obtained with this index
        return get_padded_examples(phrases_2_crd_index, padding_value=0, padding_dim=dim)

    def _get_padded_sentence(sentence):
        indexed_sentences = get_indexed_phrases_per_example([sentence], tokenizer=tokenizer, vocab=vocab)

        dim = (get_number_examples(indexed_sentences),
               get_max_length_examples(indexed_sentences),
               get_max_length_phrases(indexed_sentences))

        x, mask = get_padded_examples(indexed_sentences, padding_value=0, padding_dim=dim)

        # remove the dummy dimension added in order to compute the indexed sentence
        x = x.squeeze(0)
        mask = mask.squeeze(0)

        return x, mask

    sentence, sentence_mask = _get_padded_sentence(sentence)
    phrases, phrases_mask = get_phrases_tensor(phrases, tokenizer=tokenizer, vocab=vocab)
    phrases_negative, phrases_mask_negative = get_phrases_tensor(phrases_negative, tokenizer=tokenizer, vocab=vocab)
    phrases_2_crd, _ = _get_padded_phrases_2_crd(phrases_2_crd)
    phrases_2_crd_index, _ = _get_padded_phrases_2_crd_index(phrases_2_crd_index)

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
        "class_count": torch.tensor(batch["class_count"], dtype=torch.int),
        "sentence": sentence,
        "sentence_mask": sentence_mask,
        "phrases": phrases,
        "phrases_mask": phrases_mask,
        "phrases_negative": phrases_negative,
        "phrases_mask_negative": phrases_mask_negative,
        "phrases_2_crd": phrases_2_crd,
        "phrases_2_crd_index": phrases_2_crd_index,
    }


def read_index(filename: str):
    with open(filename, "r") as f:
        return [line.strip("\n") for line in f.readlines()]


def process_example(example, *, n_boxes_to_keep: int = 100, n_active_box: int = 3, nlp=None, get_noun_phrases=None):
    def pad_boxes():
        """
        Pad boxes and update `example` as a side effect.
        """
        n_boxes = example["pred_n_boxes"]
        n_class = len(example["pred_cls_prob"][0])
        n_attr = len(example["pred_attr_prob"][0])
        n_features = len(example["pred_boxes_features"][0])

        n_keep = n_boxes_to_keep  # boxes to keep, i.e., number of bounding box in output
        n_gen = max(n_keep - n_boxes, 0)  # boxes to gen, i.e., number of bounding box missing to get `n_keep`
        n_actual = n_keep - n_gen  # real boxes, i.e., non-padding boxes

        def zeros(d1, d2): return [[0] * d2 for _ in range(d1)]
        def pad(xs, n_keep, n_gen, n_feats): return (xs + zeros(n_gen, n_feats))[:n_keep]

        example["pred_n_boxes"] = n_keep
        example["pred_boxes"] = pad(example["pred_boxes"], n_keep, n_gen, n_feats=4)
        example["pred_boxes_features"] = pad(example["pred_boxes_features"], n_keep, n_gen, n_feats=n_features)
        example["pred_attr_prob"] = pad(example["pred_attr_prob"], n_keep, n_gen, n_feats=n_attr)
        example["pred_cls_prob"] = pad(example["pred_cls_prob"], n_keep, n_gen, n_feats=n_class)

        example['pred_boxes_mask'] = [True] * n_actual + [False] * n_gen

        example["pred_active_box_index"] = [random.randrange(0, n_actual) for _ in range(n_active_box)]

    def remove_background():
        """
        Remove bounding boxes tagged with class `__background__` and update `example` as a side effect. [^1]

        The class `__background__` is included within the classes score and its purpose is to specify whether a
        bounding box is "interesting" or not. If the bounding box is not interesting than its class is set to
        `__background__` otherwise it is set to one of the other 1600 classes.

        Please note that `__background__` is an artificial class and it has index 0.

        [^1]: in fact, we do not completely remove the boxes, instead we mask them.
        """
        boxes_mask = torch.tensor(example["pred_boxes_mask"])
        boxes_class = get_boxes_class(torch.tensor(example["pred_cls_prob"]))

        example["pred_boxes_mask"] = get_boxes_mask_no_background(boxes_mask, boxes_class).detach().tolist()

    def gt_box_index():
        # Please note that this process should be moved in the make dataset workflow, i.e., the input file should
        # contain information on the index of the ground truth bounding box. However, this information is not
        # available now, so we recompute it by matching the ground truth with predicted boxes selecting the one
        # with maximum IoU.

        pred_boxes = torch.tensor(example["pred_boxes"])
        phrases_2_crd = torch.tensor(example["phrases_2_crd"])
        boxes_mask = torch.tensor(example["pred_boxes_mask"])

        n_box = pred_boxes.size()[-2]
        n_ph = phrases_2_crd.size()[-2]

        pred_boxes = pred_boxes.unsqueeze(-3).repeat(n_ph, 1, 1)
        phrases_2_crd = phrases_2_crd.unsqueeze(-2).repeat(1, n_box, 1)
        boxes_mask = boxes_mask.unsqueeze(-2).repeat(n_ph, 1)

        iou = anchors.bbox_final_iou(pred_boxes, phrases_2_crd)

        iou = torch.masked_fill(iou, boxes_mask == 0, value=0)  # we avoid picking the index of a masked bounding box

        best_iou_index = torch.argmax(iou, dim=-1).unsqueeze(-1)

        example["phrases_2_crd_index"] = best_iou_index.detach().numpy().tolist()

    def noun_phrases():
        def extract(phrase): return get_noun_phrases(nlp(phrase))
        def clamp(phrase, noun_phrases): return noun_phrases if len(noun_phrases) > 0 else [phrase]
        def join(noun_phrases): return " ".join(noun_phrases)

        phrases = example["phrases"]

        noun_phrases = list(map(extract, phrases))
        noun_phrases = list(map(lambda x: clamp(x[0], x[1]), zip(phrases, noun_phrases)))
        noun_phrases = list(map(join, noun_phrases))
        noun_phrases = list(noun_phrases)

        example["phrases"] = noun_phrases

    def class_count():
        box_class_prob = torch.tensor(example["pred_cls_prob"])
        box_class = get_boxes_class(box_class_prob)

        n_class = box_class_prob.size()[-1]

        box_class_count = get_class_count(box_class, n_class=n_class)

        example["class_count"] = box_class_count.detach().numpy().tolist()

    example["id"] = int(example["id"])
    example["phrases_2_crd"] = bbox.scale_bbox(example["phrases_2_crd"], example["image_w"], example["image_h"])
    example["pred_boxes"] = bbox.scale_bbox(example["pred_boxes"], example["image_w"], example["image_h"])

    pad_boxes()

    # requires `pad_boxes` because it leverages on same number of bounding box
    remove_background()

    # requires `pad_boxes` to prevent ground truth to be out of bounds,
    # requires `remove_background` to chose a valid bb
    gt_box_index()

    # requires `pad_boxes` because it leverages on same number of bounding box
    # requires `remove_background` to compute unbiased frequencies
    class_count()

    if nlp is not None and get_noun_phrases is not None:
        noun_phrases()

    return example


def get_boxes_mask_no_background(boxes_mask, boxes_class, background_class_index=0):
    is_not_background = boxes_class != background_class_index
    return torch.logical_and(boxes_mask, is_not_background)


def get_class_count(box_class, n_class=-1):
    """
    Return a tensor where each element of the last dimension represent a class and its value is the number of
    bounding boxes with given class.

    :param box_class: A [*, d1] tensor
    :param n_class: Total number of classes
    :return: A [*, n_class] tensor
    """
    import torch.nn.functional as F

    class_one_hot = F.one_hot(box_class, num_classes=n_class)
    class_one_hot = torch.transpose(class_one_hot, dim0=-1, dim1=-2)
    return class_one_hot.sum(dim=-1)
