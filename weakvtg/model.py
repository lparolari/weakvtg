import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from weakvtg import anchors
from weakvtg.mask import get_synthetic_mask
from weakvtg.utils import identity


class Model(nn.Module):
    def forward(self, batch):
        raise NotImplementedError


class MockModel(Model):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, batch):
        boxes = batch["pred_boxes"]
        phrases = batch["phrases"]
        phrases_mask = batch["phrases_mask"]
        phrases_synthetic = get_synthetic_mask(phrases_mask)

        size = (*phrases_synthetic.size()[:-1], boxes.size()[-2])

        return (torch.rand(size, requires_grad=True), torch.rand(size, requires_grad=True)),


class WeakVtgModel(Model):
    def __init__(self, phrases_embedding_net=identity, phrases_recurrent_net=identity):
        super().__init__()

        self.phrases_embedding_net = phrases_embedding_net
        self.phrases_recurrent_net = phrases_recurrent_net

    def forward(self, batch):
        boxes = batch["pred_boxes"]  # [b, n_boxes, 4]
        boxes_features = batch["pred_boxes_features"]  # [b, n_boxes, 2048]
        phrases = batch["phrases"]  # [b, n_ph, n_words]
        phrases_mask = batch["phrases_mask"]  # [b, n_ph, n_words]

        img_x = get_image_features(boxes, boxes_features)
        phrases_x = get_phrases_features(phrases, phrases_mask, self.phrases_embedding_net, self.phrases_recurrent_net)
        # TODO


def get_image_features(boxes, boxes_feat):
    """
    Normalize bounding box features and concatenate its spacial features (position and area).

    :param boxes: A [*1, 4] tensor
    :param boxes_feat: A [*2, fi] tensor
    :return: A [*3, fi + 5] tensor
    """
    boxes_feat = F.normalize(boxes_feat, p=1, dim=-1)

    boxes_tlhw = anchors.tlbr2tlhw(boxes)  # [*1, 4]
    area = (boxes_tlhw[..., 2] * boxes_tlhw[..., 3]).unsqueeze(-1)  # [*1, 1]

    return torch.cat([boxes_feat, boxes, area], dim=-1)


def get_phrases_features(phrases, phrases_mask, embedding_network, recurrent_network):
    """
    Embed phrases and apply LSTM network on embeddings.

    :param phrases: A [*, d1, d2] tensor
    :param phrases_mask: A [*, d1, d2] tensor
    :param embedding_network: A function of phrases tensor
    :param recurrent_network: A function of embedding tensor, phrases length and synthetic mask
    :return: A [*, d1, emb_p] tensor
    """
    mask = get_synthetic_mask(phrases_mask)

    phrases_length = torch.sum(phrases_mask.int(), dim=-1)  # [*, d1]

    phrases_embedding = embedding_network(phrases)  # [*, d1, d2, fp]

    phrases_x = recurrent_network(phrases_embedding, phrases_length, mask)  # [*, d1, emb_p]
    phrases_x = torch.masked_fill(phrases_x, mask == 0, value=0)

    return phrases_x


def create_phrases_embedding_network(vocab, embedding_size, freeze=False):
    vocab_size = len(vocab)
    out_of_vocabulary = 0

    embedding_matrix_values = torch.zeros((vocab_size + 1, embedding_size), requires_grad=(not freeze))

    import torchtext
    glove_embeddings = torchtext.vocab.GloVe("840B", dim=300)

    glove_words = glove_embeddings.stoi.keys()

    for idx in range(vocab_size):
        word = vocab.get_itos()[idx]
        if word in glove_words:
            glove_idx = glove_embeddings.stoi[word]
            embedding_matrix_values[idx, :] = glove_embeddings.vectors[glove_idx]
        else:
            out_of_vocabulary += 1
            # nn.init.uniform_(embedding_matrix_values[idx, :], -1, 1)
            nn.init.normal_(embedding_matrix_values[idx, :])

    if out_of_vocabulary != 0:
        logging.warning(f"Found {out_of_vocabulary} words out of vocabulary.")

    embedding_matrix = nn.Embedding(vocab_size, embedding_size)
    embedding_matrix.weight = torch.nn.Parameter(embedding_matrix_values)
    embedding_matrix.weight.requires_grad = not freeze

    return embedding_matrix
