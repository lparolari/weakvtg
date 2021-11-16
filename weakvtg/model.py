import functools
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn

from weakvtg import anchors
from weakvtg.config import get_global_device
from weakvtg.mask import get_synthetic_mask
from weakvtg.utils import expand, invert


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
    def __init__(self, phrases_embedding_net, phrases_recurrent_net, image_embedding_net, get_attributes_embedding,
                 get_classes_embedding, get_phrases_embedding, get_phrases_representation, get_concept_similarity,
                 get_attribute_similarity, f_similarity, apply_concept_similarity, apply_attribute_similarity):
        super().__init__()

        self.phrases_embedding_net = phrases_embedding_net
        self.phrases_recurrent_net = phrases_recurrent_net
        self.image_embedding_net = image_embedding_net

        self.get_classes_embedding = get_classes_embedding
        self.get_attributes_embedding = get_attributes_embedding
        self.get_phrases_embedding = get_phrases_embedding
        self.get_phrases_representation = get_phrases_representation
        self.get_concept_similarity = get_concept_similarity
        self.get_attribute_similarity = get_attribute_similarity
        self.apply_concept_similarity = apply_concept_similarity
        self.apply_attribute_similarity = apply_attribute_similarity

        self.f_similarity = f_similarity

    def forward(self, batch):
        pred_n_boxes = batch["pred_n_boxes"]                    # [b]
        boxes = batch["pred_boxes"]                             # [b, n_boxes, 4]
        box_mask = batch["pred_boxes_mask"]                   # [b, n_boxes]
        boxes_features = batch["pred_boxes_features"]           # [b, n_boxes, 2048]
        boxes_class_prob = batch["pred_cls_prob"]               # [b, n_boxes, n_class]
        box_class = get_box_class(boxes_class_prob)             # [b, n_box]
        phrase = batch["phrases"]                              # [b, n_ph+, n_words+]
        phrase_mask = batch["phrases_mask"]                    # [b, n_ph+, n_words+]
        synth_mask = get_synthetic_mask(phrase_mask)

        _get_concept_similarity = self.get_concept_similarity
        _get_attribute_similarity = self.get_attribute_similarity
        _get_classes_embedding = self.get_classes_embedding
        _get_attributes_embedding = self.get_attributes_embedding
        _get_phrases_embedding = self.get_phrases_embedding
        _get_phrases_features = functools.partial(get_phrases_features,
                                                  get_phrases_embedding=_get_phrases_embedding,
                                                  get_phrases_representation=self.get_phrases_representation)
        _get_image_representation = functools.partial(get_image_representation, embedding_net=self.image_embedding_net)
        apply_concept_similarity = self.apply_concept_similarity
        apply_attribute_similarity = self.apply_attribute_similarity

        # extract positive/negative features
        img_x = get_image_features(boxes, boxes_features)
        img_x = _get_image_representation(img_x)  # [b, n_box, feat_box]

        phrase_x = _get_phrases_features(phrase, phrase_mask)  # [b, n_ph, feat_ph]

        b = img_x.size()[-3]
        n_box = img_x.size()[-2]
        feat_box = img_x.size()[-1]
        n_ph = phrase_x.size()[-2]
        feat_ph = phrase_x.size()[-1]

        img_x = img_x.view(-1, feat_box)     # [b*n_box, feat_box]
        img_x = img_x.unsqueeze(-3)          # [1, b*n_box, feat_box]
        img_x = img_x.repeat(b, 1, 1)        # [b, b*n_box, feat_box]
        img_x = img_x.unsqueeze(-3)          # [b, 1, b*n_box, feat_box]
        img_x = img_x.repeat(1, n_ph, 1, 1)  # [b, n_ph, b*n_box, feat_box]

        phrase_x = phrase_x.unsqueeze(-2)             # [b, n_ph, 1, feat_ph]
        phrase_x = phrase_x.repeat(1, 1, b*n_box, 1)  # [b, n_ph, b*n_box, feat_ph]

        prediction = predict_logits(img_x, phrase_x, f_similarity=self.f_similarity)  # [b, n_ph, b*n_box]
        prediction = prediction.view(b, n_ph, b, n_box)      # [b, n_ph, b, n_box]
        prediction = prediction.transpose(dim0=-3, dim1=-2)  # [b, b, n_ph, n_box]

        synth_mask_ = synth_mask.view(1, b, n_ph, 1)
        box_mask_ = box_mask.view(1, b, 1, n_box)

        prediction = torch.masked_fill(prediction, synth_mask_ == 0, value=-1)
        prediction = torch.masked_fill(prediction, box_mask_ == 0, value=-1)

        # concept sim
        label_e = _get_classes_embedding(box_class)
        phrase_e = _get_phrases_embedding(phrase)
        phrase_mask_ = phrase_mask.unsqueeze(-1)
        box_mask_ = box_mask.unsqueeze(-1)

        concept_sim = _get_concept_similarity((phrase_e, phrase_mask_), (label_e, box_mask_))
        concept_sim = concept_sim.unsqueeze(-4)

        prediction = apply_concept_similarity(prediction, concept_sim)

        return prediction,


def predict_logits(img_x, phrases_x, f_similarity):
    """
    Compute given similarity measure over the last dimension (features) of `img_x` and `phrases_x`.

    :param img_x: A [*, d1, d2, d3] tensor
    :param phrases_x: A [*, d1, d2, d3] tensor
    :param f_similarity: A similarity measure between [*, d3] tensors
    :return: A `[*, d1, d2]` tensor
    """
    similarity = f_similarity(img_x, phrases_x, dim=-1)

    return similarity


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


def get_image_representation(img_x, embedding_net):
    """
    Return image representation (i.e., semantic features) for image features given embedding network.
    """
    return embedding_net(img_x)


def get_phrases_features(phrases, phrases_mask, get_phrases_embedding, get_phrases_representation):
    """
    Embed phrases and apply LSTM network on embeddings.

    :param phrases: A [*, d1, d2] tensor
    :param phrases_mask: A [*, d1, d2] tensor
    :param get_phrases_embedding: A function of phrases tensor
    :param get_phrases_representation: A function of embedding tensor, phrases length and synthetic mask
    :return: A [*, d1, emb_p] tensor
    """
    mask = get_synthetic_mask(phrases_mask)

    phrases_length = torch.sum(phrases_mask.int(), dim=-1)  # [*, d1]

    phrases_embedding = get_phrases_embedding(phrases)  # [*, d1, d2, fp]
    phrases_representation = get_phrases_representation(phrases_embedding, phrases_length, mask)  # [*, d1, emb_p]

    phrases_x = torch.masked_fill(phrases_representation, mask == 0, value=0)

    return phrases_x


def get_phrases_embedding(phrases, embedding_network):
    """
    Return phrases embedding given the embedding network.
    """
    phrases_embedding = embedding_network(phrases)  # [*, d1, d2, fp]
    return phrases_embedding


def get_phrases_representation(phrases_emb, phrases_length, mask, out_features, recurrent_network, device=None):
    """
    Return phrases representation from phrases features (i.e., embeddings), phrases length and mask given a
    recurrent network.
    """
    batch_size = phrases_emb.size()[0]
    max_n_ph = phrases_emb.size()[1]
    max_ph_len = phrases_emb.size()[2]

    out_feats_dim = out_features

    # [max_ph_len, b*max_n_ph, 300]
    phrases_emb = phrases_emb.view(-1, phrases_emb.size()[-2], phrases_emb.size()[-1])
    phrases_emb = phrases_emb.permute(1, 0, 2).contiguous()

    # note: we need to fix the bug about phrases with lengths 0. On cpu required by torch
    phrases_length_clamp = phrases_length.view(-1).clamp(min=1).cpu()
    phrases_pack_emb = rnn.pack_padded_sequence(phrases_emb, phrases_length_clamp, enforce_sorted=False)
    phrases_x_o, *_ = recurrent_network(phrases_pack_emb)
    phrases_x_o = rnn.pad_packed_sequence(phrases_x_o, batch_first=False)  # (values, length)

    # due to padding we need to get indexes in this way. On device now.
    idx = (phrases_x_o[1] - 1).unsqueeze(0).unsqueeze(-1).repeat(1, 1, phrases_x_o[0].size()[2]).to(device)

    phrases_x = torch.gather(phrases_x_o[0], 0, idx)  # [1, b*max_n_ph, 2048]
    phrases_x = phrases_x.permute(1, 0, 2).contiguous().unsqueeze(0)  # [b*max_n_ph, 2048]

    # back to batch size dimension
    phrases_x = phrases_x.squeeze(1).view(batch_size, max_n_ph, out_feats_dim)  # [b, max_n_ph, out_feats_dim]
    phrases_x = torch.masked_fill(phrases_x, mask == 0, 0)  # boolean mask required

    # normalize features
    phrases_x_norm = F.normalize(phrases_x, p=1, dim=-1)

    return phrases_x_norm


def get_box_class(probability):
    """
    Return the argmax on `probability` tensor.
    """
    return torch.argmax(probability, dim=-1)


def get_box_attribute(probability):
    """
    Return the argmax on `probability` tensor.
    """
    return torch.argmax(probability, dim=-1)


def apply_concept_similarity_one(logits, *_, **__):
    """
    Return
        logits * 1
    """
    return logits


def apply_concept_similarity_product(logits, concept_similarity, *_, **__):
    """
    Return
        logits * abs(concept_similarity)
    """
    return logits * torch.abs(concept_similarity)


def apply_concept_similarity_mean(logits, concept_similarity, *, lam=.5, **__):
    """
    Return
        lam * logits + (1 - lam) * concept_similarity
    """
    return lam * logits + (1 - lam) * concept_similarity


def get_batched_batch(x, dim=0):
    """
    Return a batched version of a batch, i.e., for each batch, repeat the batch.
    :param x: A [d1, d2, ..., dN] tensor
    :param dim: An integer dimension
    :return: A [d1, d2, ..., di-1, di, di, di+1, ..., dn] tensor where di = d[dim]
    """
    return expand(x, dim=dim, size=x.size(dim))


def create_phrases_embedding_network(vocab, pretrained_embeddings, *, embedding_size=300, freeze=False,
                                     f_spell_correction=None):
    import re
    import numpy as np

    from weakvtg.utils import identity

    device = get_global_device()

    if f_spell_correction is None:
        f_spell_correction = identity

    def get_embedding_idx(words, word):
        if f_spell_correction(word) in words:
            return pretrained_embeddings.stoi[f_spell_correction(word)]
        return -1

    vocab_size = len(vocab)
    out_of_vocabulary = 0
    out_of_vocabulary_words = []

    embedding_matrix_values = torch.zeros((vocab_size + 1, embedding_size), requires_grad=(not freeze), device=device)

    pretrained_words = pretrained_embeddings.stoi.keys()

    for word_idx in range(vocab_size):
        word = vocab.get_itos()[word_idx]

        # for out of vocabulary words, we compute a meaningful representation instead of random initialization
        # we split words and compute the mean of representations

        word_split = re.split(r"\s|,|\.", word)  # split by space, column and dot

        word_count = 0
        word_rep = np.zeros([embedding_size])

        for word_tmp in word_split:
            if word_tmp in pretrained_words:
                embedding_idx = get_embedding_idx(pretrained_words, word_tmp)

                if embedding_idx != -1:
                    word_count += 1
                    word_rep = np.add(word_rep, pretrained_embeddings.vectors[embedding_idx])

        if word_count > 0:
            word_rep = word_rep / word_count
            embedding_matrix_values[word_idx, :] = word_rep
        else:
            out_of_vocabulary += 1
            out_of_vocabulary_words += [word]
            nn.init.normal_(embedding_matrix_values[word_idx, :])

    if out_of_vocabulary != 0:
        logging.warning(f"Found {out_of_vocabulary} words out of vocabulary.")
        logging.debug(f"Out of vocab words: {out_of_vocabulary_words}")

    embedding_matrix = nn.Embedding(vocab_size, embedding_size)
    embedding_matrix.weight = torch.nn.Parameter(embedding_matrix_values)
    embedding_matrix.weight.requires_grad = not freeze

    return embedding_matrix


def create_image_embedding_network(in_features, out_features, n_hidden_layer=2):
    def create_layer(in_features, out_features):
        linear = nn.Linear(in_features, out_features)
        nn.init.xavier_normal_(linear.weight)
        nn.init.zeros_(linear.bias)
        return linear

    def create_hidden_layers():
        hidden_layers = ()
        for i in range(n_hidden_layer):
            hidden_layers += create_layer(in_features, in_features),
            hidden_layers += nn.LeakyReLU(),
        return hidden_layers

    return nn.Sequential(
        *create_hidden_layers(),
        create_layer(in_features, out_features),
    )


def init_rnn(rnn):
    for name, param in rnn.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0.0)
        elif 'weight' in name:
            nn.init.xavier_normal_(param)
    return rnn
