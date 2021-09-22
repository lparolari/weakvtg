import functools
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn

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
    def __init__(self, phrases_embedding_net, phrases_recurrent_net, image_embedding_net,
                 get_classes_embedding, get_phrases_embedding, get_phrases_representation, get_concept_similarity,
                 f_similarity):
        super().__init__()

        self.phrases_embedding_net = phrases_embedding_net
        self.phrases_recurrent_net = phrases_recurrent_net
        self.image_embedding_net = image_embedding_net

        self.get_classes_embedding = get_classes_embedding
        self.get_phrases_embedding = get_phrases_embedding
        self.get_phrases_representation = get_phrases_representation
        self.get_concept_similarity = get_concept_similarity

        self.f_similarity = f_similarity

    def forward(self, batch):
        pred_n_boxes = batch["pred_n_boxes"]                    # [b]
        boxes = batch["pred_boxes"]                             # [b, n_boxes, 4]
        boxes_mask = batch["pred_boxes_mask"]                   # [b, n_boxes]
        boxes_features = batch["pred_boxes_features"]           # [b, n_boxes, 2048]
        boxes_class_prob = batch["pred_cls_prob"]               # [b, n_boxes, n_class]
        phrases = batch["phrases"]                              # [b, n_ph+, n_words+]
        phrases_mask = batch["phrases_mask"]                    # [b, n_ph+, n_words+]
        phrases_negative = batch["phrases_negative"]            # [b, n_ph-, n_words-]
        phrases_mask_negative = batch["phrases_mask_negative"]  # [b, n_ph-, n_words-]

        box_class = get_box_class(boxes_class_prob)  # [b, n_boxes]

        n_boxes = pred_n_boxes[0]
        n_ph_pos = phrases.size()[1]
        n_ph_neg = phrases_negative.size()[1]

        _get_concept_similarity = self.get_concept_similarity
        _get_classes_embedding = self.get_classes_embedding
        _get_phrases_embedding = self.get_phrases_embedding
        _get_phrases_features = functools.partial(get_phrases_features,
                                                  get_phrases_embedding=_get_phrases_embedding,
                                                  get_phrases_representation=self.get_phrases_representation)
        _get_image_representation = functools.partial(get_image_representation, embedding_net=self.image_embedding_net)

        _boxes_mask = boxes_mask.squeeze(-1).unsqueeze(-2)  # [b, 1, n_boxes]

        # extract positive/negative features
        img_x_positive = get_image_features(boxes, boxes_features)
        img_x_positive = _get_image_representation(img_x_positive)
        img_x_positive = img_x_positive.unsqueeze(-3).repeat(1, n_ph_pos, 1, 1)

        phrases_x_positive = _get_phrases_features(phrases, phrases_mask)
        phrases_x_positive = phrases_x_positive.unsqueeze(-2).repeat(1, 1, n_boxes, 1)

        img_x_negative = get_image_features(boxes, boxes_features)
        img_x_negative = _get_image_representation(img_x_negative)
        img_x_negative = img_x_negative.unsqueeze(-3).repeat(1, n_ph_neg, 1, 1)

        phrases_x_negative = _get_phrases_features(phrases_negative, phrases_mask_negative)
        phrases_x_negative = phrases_x_negative.unsqueeze(-2).repeat(1, 1, n_boxes, 1)

        # compute positive/negative logits and mask
        # scale logits given classes similarity

        def concept_similarity(phrase, phrase_mask, boxes_mask):
            phrase_mask = phrase_mask.unsqueeze(-1)
            boxes_mask = boxes_mask.unsqueeze(-1)
            box_class_embedding = _get_classes_embedding(box_class)
            phrase_embedding = _get_phrases_embedding(phrase)
            return _get_concept_similarity((phrase_embedding, phrase_mask), (box_class_embedding, boxes_mask))

        positive_concept_similarity = concept_similarity(phrases, phrases_mask, boxes_mask)
        negative_concept_similarity = concept_similarity(phrases_negative, phrases_mask_negative, boxes_mask)

        positive_concept_similarity_mask = positive_concept_similarity > 0
        negative_concept_similarity_mask = negative_concept_similarity > 0

        positive_logits = predict_logits(img_x_positive, phrases_x_positive, f_similarity=self.f_similarity)
        positive_logits = positive_logits * positive_concept_similarity
        positive_logits = torch.masked_fill(positive_logits, positive_concept_similarity_mask == 0, value=-1)
        positive_logits = torch.masked_fill(positive_logits, get_synthetic_mask(phrases_mask) == 0, value=-1)
        positive_logits = torch.masked_fill(positive_logits, _boxes_mask == 0, value=-1)

        negative_logits = predict_logits(img_x_negative, phrases_x_negative, f_similarity=self.f_similarity)
        negative_logits = negative_logits * negative_concept_similarity
        negative_logits = torch.masked_fill(negative_logits, negative_concept_similarity_mask == 0, value=+1)
        negative_logits = torch.masked_fill(negative_logits, get_synthetic_mask(phrases_mask_negative) == 0, value=+1)
        negative_logits = torch.masked_fill(negative_logits, _boxes_mask == 0, value=+1)

        return (positive_logits, negative_logits), (positive_concept_similarity, negative_concept_similarity)


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
    return torch.argmax(probability, dim=-1)


def get_concept_similarity(phrase_embedding_t, box_class_embedding_t, f_aggregate, f_similarity, f_activation):
    """
    Return the similarity between bounding box's class embedding (i.e., textual representation) and phrase.

    Please note that we mask `phrase_embedding` with `phrase_mask` in order to inhibit contribution from masked words
    computed by `f_aggregate`. However, the tensor should be already masked.

    :param phrase_embedding_t: A ([*, d2, d3, d4], [*, d2, d3, 1]) tuple of tensors
    :param box_class_embedding_t: A ([*, d1, d4], [*, d1, 1]) tuple of tensors
    :param f_aggregate: A function that computes aggregate representation of a phrase
    :param f_similarity: A similarity function
    :param f_activation: An activation function, applied on final similarity score
    :return: A [*, d1, d2] tensor
    """
    box_class_embedding, box_class_embedding_mask = box_class_embedding_t
    phrase_embedding, phrase_embedding_mask = phrase_embedding_t

    n_box = box_class_embedding.size()[-2]
    n_ph = phrase_embedding.size()[-3]

    phrase_embedding = f_aggregate(phrase_embedding_t, box_class_embedding_t, dim=-2, f_similarity=f_similarity)

    box_class_embedding = box_class_embedding.unsqueeze(-3).repeat(1, n_ph, 1, 1)
    phrase_embedding = phrase_embedding.unsqueeze(-2).repeat(1, 1, n_box, 1)

    similarity = f_similarity(box_class_embedding, phrase_embedding, dim=-1)

    phrase_embedding_synthetic_mask = phrase_embedding_mask.squeeze(-1).sum(dim=-1, keepdims=True)
    box_class_embedding_mask = box_class_embedding_mask.squeeze(-1).unsqueeze(-2)

    similarity = torch.masked_fill(similarity, mask=phrase_embedding_synthetic_mask == 0, value=-1)
    similarity = torch.masked_fill(similarity, mask=box_class_embedding_mask == 0, value=-1)

    return f_activation(similarity)


def aggregate_words_by_max(phrase_embedding_t, box_class_embedding_t, *_args, f_similarity, **_kwargs):
    """
    Return the embedding of the word word which is the most similar wrt bounding box's class, for each phrase.

    :param box_class_embedding_t: A ([*, d1, d4], [*, d1, 1]) tuple of tensors
    :param phrase_embedding_t: A ([*, d2, d3, d4], [*, d2, d3, 1]) tuple of tensors
    :param f_similarity: A similarity function
    :return: A [*, d2, d1] tensor
    """
    maximum_similarity_box_t = get_maximum_similarity_box(phrase_embedding_t, box_class_embedding_t, f_similarity)
    return get_maximum_similarity_word(phrase_embedding_t, maximum_similarity_box_t)


def aggregate_words_by_mean(phrase_embedding_t, *_args, dim, **_kwargs):
    """
    Return the averaged embedding of words in a phrase, for each phrase.

    :param phrase_embedding_t: A ([*, d1], [*, 1]) tuple of tensors
    :param dim: The dimension to reduce
    :return: A [*] tensor
    """
    (phrase_embedding, phrase_embedding_mask) = phrase_embedding_t

    # inhibit contributions of padded words
    phrase_embedding = torch.masked_fill(phrase_embedding, mask=phrase_embedding_mask == 0, value=0)

    return phrase_embedding.sum(dim=dim) / phrase_embedding_mask.sum(dim=dim)


def get_maximum_similarity_box(phrase_embedding_t, box_class_embedding_t, f_similarity):
    """
    Return the similarity (and the index wrt dimensions `[*, d2, d3]`) of the most similar word wrt bounding box's
    class.

    For each query, we have a [*, n_word, n_box, n_feat] tensor. We first compute similarity on the last dimension,
    which will let us to consider the similarity between a word and the class of a box.

    Then compute the maximum among the last dimension, leading us the most similar box class wrt each word.
    (See sketch below).

    The resulting tensor is returned.

           box1 box2 box3 ...
    word1        x                      word1   x
    word2   y                     ==>   word2   y
    word3             z                 word3   z
    ...

    :param box_class_embedding_t: A ([*, d1, d4], [*, d1, 1]) tuple of tensors
    :param phrase_embedding_t: A ([*, d2, d3, d4], [*, d2, d3, 1]) tuple of tensors
    :param f_similarity: A similarity function
    :return: A (Tensor, LongTensor) tuple with dimensions ([*, d2, d3], [*, d2, d3])
    """
    box_class_embedding, box_class_embedding_mask = box_class_embedding_t
    phrase_embedding, phrase_embedding_mask = phrase_embedding_t

    n_box = box_class_embedding.size()[-2]
    n_ph = phrase_embedding.size()[-3]
    n_word = phrase_embedding.size()[-2]

    def expand_phrase_embedding(phrase_embedding):
        phrase_embedding = phrase_embedding.unsqueeze(-2)

        dims = [1] * phrase_embedding.dim()
        dims[-2] = n_box

        return phrase_embedding.repeat(*dims)

    def expand_box_class_embedding(box_class_embedding):
        box_class_embedding = box_class_embedding.unsqueeze(-3).unsqueeze(-3)

        dims = [1] * box_class_embedding.dim()
        dims[-4] = n_ph
        dims[-3] = n_word

        return box_class_embedding.repeat(*dims)

    # the following two tensor will have size [*, d2, d3, d1, d4], which in a real word scenario may translate
    # to [*, n_ph, n_word, n_box, n_feat]
    phrase_embedding = expand_phrase_embedding(phrase_embedding)
    box_class_embedding = expand_box_class_embedding(box_class_embedding)

    phrase_embedding_mask = expand_phrase_embedding(phrase_embedding_mask).squeeze(-1)
    phrase_embedding_synthetic_mask = phrase_embedding_mask.sum(dim=-2, keepdims=True)
    box_class_embedding_mask = expand_box_class_embedding(box_class_embedding_mask).squeeze(-1)

    similarity = f_similarity(phrase_embedding, box_class_embedding, dim=-1)  # [*, d2, d3, d1]

    # we need to apply some masking for padded values such us word, phrase and box
    similarity = torch.masked_fill(similarity, mask=phrase_embedding_mask == 0, value=-1)  # word
    similarity = torch.masked_fill(similarity, mask=phrase_embedding_synthetic_mask == 0, value=-1)  # phrase
    similarity = torch.masked_fill(similarity, mask=box_class_embedding_mask == 0, value=-1)  # box

    # then, for each word, keep the box with maximum similarity
    return torch.max(similarity, dim=-1)  # [*, d2, d3]


def get_maximum_similarity_word(phrase_embedding_t, maximum_similarity_box_t):
    """
    Return the embedding of the word with maximum similarity wrt given box similarity.

    The box similarity is a tensor where each entry represents the maximum similarity of a word wrt bounding box's
    class for each phrase, for each word. The second component of the tuple is the index of the above best box.

    Consider the maximum similarity tensor as a tensor where, for each query, we have

    (Similarity)               (Index)
    word1   x                  word1   18
    word2   y        and       word2   99
    word3   z                  word3   41

    Note: we ignore index tensor.

    Then, we compute the argmax on similarity tensor in order to retrieve the index of the word with maximum
    similarity. We then use this index to gather from phrase embeddings the representation of chosen word and
    we return it.

    :param phrase_embedding_t: A ([*, d2, d3, d4], [*, d2, d3, 1]) tuple of tensors
    :param maximum_similarity_box_t: A ([*, d2, d3], [*, d2, d3]) tuple of tensors
    :return: A [*, d2, d4] tensor
    """
    phrase_embedding, phrase_embedding_mask = phrase_embedding_t
    maximum_similarity_box, maximum_similarity_box_index = maximum_similarity_box_t

    n_feat = phrase_embedding.size()[-1]

    def expand_index(index):
        index = index.unsqueeze(-1).unsqueeze(-1)

        dims = [1] * index.dim()
        dims[-1] = n_feat

        return index.repeat(*dims)

    # for each phrase, retrieve the word index with maximum similarity among words in phrase
    index = torch.argmax(maximum_similarity_box, dim=-1)  # [*, d2]
    index = expand_index(index)  # [*, d2, 1, d4]

    # get its word embedding
    best_word_embedding = torch.gather(phrase_embedding_t[0], dim=-2, index=index)

    # remove word dimension
    best_word_embedding = best_word_embedding.squeeze(-2)

    return best_word_embedding


def create_phrases_embedding_network(vocab, pretrained_embeddings, embedding_size, freeze=False):
    import re
    import numpy as np

    vocab_size = len(vocab)
    out_of_vocabulary = 0

    embedding_matrix_values = torch.zeros((vocab_size + 1, embedding_size), requires_grad=(not freeze))

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
                word_count += 1
                embedding_idx = pretrained_embeddings.stoi[word_tmp]
                word_rep = np.add(word_rep, pretrained_embeddings.vectors[embedding_idx])

        if word_count > 0:
            word_rep = word_rep / word_count
            embedding_matrix_values[word_idx, :] = word_rep
        else:
            out_of_vocabulary += 1
            nn.init.normal_(embedding_matrix_values[word_idx, :])

    if out_of_vocabulary != 0:
        logging.warning(f"Found {out_of_vocabulary} words out of vocabulary.")

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
