import torch


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
    :return: A [*, d2, d1] tensor
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


def binary_threshold(x, threshold):
    """
    Return a tensor whose values are 1 whether `x_i > threshold`, 0 otherwise.

    :param x: A [*] tensor
    :param threshold: A float value
    :return: A [*] tensor
    """
    return torch.as_tensor(x > threshold, dtype=torch.float)


def get_concept_similarity_direction(similarity, f_activation):
    """
    Activate the similarity score wrt an activation function.

    :param similarity: A [*] tensor
    :param f_activation: An activation function f([*]) -> [*]
    :return: A [*] tensor
    """
    return f_activation(similarity)


def get_attribute_similarity_direction(similarity, box_attribute_mask, adjective_mask, *, f_activation):
    """
    Activate the similarity score wrt an activation function.

    :param similarity: A [d1, ..., dN] tensor
    :param box_attribute_mask: A [c1, ..., cN] tensor, must be broadcastable
    :param adjective_mask: A [b1, ..., bN] tensor, must be broadcastable
    :param f_activation: An activation function f([d1, ..., dN]) -> [d1, ..., dN]
    :return: A [d1, ..., dN] tensor
    """
    similarity = torch.masked_fill(similarity, mask=box_attribute_mask == 0, value=1)
    similarity = torch.masked_fill(similarity, mask=adjective_mask == 0, value=1)
    return f_activation(similarity)
