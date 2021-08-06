from collections import defaultdict


def get_batch_size(batch):
    ids = batch["id"]
    return len(ids)


def percent(x):
    return x * 100


def pivot(list_of_dict):
    """
    Returns a dictionary of list given a list of dictionaries.
    """
    dict_of_list = defaultdict(list)
    for el in list_of_dict:
        for key, val in el.items():
            dict_of_list[key].append(val)
    return dict_of_list


def identity(x):
    return x


def map_dict(d, *, key_fn=None, value_fn=None):
    """
    Returns a dictionary with keys updated according to `key_fn` and values updated according to `value_fn`. Functions
    are applied, respectively, on each key and value.
    """
    if key_fn is None:
        key_fn = identity

    if value_fn is None:
        value_fn = identity

    dd = {}

    for k, v in d.items():
        dd = {**dd, **{key_fn(k): value_fn(v)}}

    return dd
