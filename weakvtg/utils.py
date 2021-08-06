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
