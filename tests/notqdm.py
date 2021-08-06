def notqdm(iterable, *args, **kwargs):
    """
    replacement for tqdm that just passes back the iterable
    useful to silence `tqdm` in tests
    """
    return iterable
