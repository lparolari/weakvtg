def get_batch_size(batch):
    ids = batch["id"]
    return len(ids)


def percent(x):
    return x * 100
