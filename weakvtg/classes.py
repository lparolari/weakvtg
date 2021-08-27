from weakvtg.iox import load_txt


def get_specials():
    return ["__background__"]


def get_classes(path):
    classes = load_classes(path)
    classes = add_specials(classes, get_specials())
    return classes


def get_class(classes, idx):
    return classes[idx]


def load_classes(path):
    classes = load_txt(path)
    classes = list(map(lambda x: x.lower().strip(), classes))
    return classes


def add_specials(classes, specials):
    return [*specials, *classes]
