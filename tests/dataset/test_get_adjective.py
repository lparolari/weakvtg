from weakvtg.dataset import get_adjective


def test_get_adjective():
    x = ["A red apple", "My dog"]
    def red_detector(xs): return list(filter(lambda x: x == "red", xs))
    def split_on_space(x): return x.split(" ")

    assert get_adjective(x, f_extract_adjective=red_detector, f_nlp=split_on_space) == ["red", "<unk>"]
