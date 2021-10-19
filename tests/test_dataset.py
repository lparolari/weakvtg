import pytest

from weakvtg.dataset import process_example


@pytest.fixture
def image_example():
    return {
        "id": 10000,
        "image_w": 360,
        "image_h": 480,
        "image_d": 3,
        "image_boxes_coordinates": [[0.0, 78.0, 360.0, 480.0]],
        "pred_n_boxes": 2,
        "pred_boxes": [[1, 2, 3, 4], [5, 6, 7, 8]],
        "pred_cls_prob": [[.3, .5, .2], [.1, .2, 7]],
        "pred_attr_prob": [[.05, .05, .4, .5], [.15, .25, .4, .2]],
        "pred_boxes_features": [[1, 2, 3, 4, 5], [11, 12, 13, 14, 15]],
    }


@pytest.fixture
def caption_example():
    return {
        "id": 100001,
        "sentence": "the ground that\'s not grass",
        "phrases": ["the ground that\'s not grass"],
        "n_phrases": 1,
        "ewiser_chunks": ["the ground"],
        "ewiser_heads": ["ground"],
        "ewiser_begin": [4],
        "ewiser_end": [10],
        "ewiser_n": [10],
        "ewiser_yago_entities": [
            ["wordnet_land_109335240", "wordnet_land_109334396", "wordnet_earth_114842992", "wordnet_soil_114844693",
             "wordnet_ground_103462747", "wordnet_turf_109463919", "wordnet_geological_formation_109287968", "type_v",
             "wordnet_growth_109295338", "wordnet_property_104012260"]],
        "phrases_2_crd": [[0.0, 78.0, 360.0, 480.0]],
    }


processed_keys = ["id", "sentence", "phrases", "n_phrases", "ewiser_chunks", "ewiser_heads", "ewiser_begin",
                  "ewiser_end", "ewiser_n", "ewiser_yago_entities", "phrases_2_crd", "image_w", "image_h", "image_d",
                  "image_boxes_coordinates", "pred_n_boxes", "pred_boxes", "pred_cls_prob", "pred_attr_prob",
                  "pred_boxes_features", "pred_active_box_index", "pred_boxes_mask"]


@pytest.mark.parametrize("key", processed_keys)
def test_preprocess_example(image_example, caption_example, key):
    assert key in process_example({**caption_example, **image_example},
                                  f_nlp=lambda x: x, f_extract_noun_phrase=lambda x: x,
                                  f_extract_adjective=lambda x: x).keys()
