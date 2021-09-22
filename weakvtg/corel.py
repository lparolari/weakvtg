def arloss(prediction, prediction_mask, box_mask, concept_direction):

    def average_over_boxes(x, m):
        return x.sum(dim=-1) / m.sum(dim=-1).unsqueeze(-1)

    def average_over_phrases(x, m):
        return x.sum() / m.sum()

    prediction = prediction * concept_direction
    prediction = average_over_boxes(prediction, box_mask)
    prediction = average_over_phrases(prediction, prediction_mask)

    arloss = -prediction

    return arloss
