import torch

from weakvtg.loss import sim_mm_n


def test_sim_mm_n():
    x = torch.tensor([[[[0.0000, 0.0000],
                        [0.0000, 0.0000]],

                       [[0.3620, 0.6571],
                        [0.4334, 0.8536]]],


                      [[[0.7001, 0.5417],
                        [0.5703, 0.6368]],

                       [[0.0000, 0.0000],
                        [0.0000, 0.0000]]]])  # [b, b, n_ph, n_box]

    index = torch.tensor([[[0, 1], [1, 1]]])  # [1, b, n_ph]
    index = index.squeeze(0)  # [b, n_ph]

    # TODO: test does not work because we exp the matrix befor return!

    assert torch.equal(sim_mm_n(x, index), torch.tensor([[0.4334, 0.8536], [0.6368, 0.6368]]))


def test_1():
    import torch
    batch_size = 2
    n_ph = 2
    n_b = 2

    x_pos = torch.tensor([[[[0.7001, 0.5417],
                        [0.5703, 0.6368]],

                       [[0.3620, 0.6571],
                        [0.4334, 0.8536]]]])

    x = torch.tensor([[[[0.0000, 0.0000],
                        [0.0000, 0.0000]],

                       [[0.3620, 0.6571],
                        [0.4334, 0.8536]]],


                      [[[0.7001, 0.5417],
                        [0.5703, 0.6368]],

                       [[0.0000, 0.0000],
                        [0.0000, 0.0000]]]])  # [b, b, n_ph, n_box]

    # x_orig = torch.tensor([[[0.7, 0.5, 0.1],
    #                         [0.2, 0.6, 0.3]],
    #
    #                        [[0.3, 0.6, 0.5],
    #                         [0.4, 0.3, 0.9]]])  # [b, n_ph, n_b]

    index = torch.argmax(x_pos, dim=-1).squeeze(0)  # [b, n_ph]
    # index = torch.tensor([[0, 1], [1, 2]])                        # [b, n_ph]

    # x deve avere il seguente shape: [b, n_ph, b, n_b] interpretabile come:
    # la prima dimensione rappresenta il batch che delimita la sentence, la seconda dimensione delimita il numero delle
    # queries nella sentence, la dimensione 3 delimita l'immagine scelta e la quarta dimensione rappresenta le boxes presenti
    # nella immagine.
    # Se cosi' fosse, allora x = x.permute(0, 2, 1, 3) e' la seguente matrice.
    # Quindi, quando le prime due dimensioni sono uguali, abbiamo gli score positivi (numeratore)
    # x = torch.tensor([[[[0.0, 0.0, 0.0],
    #                     [0.0, 0.0, 0.0]],
    #
    #                    [[0.3, 0.6, 0.5],
    #                     [0.4, 0.3, 0.9]]],
    #
    #                   [[[0.7, 0.5, 0.1],
    #                     [0.2, 0.6, 0.3]],
    #
    #                    [[0.0, 0.0, 0.0, ],
    #                     [0.0, 0.0, 0.0, ]]]])  # [b, b, n_ph, n_b]

    def sim_mm_n(prediction, index):
        # INPUT SHAPE
        # prediction has shape [b, b, n_ph, n_b]
        # index has shape [b, n_ph]

        # costruzione del tensore da adoperare nel gather. Il vettore index contiene l'indice della box (positiva)
        # da recuperare per ogni query. Poiche' noi vogliamo b*n_ph query negative, ingrandiamo il tensore.
        # Da interpretare come [for each image, all the bounding boxes indeces for each query]
        index = index.unsqueeze(-1)  # [b, n_ph, 1]
        index = index.unsqueeze(-1)  # [b, n_ph, 1, 1]
        index = index.repeat(1, 1, batch_size, n_ph)  # [b, n_ph, b, n_ph]
        index = index.reshape(batch_size, n_ph, -1)  # [b, n_ph, b*n_ph]

        # costruzione del tensore che utilizamo per ritornare l'informazione della similarita' predetta.
        # Permuto il tensore per renderlo piu' leggibile.
        # [for each sentence, for each query, for each image, similarity score for each box in the image]
        prediction = prediction.permute(0, 2, 1, 3)  # [b, n_ph, b, n_b]
        # La seguente linea da errore, anche se sembra quella piu' adatta.
        # [for each image, for each box, for each sentence, similarity for each query]
        # prediction = prediction.permute(2, 3, 0, 1)             # [b, n_b, b, n_ph]
        # La seguente linea da i risultati corretti.
        # [for each sentence, for each box, for each image, similarity for each query]
        prediction = prediction.permute(0, 3, 2, 1)  # [b, n_b, b, n_ph]
        prediction = prediction.reshape(batch_size, n_b, -1)  # [b, n_b, b*n_ph]

        # gather sulla dimensione delle bounding boxes.
        # [for each image, for each of the selected boxes (tante quante il numero di query positive), similarity for each query]
        score_n = prediction.gather(1, index)  # [b, n_ph, b*n_ph]

        # A QUESTO PUNTO SI PUO SCEGLIERE COSA FARE
        # 1) per avere una sola query al denominatore
        # score_n = score_n.max(dim=-1)[0]                            # [b, n_ph]
        # score_n = score_n.exp()                                   # [b, n_ph]
        # 2) per avere tutte le queries
        score_n = score_n.exp()                                   # [b, n_ph, b, n_ph]
        score_n = score_n.sum(dim=-1)                             # [b, n_ph]
        # CODE JUST FOR CHECK
        # score_n = score_n.max(dim=-1)[0]  # [b, n_ph]
        return score_n

    result = sim_mm_n(x, index)
    print("===== RESULTS")
    print(result, result.shape)
    torch.equal(result, torch.tensor([[0.400, 0.6000], [0.6000, 0.3000]]))

    assert torch.equal(sim_mm_n(x, index), torch.tensor([[0.4334, 0.8536], [0.6368, 0.6368]]))