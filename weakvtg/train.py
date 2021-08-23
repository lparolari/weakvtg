import logging
import os
import tempfile
import time

import numpy as np
import torch
import wandb

from weakvtg import iox
from weakvtg.loss import get_boxes_predicted
from weakvtg.mask import get_synthetic_mask
from weakvtg.prettyprint import pp
from weakvtg.timeit import get_fancy_eta, get_fancy_time, get_hms, get_delta
from weakvtg.utils import get_batch_size, percent, pivot, map_dict


def epoch(loader, model, optimizer, criterion, train=True):
    fancy_mode = "Train" if train else "Valid"
    n_batch = len(loader)

    total_examples = 0
    total_loss = 0.
    total_accuracy = 0.
    total_p_accuracy = 0.

    model.train(train)

    start_time = time.time()

    for i, batch in enumerate(loader):
        optimizer.zero_grad()

        output = model(batch)

        loss, iou, accuracy, p_accuracy = criterion(batch, output)

        if train:
            loss.backward()
            optimizer.step()

        # computing some statistics
        bs = get_batch_size(batch)

        total_examples += bs

        total_loss += loss.item() * bs
        total_accuracy += accuracy.item() * bs
        total_p_accuracy += p_accuracy.item() * bs

        fancy_batch_no = i + 1
        end_time = time.time()
        eta = get_fancy_eta(start_time, end_time, current=fancy_batch_no, total=n_batch)
        start_time = end_time
        logging.debug(f"{fancy_mode} {fancy_batch_no}/{n_batch}, "
                      f"loss: {pp(total_loss / total_examples)}, "
                      f"accuracy: {pp(percent(total_accuracy / total_examples))}, "
                      f"p_accuracy: {pp(percent(total_p_accuracy / total_examples))} | ETA: {eta}")

    loss = total_loss / total_examples
    accuracy = percent(total_accuracy / total_examples)
    p_accuracy = percent(total_p_accuracy / total_examples)

    return {"loss": loss, "accuracy": accuracy, "p_accuracy": p_accuracy}


def train(train_loader, valid_loader, model, optimizer, criterion,
          n_epochs=15, start_epoch=0, save_folder=None, suffix=None):
    train_results = []
    valid_results = []

    start_time = time.time()

    for i in range(start_epoch, n_epochs):
        logging.info(f"Start epoch {i + 1}")

        train_out = epoch(train_loader, model, optimizer, criterion, train=True)
        logging.info(f"Training completed. {pp(train_out)}")

        valid_out = epoch(valid_loader, model, optimizer, criterion, train=False)
        logging.info(f"Validation completed. {pp(valid_out)}")

        train_results += [train_out]
        valid_results += [valid_out]

        # computing some statistics
        fancy_epoch_no = i + 1
        end_time = time.time()
        running_time = get_fancy_time(*get_hms(get_delta(start_time, end_time)))
        start_time = end_time
        logging.info(f"Complete epoch {fancy_epoch_no} in {running_time}")
        wandb.log({**map_dict(train_out, key_fn=lambda x: f"train_{x}"),
                   **map_dict(valid_out, key_fn=lambda x: f"valid_{x}")})

        # save model on file
        save_model(model, fancy_epoch_no, optimizer=optimizer, folder=save_folder, suffix=suffix)

    train_results = pivot(train_results)
    valid_results = pivot(valid_results)

    logging.info(f"Training completed.")

    return train_results, valid_results


def test(loader, model, optimizer, criterion):
    test_out = epoch(loader, model, optimizer, criterion, train=False)
    logging.info(f"Testing completed. {pp(test_out)}")


def test_example(dataset, loader, model, optimizer, criterion, vocab):
    def scale(boxes, *, width, height):
        scaled_boxes = np.zeros_like(boxes)
        scaled_boxes[..., 0] = np.round(boxes[..., 0] * width)
        scaled_boxes[..., 2] = np.round(boxes[..., 2] * width)
        scaled_boxes[..., 1] = np.round(boxes[..., 1] * height)
        scaled_boxes[..., 3] = np.round(boxes[..., 3] * height)
        return scaled_boxes

    def xyxy2xywh(x):
        """
        Transform coordinates from [x_min, y_min, x_max, y_max] to [x_min, y_min, width, height].
        """
        y = np.copy(x)
        y[..., 2] = x[..., 2] - x[..., 0]
        y[..., 3] = x[..., 3] - x[..., 1]
        return y

    def ph(x): return " ".join([vocab.vocab.itos_[idx] if vocab.vocab.itos_[idx] != "<unk>" else "" for idx in x])

    def pp_score(score): return f"min={np.min(score):.3f}, max={np.max(score):.3f}, avg={np.mean(score):.3f}"

    def get_iou_all_boxes(phrases_2_crd, boxes_pred_topk):
        # Up to now, the model only compute the IoU between the best (greater score) box and the ground truth.
        # However, we want to know the IoU of all top-k boxes.
        from weakvtg.anchors import bbox_final_iou as iou

        n_box = boxes_pred_topk.size()[-2]
        boxes_gt = phrases_2_crd.unsqueeze(-2).repeat(1, 1, n_box, 1)
        iou_all = iou(boxes_pred_topk, boxes_gt)

        return iou_all

    def show_image(image, title, sentence, queries, boxes_pred, boxes_gt):
        """
        :param image: A numpy array with shape (width, height, depth)
        :param title: A string representing plot title
        :param sentence: A string representing example's sentence
        :param queries: A list of strings with example's queries
        :param boxes_pred: A list of bounding box for each query
        :param boxes_gt: A list of bounding box
        """
        import random
        import matplotlib as pl
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        pl.rcParams["figure.dpi"] = 230

        boxes_pred = xyxy2xywh(boxes_pred)
        boxes_gt = xyxy2xywh(boxes_gt)

        colors = [(random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)) for _ in queries]
        font_size = 8
        text_props = dict(facecolor="blue", alpha=0.5)

        plt.figtext(0.5, 0.01, f"{title}", ha="center", fontsize=font_size, wrap=True)

        # plot predictions
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title("Prediction", fontdict={"fontsize": font_size})

        ax = plt.gca()

        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)

        for i in range(len(queries)):
            query = queries[i]
            color = colors[i]

            for j in range(len(boxes_pred[i])):
                box = boxes_pred[i][j]

                x, y = box[0], box[1]
                xy = (x, y)
                width, height = box[2], box[3]

                if j == 0:
                    plt.text(x, y - 2, query, bbox=text_props, fontsize=5, color="white")
                if j > 0:
                    plt.text(x, y - 2, j, bbox=text_props, fontsize=5, color="white")

                rect = patches.Rectangle(xy, width, height, linewidth=1, edgecolor=[*color, .5], facecolor=[*color, .2])
                ax.add_patch(rect)

        # plot ground truth
        plt.subplot(1, 2, 2)
        plt.imshow(image)
        plt.title("Ground Truth", fontdict={"fontsize": font_size})

        ax = plt.gca()

        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)

        for query, box, color in zip(queries, boxes_gt, colors):
            x, y = box[0], box[1]
            xy = (x, y)
            width, height = box[2], box[3]

            plt.text(x, y - 2, query, bbox=text_props, fontsize=5, color="white")

            rect = patches.Rectangle(xy, width, height, linewidth=1, edgecolor=color, facecolor="none")
            ax.add_patch(rect)

        plt.show()

    for i, batch in enumerate(loader):
        idx = batch["id"]
        idx_negative = batch["id_negative"]
        height = batch["image_h"]
        width = batch["image_w"]
        boxes = batch["pred_boxes"]
        sentence = batch["sentence"]
        phrases = batch["phrases"]
        phrases_mask = batch["phrases_mask"]
        phrases_negative = batch["phrases_negative"]
        phrases_mask_neg = batch["phrases_mask_negative"]
        phrases_2_crd = batch["phrases_2_crd"]
        phrases_synthetic = get_synthetic_mask(phrases_mask)

        n_ph = phrases.size()[-2]

        # --- forward model
        optimizer.zero_grad()
        output = model(batch)
        loss, iou, accuracy, p_accuracy = criterion(batch, output)
        score_positive, score_negative = output[0]
        # Instead of retrieving the single best bounding box, we retrieve top-K
        boxes_pred = get_boxes_predicted(boxes, score_positive, phrases_synthetic)
        scores_topk, scores_topk_index = torch.topk(score_positive, k=1)
        # quick and dirty way to gather to gather from boxes the top-k score :)
        boxes_pred_topk = torch.gather(boxes.unsqueeze(-3).repeat(1, n_ph, 1, 1), dim=-2, index=scores_topk_index.unsqueeze(-1).repeat(1, 1, 1, 4))
        # --- forward model

        height_ = height.detach().numpy()[0]
        width_ = width.detach().numpy()[0]
        idx_ = idx[0].detach().numpy()
        idx_negative_ = idx_negative[0].detach().numpy()
        sentence_ = sentence[0].detach().numpy()
        sentence_str_ = ph(sentence_)
        phrases_ = phrases[0].detach().numpy()
        phrases_str_ = [ph(x) for x in phrases_]
        phrases_mask_ = phrases_mask.squeeze(-1)[0].detach().numpy()
        phrases_mask_neg_ = phrases_mask_neg.squeeze(-1)[0].detach().numpy()
        phrases_negative_ = phrases_negative[0].detach().numpy()
        score_positive_ = score_positive[0].detach().numpy()
        score_negative_ = score_negative[0].detach().numpy()
        loss_ = loss.item()
        iou_ = iou[0].detach().numpy()
        iou_all_ = get_iou_all_boxes(phrases_2_crd, boxes_pred_topk)[0].detach().numpy()
        accuracy_ = accuracy.item()
        p_accuracy_ = p_accuracy.item()
        boxes_pred_ = scale(boxes_pred_topk[0].detach().numpy(), width=width_, height=height_)
        boxes_gt_ = scale(phrases_2_crd[0].detach().numpy(), width=width_, height=height_)

        img_ = iox.load_image(os.path.join(dataset.image_filepath, f"{idx_}.jpg"))

        print(f"({i}) Example: pos={idx_}, neg={idx_negative_}")
        print(f"({i}) Image: w={width_}, h={height_}")
        print(f"({i}) Loss: {loss_}")
        print(f"({i}) Accuracy: {accuracy_}")
        print(f"({i}) Pointing Game Accuracy: {p_accuracy_}")
        for j in range(len(phrases_)):
            print(f"({i}) ({j}) Phrase+: {ph(phrases_[j])}")
            if j < phrases_negative_.shape[0]:
                print(f"({i}) ({j}) Phrase-: {ph(phrases_negative_[j])}")
            print(f"({i}) ({j}) Mask+: {phrases_mask_[j]}")
            if j < phrases_negative_.shape[0]:
                print(f"({i}) ({j}) Mask-: {phrases_mask_neg_[j]}")
            print(f"({i}) ({j}) Scores+: {pp_score(score_positive_[j])}")
            if j < score_negative_.shape[0]:
                print(f"({i}) ({j}) Scores-: {pp_score(score_negative_[j])}")
            print(f"({i}) ({j}) IoU (model): {iou_[j]}")
            print(f"({i}) ({j}) IoU (top-k): {iou_all_[j]}")
            print(f"({i}) ({j}) Best pred: {boxes_pred_[j]}")
            print(f"({i}) ({j}) Boxes GT: {boxes_gt_[j]}")

        show_image(img_, f"{idx_} (#{i}): {sentence_str_}", sentence_str_, phrases_str_, boxes_pred_, boxes_gt_)


def save_model(model, epoch, optimizer=None, scheduler=None, folder=None, suffix=None):
    if folder is None:
        folder = tempfile.gettempdir()
    if suffix is None:
        suffix = "default"

    filepath = f"{folder}/model_{suffix}_{epoch}.pth"

    data = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        **({"optimizer_state_dict": optimizer.state_dict()} if optimizer is not None else {}),
        **({"scheduler_state_dict": scheduler.state_dict()} if scheduler is not None else {}),
    }

    torch.save(data, filepath)

    logging.info(f"Model saved to {filepath}")


def load_model(filepath, model, optimizer=None, scheduler=None, device=None):
    checkpoint = torch.load(filepath, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    start_epoch = checkpoint["epoch"]

    logging.info(f"Loaded model from {filepath}")

    return start_epoch
