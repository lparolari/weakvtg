import torch
import torch.nn as nn

from weakvtg.mask import get_synthetic_mask


class WeakVtgModel(nn.Module):
    def forward(self, batch):
        raise NotImplementedError


class MockModel(WeakVtgModel):
    def forward(self, batch):
        boxes = batch["pred_boxes"]
        phrases = batch["phrases"]
        phrases_mask = batch["phrases_mask"]
        phrases_synthetic = get_synthetic_mask(phrases_mask)

        size = (*phrases_synthetic.size()[:-1], boxes.size()[-2])

        return (torch.rand(size, requires_grad=True), torch.rand(size, requires_grad=True)),
