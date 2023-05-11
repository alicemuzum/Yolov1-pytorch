import torch
import torch.nn as nn
from utils import intersection_over_union

class YoloLoss(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        super(YoloLoss,self).__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.S = S
        self.B = B
        self.C = C
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions, target):
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)

        iou_b1 = intersection_over_union(predictions[..., 21:25], target[..., 21,25]) #21 ve 25 arası bbox
        iou_b2 = intersection_over_union(predictions[..., 26:30], target[..., 21,25])
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)
        iou_maxes, bestbox = torch.max(ious, dim=0)

        exists_box = target[..., 20].unsqueeze(3) # identity of cell_i


        # for box coordinates
        box_predictions = exists_box * (
            (
            bestbox * predictions[..., 26:30]
             + (1- bestbox) * predictions[...,21:25]
            )
        )

        box_targets = exists_box * target[..., 21:25]
        box_predictions[..., 2:4] = torch.sqrt(torch.abs(box_predictions))

