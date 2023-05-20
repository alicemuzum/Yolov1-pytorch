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
        predictions = predictions.reshape(-1, self.S, self.S, self.C + (self.B * 5))
        iou_b1 = intersection_over_union(predictions[..., 21:25], target[..., 21:25]) #21 ve 25 arası bbox
        iou_b2 = intersection_over_union(predictions[..., 26:30], target[..., 21:25])
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)
        iou_maxes, bestbox = torch.max(ious, dim=0)

        exists_box = target[..., 20].unsqueeze(3) # target'ı biz VOCDataset classında getitem methodunda oluşturuyoruz.
                                                  # 20. eleman resimde obje varsa 1 yoksa 0 olur. Burda da (Batch_size, 7,7,30)
                                                  # olan targetın tüm 20. elemanlarını sliceladıktan sonra elimize (B_s, 7, 7)'lik
                                                  # bir matrix geçiyor, onunla işleme devam etmek için sonun bir boyut daha ekliyoruz.
                                                  # unsqueeze bunu index değerine 1 koymakla yapıyor yani;
                                                  # (B_s, 7, 7) -> (B_s, 7, 7, 1)

        # for box coordinates
        box_predictions = exists_box * (        # BESTBOX BİR ARGMAX DEĞERİ YANİ İOUS'DAN HANGİ ELEMAN EN YÜKSEKSE ONUN İNDEXİNİ DÖNÜYOR.
            (                                   # ALABİLECEĞİ DEĞERLER 0 YADA 1. BURDA DA HANGİSİ BESTBOX İSE ONUN PRED DEGERLERİNİ ALIYORUZ
            bestbox * predictions[..., 26:30]   # (B_s, 7, 7, 1) * (B_s, 7, 7, 4) = (B_s, 7, 7, 4)
             + (1- bestbox) * predictions[...,21:25]
            )
        )

        box_targets = exists_box * target[..., 21:25] # TARGET (S,S,25) SON 5 İ BOX COORD
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(torch.abs(box_predictions[..., 2:4] + 1e-6))

        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        box_loss = self.mse(                            # Flatten box_predictions'ın (7,7,4) sondan ikincisi dahil düzlenmesini sağlıyor
            torch.flatten(box_predictions, end_dim= -2),# yani sonuc (7*7,4) den (49,7) oluyor ve iki boyutlu matris mse ye girebiliyor.
            torch.flatten(box_targets, end_dim= -2)
        )



        #for object loss
        pred_box = (                                    # (7,7,1)
            bestbox * predictions[..., 25:26] + (1 - bestbox) * predictions[..., 20:21]
        )
        # (N*S*S)
        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * target[..., 20:21])
        )

        # for no object loss (N, S, S, 1) -> (N, S*S)
        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 20:21], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1)
        )

        no_object_loss += self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 25:26], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1)
        )

        # for class loss (N, S, S, 20) -> (N*S*S, 20)
        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :20], end_dim= -2),
            torch.flatten(exists_box * target[..., :20], end_dim=-2),
        )

        # general loss
        loss = (
            self.lambda_coord * box_loss
            + object_loss
            + self.lambda_noobj * no_object_loss
            + class_loss
        )

        return loss