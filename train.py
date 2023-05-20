import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import YoloV1
from dataset import VOCDataset
import argparse
from utils import(
    intersection_over_union,
    non_max_suppression,
    mean_average_precision,
    cellboxes_to_boxes,
    get_bboxes,
    plot_image,
    save_checkpoint,
    load_checkpoint
)
from loss import YoloLoss


seed = 123
torch.manual_seed(seed)

# Hyperparameters
LEARNING_RATE = 2E-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
WEIGHT_DECAY = 0
EPOCHS = 100
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = True
LOAD_MODEL_FILE = "overfit_100ex.pth.tar"
IMG_DIR = "../data/PascalVOC/images"
LABEL_DIR = "../data/PascalVOC/labels"


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes
        return img, bboxes

transform = Compose([transforms.Resize((448,448)), transforms.ToTensor()])

def train_fn(train_loader, model, optimizer, loss_fn):
    """
        Takes DataLoader, Model, Optimizer, Loss Function
        Does forward and backward propagation
    """
    loop = tqdm(train_loader, leave=True)
    mean_loss = []

    for batch_idx, (x, y) in enumerate(loop): # x -> (16, 3, 448,448)  y -> (16,7,7,30)
        x ,y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = loss_fn(out,y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update progress bar
        loop.set_postfix(loss = loss.item())

    print(f"Mean loss was {sum(mean_loss) / len(mean_loss)}")


def main():
    ap = argparse.ArgumentParser(description="Test program")
    ap.add_argument("-d", "--debug",required=False, action="store_const", const=True, help="No input needed. Copy code snippet to the specified position in train.py file to analyze and debug.", default=False)
    ap.add_argument("-f", "--train_file", required=False, default=100, help="Give a number input to decide which csv file will be used for training. Default is 100. Don't forget, file must be named (number)examples.csv and must exists.")
    ap.add_argument("-p", "--plot", required=False, default=False, const=True, action="store_const", help="Just run the file with the flag to plot bounding boxes.")
    args = vars(ap.parse_args())
    isDebug = args["debug"]
    isPlot = args["plot"]
    train_file = args["train_file"]
    print("Debug: ",args["debug"])

    if not isDebug: 

        model = YoloV1(split_size=7, num_boxes=2, num_classes=20).to(DEVICE,non_blocking=True)
        optimizer = optim.Adam(
            model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
        loss_fn = YoloLoss()

        train_dataset = VOCDataset(
            f"../data/PascalVOC/{str(train_file)}examples.csv",
            transform=transform,
            img_dir = IMG_DIR,
            label_dir=LABEL_DIR
        )

        test_dataset = VOCDataset(
            "../data/PascalVOC/test.csv",
            transform= transform,
            img_dir= IMG_DIR, 
            label_dir= LABEL_DIR
        )

        train_loader = DataLoader(
            dataset = train_dataset,
            batch_size= BATCH_SIZE,
            num_workers=NUM_WORKERS,
            pin_memory= PIN_MEMORY,
            shuffle= True,
            drop_last=True
        )

        test_loader = DataLoader(
            dataset = test_dataset,
            batch_size = BATCH_SIZE,
            num_workers= NUM_WORKERS,
            pin_memory= PIN_MEMORY,
            shuffle= True,
            drop_last= True
        )

        if isPlot:
                load_checkpoint(torch.load(LOAD_MODEL_FILE),model, optimizer)
                for x, y in train_loader:
                   x = x.to(DEVICE)
                   for idx in range(16):                  
                       bboxes = cellboxes_to_boxes(model(x))
                       bboxes = non_max_suppression(bboxes[idx], iou_threshold=0.5, threshold=0.4, box_format="midpoint")
                       plot_image(x[idx].permute(1,2,0).to("cpu"), bboxes)
                   import sys
                   sys.exit()

        for epoch in range(EPOCHS):

            pred_boxes, target_boxes = get_bboxes(
                train_loader, model, iou_threshold=0.5, threshold=0.4
            )

            mean_avg_prec = mean_average_precision(
                pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
            )
            print(f"Train mAP: {mean_avg_prec}")

            # if mean_avg_prec > 0.9:
            #    checkpoint = {
            #        "state_dict": model.state_dict(),
            #        "optimizer": optimizer.state_dict(),
            #    }
            #    save_checkpoint(checkpoint, filename=LOAD_MODEL_FILE)
            #    import time
            #    time.sleep(10)

            train_fn(train_loader, model, optimizer, loss_fn)

    else:
            # Copy code snippet here and run the file with "-d" to specify debugging process.
        pass

if __name__ == "__main__":
    main()