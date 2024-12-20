# %%
from histgram_3dconv_norefine import MyModel, ISNetBackbone
from video_histgram_dataloader import CustomDataset

# %%
import random
import argparse
parser = argparse.ArgumentParser(description="Training script")
parser.add_argument('--fold', type=int, required=True, help='Fold number for cross-validation')
parser.add_argument('--gpu', type=str, default="0", help='GPU id to use')
parser.add_argument('--refine_mode', type=str, default="residual", help='Refine mode', choices=["residual", "direct"])
parser.add_argument('--noise_level', type=float, default=1, help='Noise level') 
parser.add_argument('--steps', type=int, default=25000, help='Number of steps to train')
parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate') 
parser.add_argument('--weight_decay', type=float, default=1e-2, help='Weight decay')
parser.add_argument('--mask_upsample', type=str, default="interpolate", help='Mask upsample method', choices=["interpolate", "transpose_conv", "shuffle"])
parser.add_argument('--refine_see_bg', action="store_true", help='If refine operator can see background')
parser.add_argument('--backbone', type=str, default="4", help='Backbone size to use', choices=["0", "1", "2", "3", "4", "5", "is"])
parser.add_argument('--refine_steps', type=int, default=5, help='Number of refine steps')
parser.add_argument('--background_type', type=str, default="mog2", choices=["mog2", "sub"], help='Background type, mog2 means MOG2, sub means SuBSENSE')
parser.add_argument('--histogram', action="store_true", help='If use histogram')
parser.add_argument('--clip', type=float, default=1, help='Gradient clip norm')
parser.add_argument('--note', type=str, default="", help='Note for this run (for logging purpose)')
parser.add_argument('--conf_penalty', type=float, default=0, help='Confidence penalty, penalize the model if it is too confident')
parser.add_argument('--image_size', type=int, default=512, help="Image size", choices=[512, 640])
parser.add_argument('--hard_shadow', action="store_true", help='If use hard shadow')
parser.add_argument('--lambda2', type=float, default=30, help='Lambda2 for pretrained weights and new weights')
parser.add_argument('--lr_min', type=float, default=1e-5, help='Minimum learning rate')
parser.add_argument('--print_every', type=int, default=100, help='Print every n steps')
parser.add_argument('--val_size', type=int, default=4096, help='Validation size')
parser.add_argument('--lora', action="store_true", help='If use LoRA')
parser.add_argument('--save_name', type=str, default=str(random.randint(1000000, 99999999999)), help='Model save name')
parser.add_argument('--final_weight_decay', type=float, default=3e-2, help='Final weight decay')
parser.add_argument('--use_difference', action="store_true", help='If use difference between current and background ratehr than background frame')
parser.add_argument('--num_classes', type=int, default=3, help='Number of classes')
parser.add_argument('--recent_frames', type=str, default="conv3d", help='Recent frames method', choices=["conv3d", "linear", "none"])
parser.add_argument('--checkpoint', type=str, default="", help='Load checkpoint')
import shlex
args = parser.parse_args(shlex.split('--fold 1 --steps 15000 --learning_rate 5e-5 --weight_decay 1.3e-2 --background_type mog2 --refine_step 1 --backbone is --image_size 640 --gpu 0 --clip 2 --conf_penalty 0.05 --mask_upsample shuffle --lambda2 100 --hard_shadow --print_every 200 --recent_frames conv3d --lr_min 1e-6 --num_classes 3'))


# %%
import torch

val_dataset = CustomDataset("/home/wg25r/fastdata/CDNet", "/home/wg25r/fastdata/CDNet", args, "val", filename=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=8, shuffle=True, num_workers=32, pin_memory=True, persistent_workers=True, prefetch_factor=2, drop_last=True) 

# %%wanglezhege
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
if args.backbone == "is":
    model = ISNetBackbone(args)
else:
    model = MyModel(args)
model = torch.nn.DataParallel(model).cuda()
# model.load_state_dict(torch.load("86677878221_40.pth"))
model.load_state_dict(torch.load("41047147845_12.pth", weights_only=False))

# %%
class BinaryConfusion:
    def __init__(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0

    def update(self, y_true, y_pred):
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        assert y_true.shape == y_pred.shape
        self.tp += torch.sum((y_true == 1) & (y_pred == 1))
        self.fn += torch.sum((y_true == 1) & (y_pred == 0))
        self.fp += torch.sum((y_true == 0) & (y_pred == 1))
        self.tn += torch.sum((y_true == 0) & (y_pred == 0))


    def get_f1(self):
        precision = self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0
        recall = self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0
        if precision + recall == 0:
            return 0
        return 2 * (precision * recall) / (precision + recall)

# %%
videonames = list(set(["_".join(i.split("_")[:2]) for i in os.listdir("/mnt/fastdata/CDNet/in")]))

# %%
torch.save(model.state_dict(), "model_ho.pth")
import tqdm
os.makedirs("results", exist_ok=True)
confusions = {}
for i in videonames:
    confusions[i] = BinaryConfusion()
import cv2
with torch.no_grad():
    model.eval()
    for i, (images, masks, ROI, filenames) in enumerate(tqdm.tqdm(val_dataloader)):
    # print(1)
        video_names = ["_".join(filename.split("_")[:2]) for filename in filenames]
        images = images.cuda()
        masks = masks.cuda()
        outputs = model(images).argmax(1) == 1
        assert outputs.shape == masks.shape == ROI.shape
        for j in range(outputs.shape[0]):
            outputs_ = outputs[j][(ROI[j]>0.9)]
            masks_ = masks[j][(ROI[j]>0.9)]
            assert outputs_.shape == masks_.shape
            confusions[video_names[j]].update(masks_, outputs_ > 0.5)
            
            # concate masks[j], output[j], images[j] into one image and save it
            mask = masks[j].cpu().numpy()
            mask = mask * 255
            mask = cv2.cvtColor(mask.astype("uint8"), cv2.COLOR_GRAY2BGR)
            mask = mask.astype("uint8")
            output = outputs[j].cpu().numpy()
            output = output * 255
            output = cv2.cvtColor(output.astype("uint8"), cv2.COLOR_GRAY2BGR)
            output = output.astype("uint8")
            image = images[j].cpu().numpy()
            image = image
            image = image.astype("uint8")
            image = image.transpose(1, 2, 0)
            # print(image.shape, mask.shape, output.shape)
            roi = ROI[j].cpu().numpy().astype("uint8") * 255
            roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
            to_save = cv2.hconcat([mask, image[:,:,:3], output, roi])
            cv2.imwrite(f"results/{filenames[j]}.png", to_save)
                    
        if i % 100 == 0:
            f1 = {}
            for i in videonames:
                f1[i] = confusions[i].get_f1()

            # %%
            f1_class = {}
            for i in videonames:
                class_name = i.split("_")[0]
                if class_name not in f1_class:
                    f1_class[class_name] = []
                if f1[i] != 0:
                    f1_class[class_name].append(f1[i].item())

            print(f1_class)
            
            
f1 = {}
for i in videonames:
    f1[i] = confusions[i].get_f1()

# %%
f1_class = {}
for i in videonames:
    class_name = i.split("_")[0]
    if class_name not in f1_class:
        f1_class[class_name] = []
    if f1[i] != 0:
        f1_class[class_name].append(f1[i].item())

print(f1_class)