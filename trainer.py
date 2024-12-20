import torch
import numpy as np
import os
def printred(text):
    print(f"\033[31m{text}\033[0m")

def printgreen(text):
    print(f"\033[32m{text}\033[0m")
import copy
batch_size = 8
gradient_accumulation = False
REFINE = True
from sklearn.metrics import f1_score
from peft import LoraConfig, TaskType
from peft import get_peft_model

class BinaryConfusion:
    def __init__(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0

    def update(self, y_true, y_pred):
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
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


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )
    
    

class Trainer:
    def __init__(self, model, optimizer, lr_scheduler, train_dataloader, val_dataloader, logger, loss_fn, args, regularization_loss):
        # if args.lora:
        #     config = LoraConfig(
        #         r=128,
        #         lora_alpha=128,
        #         lora_dropout=0.1,
        #         target_modules=["query", "key", "value", "dense1", "dense2", "dwconv.dwconv"],
        #         bias="none",
        #     ) 
        #     model = get_peft_model(model, config)
        #     print_trainable_parameters(model) 
        # lora should be only for the backbone
        self.model = model
        self.model_0 = copy.deepcopy(list(model.module.backbone.parameters()))
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.logger = logger
        self.loss_fn = loss_fn
        self.running_loss = []
        self.running_f1 = []
        self.runing_reg_loss = []
        self.step = 0
        self.validate_f1 = False
        self.args = args
        self.regularization_loss = regularization_loss
        # self.scaler = torch.GradScaler()

    def getgrad(self):
        grads = [] #it was just [0] 
        for param in self.model.parameters():
            if param.grad is not None: #why need this
                grads.append(param.grad.view(-1).cpu())
        grads = torch.cat(grads) 
        return grads 
    
    def train_step(self, X, Y, ROI):       
        self.model.train()
        self.optimizer.zero_grad()

        pred = self.model(X.to("cuda:0")) 
        # print(pred.shape)
        loss = self.loss_fn(pred, Y, ROI)
        reg_loss = self.regularization_loss(self.model_0, self.model.module.backbone.parameters())
        loss = loss + reg_loss * self.args.lambda2
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0) 
        self.optimizer.step()
        e = 1e-6
        # pred = torch.sigmoid(pred)
        pred = pred.argmax(dim=1)
        
        pred_ = (pred==1)[ROI > 0.9]
        if self.args.hard_shadow:
            pred = torch.where(ROI > 0.9, pred, 0)
            pred = pred / 2.0
        else:
            pred = torch.where(ROI > 0.9, pred, 0)
        pred_ = pred_.float()
        Y = Y[ROI > 0.9] > 0.5
        Y = Y.float()


        f1 = ((2 * pred_ * Y).sum() + e) / ((pred_ + Y).sum() + e)
        
        self.running_loss += [loss.item()]
        self.running_f1 += [f1.item()]
        self.runing_reg_loss += [reg_loss.item()]
        # print(pred.shape)
        return pred.float()
        # return pred[:,-1].float()

    def get_cat_f1(self):
        total = 0
        count = 0
        for i in self.confusions:
            f1 = self.confusions[i].get_f1()
            if f1 != 0:
                total += f1
                count += 1
                print(i, self.confusions[i].get_f1().item())
        return total / (count + 1e-6)
    
    
    def validate(self, X, Y, ROI, filenames): 
        self.model.eval()
        with torch.no_grad():
            pred = self.model(X)
            for j in range(pred.shape[0]):
                outputs_ = pred.argmax(dim=1)[j][(ROI[j]>0.9)]
                masks_ = Y[j][(ROI[j]>0.9)]
                assert outputs_.shape == masks_.shape
                video_name = "_".join(filenames[j].split("_")[:2])
                self.confusions[video_name].update(masks_, outputs_ > 0.5)
            loss = self.loss_fn(pred, Y, ROI)
            e = 1e-6
            # pred = torch.sigmoid(pred)
            rough_pred = pred[:,0]
            pred = pred.argmax(dim=1)
            pred_ = (pred==1)[ROI > 0.9]
            # print(pred_.shape, Y.shape)
            if self.args.hard_shadow:
                pred = torch.where(ROI > 0.9, pred, 0)
                pred = pred / 2.0 # why /2.0 0-2 -> 0-1
            else:
                pred = torch.where(ROI > 0.9, pred, 0) 
            rough_pred = torch.where(ROI > 0.9, rough_pred, 0)
            pred_ = pred_.float()
            Y = Y[ROI > 0.9] > 0.5
            Y = Y.float()
            # print(pred_.shape, Y.shape)
            
            f1 = ((2 * pred_ * Y).sum() + e) / ((pred_ + Y).sum() + e) 
            # if not self.validate_f1:
            #     sklearn_f1 = f1_score(Y.cpu().numpy(), pred_.cpu().numpy())
            #     if not np.isclose(f1.item(), sklearn_f1):
            #         print(f"\033[93m Mismatch {f1.item()}, {sklearn_f1}\033[0m")
            # self.validate_f1 = True
        return loss, f1, pred.float(), rough_pred
    
    
    def train_epoch(self):
        import tqdm
        pred = torch.zeros((1, 512, 512)).cuda()
        # self.scaler = torch.GradScaler()
        for train_i, (X, Y, ROI) in enumerate(tqdm.tqdm(self.train_dataloader, ncols=60)):
            train_pred = self.train_step(X.cuda(), Y.cuda(), ROI.cuda()) 
            self.lr_scheduler.step()
            self.scheduler_steps += 1
            
            # weight decay = args.weight_decay to args.final_weight_decay with linear increase for total steps
            
            # weight_decay = self.args.weight_decay + (self.args.final_weight_decay - self.args.weight_decay) * self.scheduler_steps / self.args.steps
            # for param_group in self.optimizer.param_groups:
            #     param_group['weight_decay'] = weight_decay
            if self.scheduler_steps == self.args.steps:
                5/0
            # print(ROI[0].max())
            
            if train_i % 100 == 0:
                weight_decay = self.optimizer.param_groups[0]["weight_decay"] * 1.001
                for param_group in self.optimizer.param_groups:
                    param_group['weight_decay'] = weight_decay
                
            if train_i % self.args.print_every == 0:
                if train_i != 0:
                    grad = self.getgrad()
                    print(f"\nMean Grad: {grad.mean()}, Max Grad: {grad.max()}, Min Grad: {grad.min()}")
                    # self.logger.log({"pstep":self.step, "grad": self.logger.Histogram(grad)})
            # if train_i % 100 == 0: 
                self.logger.log({"pstep":self.step, "loss": np.mean(self.running_loss), "reg_loss": np.mean(self.runing_reg_loss), "f1": np.mean(self.running_f1), "lr": self.optimizer.param_groups[0]["lr"]})
                printred(f"Epoch {self.step}, Step {train_i}, Loss: {np.mean(self.running_loss)}, F1: {np.mean(self.running_f1)}")
                val_runnning_loss, val_running_f1 = 0, 0
                videonames = list(set(["_".join(i.split("_")[:2]) for i in os.listdir("/mnt/fastdata/CDNet/in")]))

                self.confusions = {}
                for i in videonames:
                    self.confusions[i] = BinaryConfusion()
                    
                for val_i, (val_X, val_Y, val_ROI, filenames) in enumerate(tqdm.tqdm(self.val_dataloader, ncols=60)):
                    # print(val_ROI[0].max())
                    val_loss, val_f1, val_pred, rough_pred = self.validate(val_X.cuda(), val_Y.cuda(), val_ROI.cuda(), filenames)
                    # print(val_f1
                    val_runnning_loss += val_loss
                    val_running_f1 += val_f1
                # print(ROI[0].max(), val_ROI[0].max())
                print(f"Train pred Range: {train_pred.min()}, {train_pred.max()}")
                print(Y[-batch_size].unsqueeze(0).shape, val_Y[0].unsqueeze(0).shape)
                self.logger.log({
                                "pstep":self.step,
                                "weight_decay":self.optimizer.param_groups[0]["weight_decay"], 
                                "val_pred": self.logger.Image(val_pred[0].unsqueeze(0)),
                                "val_gt": self.logger.Image(val_Y[0].unsqueeze(0)),
                                "val_in": self.logger.Image(val_X[0][:3]),
                                "val_roi": self.logger.Image((val_ROI[0].unsqueeze(0) * 255).to(torch.uint8)),
                                # "val_BG1": self.logger.Image(val_X[0][-24:-27]),
                                # "val_BG2": self.logger.Image(val_X[0][-27:]),
                                "train_pred": self.logger.Image(train_pred[0].unsqueeze(0)),
                                "train_gt": self.logger.Image(Y[-batch_size].unsqueeze(0)),
                                "train_in": self.logger.Image(X[-batch_size][:3]),
                                "train_roi": self.logger.Image((ROI[-batch_size].unsqueeze(0) * 255).to(torch.uint8)),
                                # "train_BG1": self.logger.Image(X[-batch_size][-24:-27]),
                                # "train_BG2": self.logger.Image(X[-batch_size][-27:]),
                                "rough_pred": self.logger.Image(rough_pred[0].unsqueeze(0)),
                                "cat_f1": self.get_cat_f1()
                }) 
                
                self.logger.log({"pstep":self.step, "val_loss": val_runnning_loss / len(self.val_dataloader), "val_f1": val_running_f1 / len(self.val_dataloader)})
                printgreen(f"Validation Loss: {val_runnning_loss / len(self.val_dataloader)}, Validation F1: {val_running_f1 / len(self.val_dataloader)}")

                self.running_loss = []
                self.running_f1 = []
                self.runing_reg_loss = []
                # self.lr_scheduler.step(val_running_f1 / len(self.val_dataloader))
                self.step += 1
                if self.step >= 1000:
                    raise StopIteration
                torch.save(self.model.state_dict(), f"{self.args.save_name}_{self.step}.pth")

    def train(self):
        self.scheduler_steps = 0
        while True:
            try:
                self.train_epoch()
            except ZeroDivisionError:
                break
