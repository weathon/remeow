import torch
import numpy as np

def printred(text):
    print(f"\033[31m{text}\033[0m")

def printgreen(text):
    print(f"\033[32m{text}\033[0m")
batch_size = 8
gradient_accumulation = False
REFINE = True
from sklearn.metrics import f1_score
class Trainer:
    def __init__(self, student, teacher, optimizer, lr_scheduler, train_dataloader, val_dataloader, logger, loss_fn, args):
        self.student = student.to("cuda:0")
        self.teacher = teacher.to("cuda:1")
        
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.logger = logger
        self.loss_fn = loss_fn
        self.running_loss = []
        self.running_f1 = []
        self.step = 0
        self.validate_f1 = False
        self.args = args

    def getgrad(self):
        grads = [] #it was just [0] 
        for param in self.model.parameters():
            if param.grad is not None: #why need this
                grads.append(param.grad.view(-1).cpu())
        grads = torch.cat(grads) 
        return grads 
    
    def update_teacher(self):
        teacher_dict = self.teacher.state_dict()    
        student_dict = self.student.state_dict()
        for k in teacher_dict.keys():
            teacher_dict[k] = student_dict[k] * self.args.m + teacher_dict[k] * (1 - self.args.m)
    
    def train_step(self, X_student, X_teacher, Y,ROI):  
        X_student = X_student.to("cuda:0")
        X_teacher = X_teacher.to("cuda:1") 
             
        self.student.train()
        self.teacher.eval()
        
        with torch.no_grad():
            teacher_pred = torch.sigmoid(self.teacher(X_teacher) / self.args.temp)
        
        self.optimizer.zero_grad()
    
        pred = self.model(X_student.to("cuda:0")) 
        loss_teacher_student = self.loss_fn(pred, teacher_pred.to("cuda:0"), ROI)
        loss_gt_student = self.loss_fn(pred, Y.to("cuda:0"), ROI)
        loss = loss_teacher_student + loss_gt_student
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0) 
        self.optimizer.step()
        e = 1e-6
        pred_ = pred[:,-1][ROI > 0.9] > 0.5
        pred = torch.where(ROI > 0.9, pred, 0)
        pred_ = pred_.float()
        Y = Y[ROI > 0.9] > 0.5 
        Y = Y.float()
        
        f1 = ((2 * pred_ * Y).sum() + e) / ((pred_ + Y).sum() + e)
        
        self.running_loss += [loss.item()]
        self.running_f1 += [f1.item()]
        return pred[:,-1].float()

    def validate(self, X, Y, ROI): 
        self.model.eval()
        with torch.no_grad():
            pred = self.model(X)
            loss = self.loss_fn(pred, Y, ROI)
            e = 1e-6
            # pred = torch.sigmoid(pred)
            rough_pred = pred[:,0]
            pred = pred[:,-1]
            pred_ = pred[ROI > 0.9] > 0.5
            pred = torch.where(ROI > 0.9, pred, 0) 
            rough_pred = torch.where(ROI > 0.9, rough_pred, 0)
            pred_ = pred_.float()
            Y = Y[ROI > 0.9] > 0.5
            Y = Y.float()
            f1 = ((2 * pred_ * Y).sum() + e) / ((pred_ + Y).sum() + e)
           

        return loss, f1, pred.float(), rough_pred
    
    
    def train_epoch(self):
        import tqdm
        pred = torch.zeros((1, 512, 512)).cuda()
        for train_i, (X_student, X_teacher, Y, ROI) in enumerate(tqdm.tqdm(self.train_dataloader, ncols=60)):
            train_pred = self.train_step(X_student, X_teacher, Y, ROI)
            self.lr_scheduler.step()
            self.scheduler_steps += 1
            if self.scheduler_steps == self.args.steps:
                5/0
                
            if train_i % 500 == 0:
                if train_i != 0:
                    grad = self.getgrad()
                    print(f"\nMean Grad: {grad.mean()}, Max Grad: {grad.max()}, Min Grad: {grad.min()}")

                weight_decay = self.optimizer.param_groups[0]["weight_decay"] * 1.02
                for param_group in self.optimizer.param_groups:
                    param_group['weight_decay'] = weight_decay
                self.logger.log({"pstep":self.step,"loss": np.mean(self.running_loss), "f1": np.mean(self.running_f1), "lr": self.optimizer.param_groups[0]["lr"]})
                printred(f"Epoch {self.step}, Step {train_i}, Loss: {np.mean(self.running_loss)}, F1: {np.mean(self.running_f1)}")
                val_runnning_loss, val_running_f1 = 0, 0
                for val_i, (val_X, _, val_Y, val_ROI) in enumerate(tqdm.tqdm(self.val_dataloader, ncols=60)):
                    val_loss, val_f1, val_pred, rough_pred = self.validate(val_X.cuda(), val_Y.cuda(), val_ROI.cuda())
                    val_runnning_loss += val_loss
                    val_running_f1 += val_f1
                print(f"Train pred Range: {train_pred.min()}, {train_pred.max()}")
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
                }) 
                
                self.logger.log({"pstep":self.step, "val_loss": val_runnning_loss / len(self.val_dataloader), "val_f1": val_running_f1 / len(self.val_dataloader)})
                printgreen(f"Validation Loss: {val_runnning_loss / len(self.val_dataloader)}, Validation F1: {val_running_f1 / len(self.val_dataloader)}")

                self.running_loss = []
                self.running_f1 = []
                # self.lr_scheduler.step(val_running_f1 / len(self.val_dataloader))
                self.step += 1
                if self.step >= 1000:
                    raise StopIteration
                torch.save(self.model.state_dict(), f"model.pth")

    def train(self):
        self.scheduler_steps = 0
        while True:
            try:
                self.train_epoch()
            except ZeroDivisionError:
                break
