import torch
import numpy as np

class trainer:
    def __init__(self, model, optimizer, lr_scheduler, train_dataloader, val_dataloader, logger, loss_fn):
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.logger = logger
        self.loss_fn = loss_fn
        self.running_loss = []
        self.running_f1 = []
        self.step = 0

    
    def train_step(self, X, Y, ROI):
        self.model.train()
        self.optimizer.zero_grad()
        pred = self.model(X).squeeze(1)
        loss = self.loss_fn(pred, Y, ROI)
        loss.backward()
        self.optimizer.step()
        e = 1e-6
        # pred = torch.sigmoid(pred)
        pred = pred[ROI > 0.9] > 0.5
        pred = pred.float()
        Y = Y[ROI > 0.9] > 0.5
        Y = Y.float()
        f1 = ((2 * pred * Y).sum() + e) / ((pred + Y).sum() + e)
        
        self.running_loss += [loss.item()]
        self.running_f1 += [f1.item()]

    def validate(self, X, Y, ROI):
        self.model.eval()
        with torch.no_grad():
            pred = self.model(X).squeeze(1)
            loss = self.loss_fn(pred, Y, ROI)
            e = 1e-6
            # pred = torch.sigmoid(pred)
            pred_ = pred[ROI > 0.9] > 0.5
            pred_ = pred_.float()
            Y = Y[ROI > 0.9] > 0.5
            Y = Y.float()
            f1 = ((2 * pred_ * Y).sum() + e) / ((pred_ + Y).sum() + e)
        return loss, f1, pred.float()
    
    
    def train_epoch(self):
        import tqdm
        for train_i, (X, Y, ROI) in enumerate(tqdm.tqdm(self.train_dataloader, ncols=60)):
            self.train_step(X.cuda(), Y.cuda(), ROI.cuda())
            if train_i%100 == 0: 
                self.logger.log({"pstep":self.step,"loss": np.mean(self.running_loss), "f1": np.mean(self.running_f1)})
                print(f"\n Epoch {self.step}, Step {train_i}, Loss: {np.mean(self.running_loss)}, F1: {np.mean(self.running_f1)}")
                val_runnning_loss, val_running_f1 = 0, 0
                for val_i, (val_X, val_Y, val_ROI) in enumerate(self.val_dataloader):
                    val_loss, val_f1, pred = self.validate(val_X.cuda(), val_Y.cuda(), val_ROI.cuda())
                    val_runnning_loss += val_loss
                    val_running_f1 += val_f1
                self.logger.log({"pstep":self.step,
                                 "pred": self.logger.Image(pred.unsqueeze(1)),
                                 "gt": self.logger.Image(val_Y.unsqueeze(1)),
                                 "in": self.logger.Image(val_X[:,:3]),
                                 "roi": self.logger.Image(val_ROI.unsqueeze(1)),
                                 "BG1": self.logger.Image(val_X[:,3:6]),
                                    "BG2": self.logger.Image(val_X[:,6:9])
                })
                
                self.logger.log({"pstep":self.step, "val_loss": val_runnning_loss / len(self.val_dataloader), "val_f1": val_running_f1 / len(self.val_dataloader)})
                print(f"Validation Loss: {val_runnning_loss / len(self.val_dataloader)}, Validation F1: {val_running_f1 / len(self.val_dataloader)}")

                self.running_loss = []
                self.running_f1 = []
                self.lr_scheduler.step()
                self.step += 1
                if self.step >= 1000:
                    raise StopIteration
                torch.save(self.model.state_dict(), f"model.pth")

    def train(self):
        while True:
            try:
                self.train_epoch()
            except StopIteration:
                break
