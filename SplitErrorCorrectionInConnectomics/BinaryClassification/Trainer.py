import pytorch_lightning as pl
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
import numpy as np 
import torch
from torch.utils.data import DataLoader, TensorDataset
import torchmetrics

from Models.CNN.model_CNN import CNN
from Models.PointCloud.model_point_cloud import PointNetCls
from Models.PointCloud.utils import take_all_coordsV2
from Models.Siamese.SiameseCNN import SiameseCNN
from Models.ResNet.ResNet import ResNet
from Models.LSTM_CNN.LSTM_CNN import TimmModel

class Classifier(pl.LightningModule):
    def __init__(self, weight, batch_size, data_class, val_loader, coords = 1000, channels = 1, pointCloud_bool = True , model_type = 'PointCloud', PC_params = None):
        super().__init__()
        self.save_hyperparameters()
        self.model_type = model_type
        if model_type == 'CNN':
            self.model = CNN(channels)
        elif model_type == 'PointCloud':
            self.model = PointNetCls(k=1)
        elif model_type == 'Siamese':
            self.model = SiameseCNN()
        elif model_type == 'LSTM_CNN':
            self.model = TimmModel(backbone = 'tf_efficientnetv2_s_in21ft1k', n_slice_per_c = 18, in_chans= 1, image_size = 150)
            
        self.lr = 1e-4
        self.acc = torchmetrics.classification.BinaryAccuracy(threshold=0.5)
        self.recall = torchmetrics.classification.BinaryRecall(threshold=0.5)
        self.pre = torchmetrics.classification.BinaryPrecision(threshold=0.5)
        #weihts is the weights of the 0 class. We put a smaller weights for 0 class as it is less present
        self.weight = weight
        self.batch_size = batch_size
        self.data_class = data_class # class to load dataloder at each iter
        self.val_loader = val_loader
        self.train_loader = None
        self.coords = coords
        self.pointCloud_bool = pointCloud_bool
        self.PC_params = PC_params
        
    def train_dataloader(self):
        del self.train_loader
        X_training, y_training = self.data_class.augmentDataTrain() # Create new augmentations
        if self.pointCloud_bool  == True :
            X_training_coords, Y_training_coords = take_all_coordsV2(np.array(X_training), np.array(y_training),self.coords, up_sample0 = 5, up_sample1 = 25,
                                                                     PC_params = self.PC_params)
            #print(X_training_coords.shape)
            #For Point Cloud, we compute the weight for the binary Classification here, because we do the upsampling here.
            self.weight = torch.sum(Y_training_coords==1)/(torch.sum(Y_training_coords==0))
            X_training_coords =  X_training_coords.type(torch.float32)
            Y_training_coords =  Y_training_coords.type(torch.float32)
            del X_training
            self.train_loader = DataLoader(TensorDataset(X_training_coords, Y_training_coords), batch_size = self.batch_size, shuffle=True, num_workers = 4)
            del X_training_coords
        else: 
            self.train_loader = DataLoader(TensorDataset(X_training, y_training), batch_size = self.batch_size, shuffle=True, num_workers = 4)
            del X_training
        del y_training
        return self.train_loader
        
    def val_dataloader(self):
        del self.val_loader
        X_val, y_val = self.data_class.prepareValData(self.pointCloud_bool)
        if self.pointCloud_bool  == True :
            X_val_coords, Y_val_coords = take_all_coordsV2(np.array(X_val), np.array(y_val), self.coords, up_sample0 = 30, up_sample1 = 30,
                                                           PC_params= self.PC_params)
            X_val_coords =  X_val_coords.type(torch.float32)
            Y_val_coords =  Y_val_coords.type(torch.float32)
            del X_val
            self.val_loader = DataLoader(TensorDataset(X_val_coords, Y_val_coords), batch_size = self.batch_size, shuffle=True, num_workers = 4)
            del X_val_coords
        else: 
            self.val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size = self.batch_size, shuffle=True, num_workers = 4)
            del X_val
        del y_val
        return self.val_loader
        
    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        #lr_scheduler = optim.lr_scheduler.MultiStepLR(
        #    optimizer, milestones=[50, 100], gamma = 0.1)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                T_max=400, eta_min=1e-7, last_epoch=- 1, verbose=False)

        return [optimizer],[lr_scheduler]
        #return optimizer
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self.model(x)
        loss = F.binary_cross_entropy(preds, y, reduction = 'none')
        weights = self.compute_weights(y)
        loss = loss * weights # Apply the weights to the Loss (unbalanced class)
        loss = loss.mean() 
        self.log("train/loss", loss, prog_bar=True, on_epoch = True, sync_dist=True)
        self.log("train/Acc", self.acc(preds,y), on_epoch = True, sync_dist=True)
        self.log("train/Rec", self.recall(preds,y), on_epoch = True, sync_dist=True)
        self.log("train/Pre", self.pre(preds,y), on_epoch = True, sync_dist=True)
        return loss
    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self.model(x)
        loss = F.binary_cross_entropy(preds, y, reduction = 'none')
        weights = self.compute_weights(y)
        loss = loss * weights # Apply the weights to the Loss (unbalanced class)
        loss = loss.mean() 
        self.log("Val/loss", loss, prog_bar=True, on_epoch = True,sync_dist=True)
        self.log("Val/Acc", self.acc(preds,y), on_epoch = True,sync_dist=True)
        self.log("Val/Rec", self.recall(preds,y), on_epoch = True,sync_dist=True)
        self.log("Val/Pre", self.pre(preds,y), on_epoch = True,sync_dist=True)
        return loss
    def compute_weights(self, targets):
        #Compute the weights for the BCE
        weights = torch.ones(size  = (len(targets),1))
        for i in range(len(targets)):
            if targets[i]==1:
                weights[i] = self.weight
        return weights.to('cuda')
    def test_step(self, batch, batch_idx):
        x, y = batch
        preds = self.model(x) 
        return preds


