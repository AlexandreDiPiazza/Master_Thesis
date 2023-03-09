import pytorch_lightning as pl
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning.loggers import WandbLogger
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
import yaml

from dataPreparation import Dataset
from Models.PointCloud.utils import take_all_coordsV2, take_all_coords
from Trainer import Classifier



def train_model(config, weight,  _data):
    # Model Checkpoints
    wandb_logger = WandbLogger(name=config['dataset']['run_name'], project=config['dataset']['project_name'])
    checkpoint_callback = ModelCheckpoint(
        dirpath= config['dataset']['ckpt_save'] + config['dataset']['run_name'],
        every_n_epochs=10,
        save_top_k=-1
    )
    PC_params = {'translation': False, 'scaling' : False, 'jittering': False}
    model = Classifier(weight=weight, batch_size=config['training']['batch'], data_class=_data, val_loader=None, coords = config['dataset']['nbr_coords'],
                       channels = config['dataset']['channels'], pointCloud_bool = config['dataset']['pointCloud_bool'] , model_type = config['model']['type'],
                       PC_params = PC_params)
    model.lr = config['training']['lr']

    trainer = pl.Trainer(max_epochs=config['training']['epochs'], devices=4, accelerator='gpu',
                         accumulate_grad_batches=config['training']['accumulate_grad_batches'],  # do as if the batch_size was 4*4
                         logger=wandb_logger, callbacks=[checkpoint_callback],
                         fast_dev_run=False,
                         auto_lr_find=False,
                         reload_dataloaders_every_n_epochs=1)  # ,
    trainer.fit(model, train_dataloaders=None, val_dataloaders=None)  # we don't need the loaders, they are already define in Classifier
    
def fastRun(config, val_loader, weight, _data):
    PC_params = {'translation': False, 'scaling' : False, 'jittering': True}
    model = Classifier(weight=weight, batch_size=config['training']['batch'], data_class=_data, val_loader=val_loader, coords = config['dataset']['nbr_coords'],
                       channels = config['dataset']['channels'], pointCloud_bool = config['dataset']['pointCloud_bool'] , model_type = config['model']['type'],
                       PC_params = PC_params)
    model.lr = config['training']['lr']
    trainer = pl.Trainer(devices=1, accelerator='gpu',accumulate_grad_batches=2, fast_dev_run=True)
    trainer.fit(model, train_dataloaders=None, val_dataloaders=None)  # we don't need the loaders, they are already define in Classifier


def findLR(config, val_loader, weight, _data):
    from ignite.handlers import FastaiLRFinder
    nbr_coords = config['dataset']['nbr_coords'] # for PC
    trainer = pl.Trainer(devices=1, accelerator='gpu')
    PC_params = {'translation': True, 'scaling' : True, 'jittering': True}
    model = Classifier(weight=weight, batch_size=config['training']['batch'], data_class=_data, val_loader=None, coords = nbr_coords,
                       channels = config['dataset']['channels'], pointCloud_bool = config['dataset']['pointCloud_bool'] , model_type = config['model']['type'],
                       PC_params = PC_params)

    lr_finder = trainer.tuner.lr_find(model, num_training=5000)
    lr_finder.results
    lr = lr_finder.suggestion()
    print(f'Auto-find model LR: {lr}')

    # Plot with
    fig = lr_finder.plot(suggest=True)
    fig.show()

def valSplit(X,y):
    X_train, X_val, y_train, y_val = train_test_split(X, y,stratify=y,test_size=0.1, random_state=42)
    return torch.from_numpy(X_train), torch.from_numpy(X_val), torch.from_numpy(y_train), torch.from_numpy(y_val)

if __name__ == '__main__':
    pl.seed_everything(42)
    config_path = "./Models/PointCloud/config.yaml"

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    print('')
    path = config['dataset']['path_train']
    print('Model: {}, data: {}, channels: {}, Surface: {}'.format(config['model']['type'], path, config['dataset']['channels'], config['dataset']['surface']))
    ### PARAMETERS OF THE CONFIG
    channels = config['dataset']['channels'] # nbr of channels in the input, 1 or 2 
    pointCloud_bool = config['dataset']['pointCloud_bool']  # if we do PointCloud or no, it goes with channels = 2
    nbr_coords = config['dataset']['nbr_coords'] # for PC
    data_spe = (config['dataset']['aug_0'],config['dataset']['aug_1'])
    surface =  config['dataset']['surface']
    ### LOADING THE DATA 
    if surface == False:
        if config['dataset']['big'] == False: 
            print('Loading the data, entire volume, small')
            X = np.load(path + 'X_train.npy')#[:,:,:,75-45:75+45,75-45:75+45]
            y = np.load(path + 'Y_train.npy')
        else: 
            print('Loading the data, entire volume, big')
            X = np.load(path + 'X_trainBIG.npy')
            y = np.load(path + 'Y_trainBIG.npy')
    else:
        if config['dataset']['big'] == False: 
            print('Loading the data, only surface, small')
            X = np.load(path + 'X_train_surface.npy')
            y = np.load(path + 'Y_train_surface.npy')
        else:
            print('Loading the data, only surface, big')
            X = np.load(path + 'X_train_surfaceBIG.npy')
            y = np.load(path + 'Y_train_surfaceBIG.npy')
            
    print('X_raw shape: {} , y_raw shape :{}'.format(np.shape(X), np.shape(y)))
    aug_pair = (data_spe[0], data_spe[1])
    # Train-Val Split: 
    X_train_, X_val_, y_train_, y_val_ =  valSplit(X, y)
    _data = Dataset(X_train_, y_train_, X_val_, y_val_, aug_0=aug_pair[0], aug_1=aug_pair[1], channels = channels, pointCloud_bool = pointCloud_bool)
    
    X_val, y_val = _data.prepareValData(pointCloud_bool)
    # Load the Training Data even if useless, just to compute the weight and print the statistics to make sure no error 
    X_train, y_train = _data.augmentDataTrain()
    weight = torch.sum(y_train == 0) / torch.sum(y_train == 1)  # Weight bc inbalanced class
    print('_')
    print('TRAIN: Shape: {}, nbr of zeros: {}, nbrs of ones: {}'.format(X_train.shape, torch.sum(y_train == 0), torch.sum(y_train==1)))
    print('Weight: {}'.format(weight))
    print('VAL: Shape: {}, nbr of zeros: {}, nbrs of ones: {}'.format(X_val.shape, torch.sum(y_val == 0), torch.sum(y_val==1)))

    
    
    
    # At this point the data is still a full 3D image, we now turn take only the coordinates
    if pointCloud_bool == True:
        PC_params = {'translation': False, 'scaling' : False, 'jittering': False}
        X_val_fin, y_val_fin = take_all_coordsV2(np.array(X_val), np.array(y_val),nbr_coords, up_sample0 = 20, up_sample1 = 20, PC_params = PC_params)
        #X_val_fin2 = take_all_coords(np.array(X_val), nbr_coords)
     
        X_val_fin =  X_val_fin.type(torch.float32)
        y_val_fin =  y_val_fin.type(torch.float32)
        print('Point Cloud, final Validation shape: {}, 1s:{} 0s:{}'.format(X_val_fin.shape, (y_val_fin.eq(1).sum().item()), (y_val_fin.eq(0).sum().item()) ))
        print(np.sum(np.array(y_val_fin)==1))
        Val = TensorDataset(X_val_fin, y_val_fin) # Train dataset is defined during training bc it changes every epoch
    else: 
        Val = TensorDataset(X_val, y_val)
   
    val_loader = DataLoader(Val, batch_size=config['training']['batch'], shuffle=False)
    del X; del y; del X_train; del X_val; del y_train; del y_val #free memory
    
    train_model(config , weight, _data)





