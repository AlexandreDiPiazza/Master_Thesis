import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
import itertools
from torch import nn
import pytorch_lightning as pl
import yaml
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.spatial.transform import Rotation
from sklearn.metrics import precision_recall_curve
import time 
from ptflops import get_model_complexity_info

from Trainer import Classifier
from dataPreparation import Dataset
from Models.PointCloud.utils import take_coords
from sklearn.model_selection import train_test_split

def to2Channels(tensor: torch.Tensor, ax: int) -> torch.Tensor:
    """
    The input image has 1 channel with 0,1,2 ( 1 for first object, 2 for second object)
    We return a 2 channels image with 1 and 0.
    :param tensor: (1,Z,Y,Z) size tensor, containing 0 for background, and 1 and 2 for the two objects
           ax: if 0, put 1 into first axis, and 2 in the other.
               if 1, put 2 into first axis, and 1 in the other.
    :return:
    a1: tensor shape (1,Z,Y,Z), with 0 for background and 1  for the first object
    a2: tensor shape (1,Z,Y,Z), with 0 for background and 1 for the second object.
    """
    array = np.array(torch.clone(tensor))
    if ax == 0:  # We choose a1 for label 1, we swap so that not always same object get the same label
        a1 = array * (array == 1).copy()  # Contains all 1 indices of the array
        a2 = (array * (array == 2) / 2).copy()  # Contains all the 2 of the array
    elif ax == 1:
        a1 = (array * (array == 2) / 2).copy()  # Contains all the 1 of the array
        a2 = array * (array == 1).copy()  # Contains all the 2 of the array
    return torch.from_numpy(a1).type(torch.int8), torch.from_numpy(a2).type(torch.int8)
    

def toValues(tensor: torch.Tensor, ax: int) -> torch.Tensor:
    """
    The input image has 1 channel with 0,1,2 ( 1 for first object, 2 for second object)
    We return a 2 channels image with 1 and 0.
    :param tensor: (1,Z,Y,Z) size tensor, containing 0 for background, and 1 and 2 for the two objects
            ax: if 0, put 1 for first object, -1 for second object
                if 1, put -1 for first object, 1 for second object
    :return:
        tensor shape (1,Z,Y,Z), with 0 for background, and -1 or 1 for each object
    """
    array = np.array(torch.clone(tensor)).astype(np.int8)

    if ax == 0: # 1 for the first object, -1 for the second object
        array[array==2] = -1
    elif ax == 1: # -1 for the first object, 1 for the second object
        array[array==1] = -1
        array[array==2] = 1
    return torch.from_numpy(array).type(torch.int8)


def normalize(tensor:torch.tensor) -> torch.tensor:
    """
    normalize the coordinates between 0 and 1, we dont touch the last line which is a label
    """
    normalized_tensor  = tensor.clone()
    
    # Normalize the first 3 rows
    for i in range(3):
        min_val = normalized_tensor[i].min()
        max_val = normalized_tensor[i].max()
        normalized_tensor[i] = (normalized_tensor[i] - min_val) / (max_val - min_val)
    return normalized_tensor
    

def rotate_tensor(coordinates: torch.tensor, nbr_coorods) -> torch.tensor: 
    #print(coordinates.shape)
    # Generate three random rotation angles
    alpha = torch.rand(1) * 2 * np.pi
    #alpha = 0*torch.rand(1) # no rotation around z 
    beta = torch.rand(1) * 2 * np.pi
    gamma = torch.rand(1) * 2 * np.pi
    #gamma = 0*torch.rand(1)
    #print('Angle: ', alpha, beta, gamma)
    
    # Define the rotation matrices around each of the axes
    Rx = torch.tensor([[1, 0, 0],
                       [0, torch.cos(alpha), -torch.sin(alpha)],
                       [0, torch.sin(alpha), torch.cos(alpha)]])
    
    Ry = torch.tensor([[torch.cos(beta), 0, torch.sin(beta)],
                       [0, 1, 0],
                       [-torch.sin(beta), 0, torch.cos(beta)]])
    
    Rz = torch.tensor([[torch.cos(gamma), -torch.sin(gamma), 0],
                       [torch.sin(gamma), torch.cos(gamma), 0],
                       [0, 0, 1]])
    
    # Combine the rotation matrices to get the overall rotation matrix
    R = torch.mm(torch.mm(Rz, Ry), Rx).to('cuda')
   
    # Apply the rotation to all the coordinates
    rotation_center = torch.tensor([0.5,0.5,0.5]).unsqueeze(1).to('cuda')
    rotated_coordinates = torch.mm(R, coordinates-rotation_center) + rotation_center
    return torch.clamp(rotated_coordinates, 0, 1)
    
def flip(point_cloud, permut):
    #print('doing the flips')
    #print(permut)
    z_flip = 1 ; y_flip = 1 ; x_flip = 1;
    if permut[0] == 1:
        z_flip = -1
    if permut[1] == 1:
        y_flip = -1
    if permut[2] == 1:
        x_flip = -1
        
    rotation_center = torch.tensor([0.5,0.5,0.5]).unsqueeze(1).to('cuda')
    flip_axes = torch.tensor([z_flip, y_flip, x_flip]).float().to(point_cloud.device)
    flipped_point_cloud = ((point_cloud-rotation_center) * flip_axes[:, None]) + rotation_center
    return torch.clamp(flipped_point_cloud, 0,1)
    
def scale_tensor(coordinates: torch.tensor) -> torch.tensor:
    scale_factors = torch.rand(3) * 0.1 + 0.95
    scaled_coordinates = coordinates * scale_factors[:, None].to('cuda')
    return torch.clamp(scaled_coordinates, 0, 1)

def translate_tensor(coordinates: torch.tensor, ax) -> torch.tensor:
    translation = (torch.rand(3) - 0.5) * 0.1
    if ax < 3: #no translation
        translation[0] = 0
        translation[1] = 0
        translation[2] = 0
    elif ax == 3:
        translation[1] = 0
        translation[2] = 0
    elif ax == 4:
        translation[0] = 0
        translation[2] = 0
    elif ax == 5:
        translation[0] = 0
        translation[1] = 0
    translated_coordinates = coordinates + translation[:, None].to('cuda')
    return torch.clamp(translated_coordinates, 0, 1)
    
def jitter_tensor(coordinates: torch.tensor) -> torch.tensor:
    jitter = (torch.rand(*coordinates.shape) - 0.5) * 0.001
    jittered_coordinates = coordinates + jitter.to('cuda')
    return torch.clamp(jittered_coordinates, 0, 1)


def test_transform(ori_image: torch.tensor, permut)->torch.tensor:
    """"
    :param ori_image: image before transformation, shape 1,Z,Y,X
           permut: permuations to perform, list of ints
    :return: tensor, image after transformation, shape 1,Z,Y,X
    """
    #print('PERMUTATION', permut)
    #print('#')
    my_transforms1 = transforms.Compose([
        # SYMMETRIE PAR RAPPORT AU PLAN Z,Y. (valeurs z,y change pas, mais changement pour les x)
        transforms.RandomHorizontalFlip(p=1)])
    my_transforms2 = transforms.Compose([
        # SYMMETRIE PAR RAPPORT AU PLAN Z,X. (valeurs z,x change pas, mais changement pour les y)
        transforms.RandomVerticalFlip(p=1)])
    if permut[0] == 1:
        ori_image = my_transforms1(ori_image)
    if permut[1] == 1:
        ori_image = my_transforms2(ori_image)
    if permut[2] == 1:
        ori_image = transforms.functional.rotate(img=ori_image, angle=90)
    if permut[3] == 1:
        ori_image = torch.from_numpy(np.array(ori_image)[:, ::-1, :, :].copy())  # ZFlip
    #print(torch.mean(ori_image))
    return ori_image
    
def parameters_of_model(model):
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params}")
    flops, params = get_model_complexity_info(model, (2,18,150,150), as_strings=True, print_per_layer_stat=True)
    print(f"Number of FLOPs in the forward pass: {flops}")
    

if __name__ == '__main__':
    print('We set a seed bc we take a random N nbr of poins of coordinates in the Point Cloud')
    pl.seed_everything(3)
    config_path = "./Models/PointCloud/config.yaml"
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    print('')
    print(config)
    
    path = config['dataset']['path_test']
    surface =  config['dataset']['surface']
    pointCloud_bool = config['dataset']['pointCloud_bool'] 
    channels = config['dataset']['channels']
    nbr_coords = config['dataset']['nbr_coords']
    ### LOADING THE DATA 
    if surface == False:
        if config['dataset']['big'] == False: 
            print('Loading the data, entire volume, Small')
            X = np.load(path + 'X_test.npy')
            y = np.load(path + 'Y_test.npy')
        else: 
            print('Loading the data, entire volume, Big')
            X = np.load(path + 'X_testBIG.npy')
            y = np.load(path + 'Y_testBIG.npy')
    else:
        if config['dataset']['big'] == False: 
            print('Loading the data, only surface')
            X = np.load(path + 'X_test_surface.npy')#[:,:,:,:,:]
            y = np.load(path + 'Y_test_surface.npy')#[0:20]
        else: 
            print('Loading the data, only surface, bIG')
            X = np.load(path + 'X_test_surfaceBIG.npy')
            y = np.load(path + 'Y_test_surfaceBIG.npy')
   
    
    Z_size, Y_size, X_size = np.shape(X[0, 0, :, :, :])
    print('N', len(y))
    print('0s', np.sum(y == 0))
 
    
    #_data = Dataset(X, y,None,None,None,None, channels = channels, pointCloud_bool = pointCloud_bool)
    X_test = torch.from_numpy(X).type(torch.float32)
    y_test = torch.reshape(torch.tensor(y), shape=(len(y), 1)).type(torch.float32)
    print(np.shape(y_test))
    print(np.shape(X_test))
    print(np.unique(X_test))
    
    
    Test = TensorDataset(X_test, y_test)
    test_loader = DataLoader(Test, batch_size=1, shuffle=False, num_workers=4)
    
    model_type = config['model']['type']
    
    # just initialize the model without parameters
    PC_params = {'translation': None, 'scaling' :None, 'jittering': None}
    cls = Classifier(weight=None, batch_size=None, data_class= None, val_loader=None, coords = None,
                       channels = channels, pointCloud_bool = config['dataset']['pointCloud_bool'] , model_type = model_type, 
                       PC_params = PC_params)
    
    ckpt_path = config['dataset']['ckpt_save'] + config['dataset']['run_name']
    print('CheckPoints Path: {}'.format(ckpt_path))
    
    
    epoch_nbr   = '/epoch=149-step=3750.ckpt'

    
    
    model = cls.load_from_checkpoint(ckpt_path + epoch_nbr).model.to('cuda')
    
    type_ = 'SNEMI3D'
    if type_ == 'SNEMI3D':
        candidates_df  = pd.read_csv('./datasets/SNEMI3D/Test/df_test.csv')
    elif type_ == 'MouseCortex':
        candidates_df = pd.read_csv('./datasets/MouseCortex/Test/df_test.csv')
        
    IDs1 = list(candidates_df['ID1-Seg'])
    IDs2 = list(candidates_df['ID2-Seg'])
    count = 0
    treshold = config['testing']['treshold']
    treshold = 0.8
    print('Cheking treshold: {}'.format(treshold))
    TP = 0;
    FP = 0
    TN = 0;
    FN = 0
    results = []
    model.eval()
    print('IF we are in test mode it should show False: ', model.training)
    if pointCloud_bool == True:
        input_ = torch.zeros(size=(1, 4, 2*config['dataset']['nbr_coords']), dtype=torch.float32).to('cuda')
        all_permutations =  ["".join(seq) for seq in itertools.product("01", repeat=3)] # create the permuations of 0,1
    else:
        input_ = torch.zeros(size = (1,channels,18,150,150)).to('cuda') # dimmensions of the cube
        all_permutations = ["".join(seq) for seq in itertools.product("01", repeat=4)] # create the permuations of 0,1
        
    all_y = np.zeros(shape=(len(test_loader),))
    all_preds = np.zeros(shape = (len(test_loader),1))
    rotation_PC = True
    del y 
    with torch.no_grad():
        counting = 0
        for i, data in enumerate(test_loader):
            preds = [] # the ensemble predictions for this ID
            x, y = data
            y = y.to('cuda')
        
            if  pointCloud_bool == True: 
                im_ori = torch.zeros(size=(4,2 * nbr_coords)).to('cuda')
                
                coords1 = torch.from_numpy(take_coords(np.array(x[0,0,:,:,:]), value = 1, nbr_coords = nbr_coords)).to('cuda')
                coords2 = torch.from_numpy(take_coords(np.array(x[0,0,:,:,:]), value = 2, nbr_coords = nbr_coords)).to('cuda')
                im_ori[0:3, :nbr_coords] = coords1
                im_ori[3, : nbr_coords] = 0
                im_ori[0:3, nbr_coords:] = coords2
                im_ori[3,nbr_coords:] = 1
                im_ori = normalize(im_ori)
                for permutation in all_permutations:  #3 permutations for the flips in all directions
                    perm = list(map(int, permutation))
                    im_trans = torch.clone(im_ori)
                    coords_trans = im_trans[0:3,:]
                    coords_labels_trans = im_trans[3,:]
                
                    perm = list(map(int, permutation))
                    coords_tran = flip(coords_trans, perm)
                    for R in range(10) : # nbr of rotations per perm
                        if R == 0 : # no rotation for the first one
                            input_[0,0:3,:] = coords_trans
                            input_[0,3,:]   = coords_labels_trans
                            input_ = input_.type(torch.float32)
                            output = model.forward(input_)
                            preds.append(output.item())
                        
                        else: 
                            coords_trans = rotate_tensor(coords_trans, nbr_coords) # rotation
                            input_[0,0:3,:] = coords_trans
                            input_[0,3,:]   = coords_labels_trans
                            output = model.forward(input_.type(torch.float32))
                            preds.append(output.item())
                      
                continue
                #Second_input (reversed_oreder)
                im_ori[0:3, :nbr_coords] = coords2  # fill with coordinates obj1
                im_ori[3, :nbr_coords] = 0  # Add dimmension, fill with 0 for first object
                im_ori[0:3, nbr_coords:] = coords1  # fill with coordinates obj2
                im_ori[3, nbr_coords:] = 1  # add dimmension, fill with 1 for second object
                im_ori = normalize(im_ori)
            
                for permutation in all_permutations:  #3 permutations for the flips in all directions
                    perm = list(map(int, permutation))
                    im_trans = torch.clone(im_ori)
                    coords_trans = im_trans[0:3,:]
                    coords_labels_trans = im_trans[3,:]
                    perm = list(map(int, permutation))
                    #flip
                    coords_tran = flip(coords_trans, perm)
                    for R in range(5) : # nbr of rotations per perm
                        if R == 0: # no rotation for the first one
                            input_[0,0:3,:] = coords_trans
                            input_[0,3,:]   = coords_labels_trans
                            output = model.forward(input_.type(torch.float32))
                            preds.append(output.item())

                        else: 
                            coords_trans = rotate_tensor(coords_trans, nbr_coords) # rotation
                            input_[0,0:3,:] = coords_trans
                            input_[0,3,:]   = coords_labels_trans
                            output = model.forward(input_.type(torch.float32))
                            preds.append(output.item())
            else:
                x_new = torch.clone(x)
                for permutation in all_permutations: # 16 permuations
                    perm = list(map(int, permutation)) # transform the permuation into a list of int instead of strings
                    x_new[0,:,:,:,:] = test_transform(x[0,:,:,:,:], perm)
                    if channels == 2:
                        x1, x2 = to2Channels(x_new, ax=0)  # We check the two and do the average
                        x3, x4 = to2Channels(x_new, ax=1)  # we check the two and do the average
                        input_[:, 0, :, :, :] = x1.to('cuda')
                        input_[:, 1, :, :, :] = x2.to('cuda')
                        preds.append(model.forward(input_.type(torch.float32)).item())
                        preds.append(model.forward(input_.type(torch.float32)).item())
                        input_[:, 0, :, :, :] = x3.to('cuda')
                        input_[:, 1, :, :, :] = x4.to('cuda')
                        if model_type == 'ResNet':  
                            preds.append(Sigmoid(model.forward(input_.type(torch.float32))).item())
                        else:
                            preds.append(model.forward(input_.type(torch.float32)).item())
                    elif channels ==1:
                        x1 = toValues(x_new, ax=0).type(torch.float32).to('cuda')  # We check the two and do the average
                        x2 = toValues(x_new, ax=1).type(torch.float32).to('cuda')   # we check the two and do the average
                        if model_type == 'ResNet':  
                            preds.append(Sigmoid(model.forward(x1.type(torch.float32))).item())
                            preds.append(Sigmoid(model.forward(x2.type(torch.float32))).item())
                        else:
                            preds.append(model.forward(x1.type(torch.float32)).item())
                            preds.append(model.forward(x2.type(torch.float32)).item())
                            
                            
            
                
            pred = np.mean(preds)
            all_y[count,] = y.item()
            all_preds[count,:] = pred
            if y.item() == 1:  # P
                if pred > treshold:
                    TP += 1
                    results.append([IDs1[count], IDs2[count], y.item(), round(pred, 3), 1, 0, 0, 0])
                else:
                    FN += 1
                    results.append([IDs1[count], IDs2[count], y.item(), round(pred, 3), 0, 1, 0, 0])
            else:
                if pred < treshold:
                    TN += 1
                    results.append([IDs1[count], IDs2[count], y.item(), round(pred, 3), 0, 0, 1, 0])
                else:
                    FP += 1
                    results.append([IDs1[count], IDs2[count], y.item(), round(pred, 3), 0, 0, 0, 1])
            count += 1
            # print('True', y)
            # print('Predictions', preds)
       

 

    # Compute the AUC value
    auc = roc_auc_score(all_y, all_preds)
    # Compute the ROC curve
    fpr, tpr, thresholds = roc_curve(all_y, all_preds)
    # Plot the ROC curve
    plt.plot(fpr, tpr, label='AUC = {:.3f}'.format(auc))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    # Save the plot to a file
    plt.savefig(ckpt_path + '/roc.png', dpi=300)
    
    df = pd.DataFrame(results, columns=["ID1", "ID2", "True", "Pred", "TP", "FN", "TN", "FP"])
    Acc = (TP + TN) / (TP + FN + TN + FP)
    Rec = (TP) / (TP + FN)
    Pre = (TP) / (TP + FP)
    metrics = [[Acc, Rec, Pre, TP, FP, TN, FN]]
    df2 = pd.DataFrame(metrics, columns=['Acc', 'Rec', 'Pre', 'TP', 'FP', 'TN', 'FN'])
    #np.save(ckpt_path + '/predictions.npy', all_preds)
    #df.to_csv(ckpt_path + '/list_treshold_' + str(treshold) +'.csv')
    #df2.to_csv(ckpt_path + '/metrics_treshold' + str(treshold) +  '.csv')