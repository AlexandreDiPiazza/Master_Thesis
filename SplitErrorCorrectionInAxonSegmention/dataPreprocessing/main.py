import numpy as np
import imageio
import h5py
import os
from PIL import Image, ImageSequence

from utils import segOnlyAxonsv2, removeNonAxonsv2

def axonsOnly(gTPath, tif, predPath, axonIDs, savePath):
    """
    Transform the gT and Seg by keeping only the axons
    :param gTPath: path of GT data
    :param tif: boolean, to load data
    :param predPath: path of segmentation data
    :param axonIDs: list of axonIDs
    :param savePath: where to save the data
    :return:
    """
    # Loading the ground-truth
    if tif == True:
        gT_tif = Image.open(gTPath)
        w, h = gT_tif.size
        gT = np.empty(shape=(100, h, w))
        for i, page in enumerate(ImageSequence.Iterator(gT_tif)):
            gT[i, :, :] = np.array(gT_tif)
        gT = gT.astype(dtype='uint16')
    else:
        gT = h5py.File(gTPath, 'r')
        gT = np.array(gT['/main'])
        
    
    # In the case of Mouse dataset
    
    axonIDs = list(np.unique(gT)) # gT contains already only axons so we can just take all the IDs
    """
    # Preparing Only axons Dataset
    mask_axons = np.isin(gT, test_elements=axonIDs)  # Create a Mask keeping only the pixels corresponding to axons
    axon_gT = gT * mask_axons  # Applying the mask to keep only the axons
    z, h, w = np.shape(axon_gT)
    print(f"Values before: {len(np.unique(gT))} Values after: {len(np.unique(axon_gT))}")
    """
    axon_gT = gT.copy()
    np.save(savePath + 'gT_axons.npy', axon_gT)

    # Loading the segmentations
    pred = h5py.File(predPath, 'r')
    aff_map_train = np.array(pred['/main'])
    seg1 = np.array(pred['/main'])

    seg1_axons = segOnlyAxonsv2(seg1,axon_gT,axonIDs)
    seg1_final_axons = removeNonAxonsv2(seg1_axons, axon_gT)
    np.save(savePath + 'seg_axons.npy', seg1_final_axons)

    # There are still in segn_axons some objects that don't really correspond to the axons, bc only
    # a small part of the bigs correspond to the axons and thus this object was chosen, but shouldt be deleted

if __name__ == '__main__':
    """
    We take only the axons from the dense segmentation
    """
    gTPathTrain = ('../../pytorch_connectomics/datasets/SNEMI3D/train_label.tif')
    predTrainPath = ('../../pytorch_connectomics/outputs/SNEMI_UNet/train/pred.h5')
    axon_ids_train = [3, 4, 42, 52, 69, 71, 79, 82, 86, 87, 89, 94, 101, 103, 104, 108, 116, 129, 264,
                      237, 245, 249, 169, 261, 232, 200, 257, 233, 145, 248, 250, 164, 199, 223, 241]
    savePathTrain = '../datasets/axons_only/train_New/'
    #axonsOnly(gTPath = gTPathTrain, tif = True, predPath = predTrainPath, axonIDs = axon_ids_train, savePath=savePathTrain)

    gTPathTest = ('../../pytorch_connectomics/Dip_GT_pred/test-labels.h5')
    predTestPath = ('../../pytorch_connectomics/Dip_GT_pred/pred_base_resnet.h5')
    axon_ids_test = [121, 124, 132, 133, 135, 138, 141, 145, 146, 147, 157, 164, 165, 168, 169, 183, 184,
                     212, 213, 235, 249, 272, 338, 380, 385, 406, 408, 123, 125, 140, 144, 148, 159, 170,
                     182, 206, 211, 219, 225, 259, 298, 307]

    predTestPath = '../datasets/MouseCortex/dense/seg_16nm_v0.h5'
    gTPathTest = ('../datasets/MouseCortex/axons_only/Test/axon_16nm_v3.1.h5')
    axon_ids_test = None 
    savePathTest = '../datasets/MouseCortex/axons_only/Test/'
    axonsOnly(gTPath=gTPathTest, tif = False, predPath=predTestPath, axonIDs=axon_ids_test, savePath=savePathTest)




