import numpy as np
import time
from utils import findNeighborsLongRange
from utils import segLabelsforAxonv3
import pandas as pd
import pickle


def findAllSplitErrors(gt: np.ndarray, seg: np.ndarray, save: str, longRange: bool) -> list:
    """
    Given the GT and the Output Segmenation, we find the all the split errors that were made. ( False Split)
    Added: don't take in objects with deltaZ < 2, they are too small and we don't take it into account later anyway.
    :param gt: groundtruth
    :param seg: segmentation
    :param axons_ids: list of axons IDS in the groundtruth
    :param longRange: if True, take not only neighbors but z+2, x+2 etc...  some longer range.
    :return: list of split errors: [(id1_1, id1_2),(id2_1, id2_2),...,(idN_1, idN_2)]
    """
    # 10, 100 for Mouse Cortex
    # 3, 1024 for SNEMI3D
    limit1 = 10; limit2 = 100; 
    list_split_errors = []
    list_ids = []
    pairs_seen = []
    # Find all the split errors that occur.
    Z, Y, X = np.shape(seg)
    dict_ids_gt_seg, seg_ids_stats = segLabelsforAxonv3(output = seg, gt = gt, Z= Z, Y = Y, X = X)

    # save the dicts
    with open("./MouseCortex/Dicts/dict_seg_stats.pickle", "wb") as handle:
            pickle.dump(seg_ids_stats, handle)
    with open("./MouseCortex/Dicts/dict_ids_gt_seg.pickle", "wb") as handle:
            pickle.dump(dict_ids_gt_seg, handle)
    
    # read the dcit:
    with open("./MouseCortex/Dicts/dict_seg_stats.pickle", 'rb') as handle:
        seg_ids_stats = pickle.load(handle)
    with open("./MouseCortex/Dicts/dict_ids_gt_seg.pickle", 'rb') as handle:
        dict_ids_gt_seg = pickle.load(handle)
    
    """
    with open("./SNEMI3D/Test/Dicts/dict_seg_stats.pickle", "wb") as handle:
            pickle.dump(seg_ids_stats, handle)
    with open("./SNEMI3D/Test/Dicts/dict_ids_gt_seg.pickle", "wb") as handle:
            pickle.dump(dict_ids_gt_seg, handle)
    
    # read the dcit:
    with open("./SNEMI3D/Test/Dicts/dict_seg_stats.pickle", 'rb') as handle:
        seg_ids_stats = pickle.load(handle)
    with open("./SNEMI3D/Test/Dicts/dict_ids_gt_seg.pickle", 'rb') as handle:
        dict_ids_gt_seg = pickle.load(handle)
    """
        
    N_axons = len(dict_ids_gt_seg)
    print('Starting iteration')
    i = -1
    for axon_id in dict_ids_gt_seg:
        i+=1
        print(round(i / N_axons * 100, 3))
        segIDs = dict_ids_gt_seg[axon_id]  # v2
        count = 0
        for segID in segIDs:  # iterate through each labels found segmentation
            N1 = seg_ids_stats[segID][0]
            count1 = seg_ids_stats[segID][7][axon_id] # nbr of pixels in GT for this seg_id in the gt for the axon_id
            if round(count1 / N1, 3) < 0.1:
                continue  # We don't consider this one as a true label corresponding to axonID, it only touches

            # Take the values of z to compute the delta z, and same for values of x and y to reduce search space
            # when computing the neighbors
            z_max = seg_ids_stats[segID][1] ; z_min = seg_ids_stats[segID][2]
            y_max = seg_ids_stats[segID][3] ; y_min = seg_ids_stats[segID][4]
            x_max = seg_ids_stats[segID][5] ; x_min = seg_ids_stats[segID][6]
            deltaz1 = seg_ids_stats[segID][1] - seg_ids_stats[segID][2] # delta z of this axon
            deltay1 = seg_ids_stats[segID][3] - seg_ids_stats[segID][4] # delta z of this axon
            deltax1 = seg_ids_stats[segID][5] - seg_ids_stats[segID][6] # delta z of this axon
            if deltaz1 < limit1 and (deltay1 < limit2 or deltax1 < limit2):  # We don't take in account too smal objects, they are too noisy and broken pieces.
                continue
            #Take the values of
            if longRange == False:
                neighs = seg_ids_stats[segID][8]
            else:
                #This part needs to be updated to similar method as no long range, casue too long
                neighs = findNeighborsLongRange(arr=seg[z_min: z_max+1, y_min:y_max+1, x_min:x_max+1]
                                                , axon_ID=segID, long_range_z=2, long_range_xy=4)

            for neigh in neighs:  # iterate throug the neighbors
                if any(x in ((neigh, segID), (segID, neigh)) for x in pairs_seen): # We already saw this pair on the other side
                    continue

                N2 = seg_ids_stats[neigh][0]
                try:
                    count2 = seg_ids_stats[neigh][7][axon_id]
                except:
                    continue # the key doesnt exist, the neighbor is not even part of the axon at all

                deltaz2 = seg_ids_stats[neigh][1] - seg_ids_stats[neigh][2]  # min - max
                deltay2 = seg_ids_stats[neigh][3] - seg_ids_stats[neigh][4] # delta z of this axon
                deltax2 = seg_ids_stats[neigh][5] - seg_ids_stats[neigh][6] # delta z of this axon
                if deltaz2 < limit1 and (deltay2 < limit2 or deltax2 < limit2):  # Again, we don't take in account too small pieces, they are broken ones.
                    continue
                if round(count2 / N2, 3) < 0.1:
                    continue  # We don't consider this one as a true label corresponding to axonID, it only touches
                if neigh in segIDs:  # if the neighbor is also a label corresponding to the gt, there is a split error
                    if (axon_id, segID, neigh) not in list_split_errors:  # checkin if put in (symmetric)
                        list_split_errors.append([axon_id, segID, neigh, N1, N2, deltaz1, deltaz2, deltay1, deltay2, deltax1, deltax2])
                        list_ids.append(axon_id)
                        list_ids.append(neigh)
                        pairs_seen.append((segID, neigh))

    df = pd.DataFrame(list_split_errors,
                      columns=["Axon-GT", "ID1-Seg", "ID2-Seg", "ID1-Pixels", "ID2-Pixels", "z1", "z2","y1","y2","x1","x2"])
    df.to_csv(save)
    return list_split_errors



if __name__ == '__main__':
    merges = []
    start_time = time.time()
    data_type = 'MouseCortex'
    if data_type == 'MouseCortex':
        data = 'Train'
        D1 = '../../datasets/MouseCortex/axons_only/'  # image
        # Prediction from VCG file on only axons
        seg = np.load(D1 + 'seg_axons.npy') 
        gt = np.load(D1 +  'gT_axons_corrected.npy') 
        print('Finished Loading')
        save_path = './MouseCortex/list_split_errors.csv'
    elif data_type == 'SNEMI3D':
        data = 'test'
        D1 = '../../datasets/SNEMI3D/axons_only/'+data+'/'  # image
        # Prediction from VCG file on only axons
        seg = np.load(D1 + 'seg1_axons.npy')
        gt = np.load(D1 + 'gT_axons.npy') 
        print('Finished Loading')
        save_path = './SNEMI3D/Test/list_split_errors.csv'

    longRange = False
    import time
    start = time.time()
    list_split_errors = findAllSplitErrors(gt=gt, seg=seg, save=save_path, longRange=longRange)
    print('TIME: ', time.time() - start_time)