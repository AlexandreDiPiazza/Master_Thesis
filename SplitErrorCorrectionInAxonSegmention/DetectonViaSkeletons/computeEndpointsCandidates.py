import time
import numpy as np
import kimimaro
from utils import euclDist
import pandas as pd
import pickle 

def findEndpoints(skel: np.ndarray, ID: int, Z, Y, X) -> list:
    """
    Based on the Skeleton representation as a graph, we only take the nodes that have only one edge, those are the endpoints
    """
    # 15, 75, but can reduce a bit, middle not needed 
    # No limit for SNEMI, too small 
    limit1 = 15 ; limit2 = 75 # so not take the points near the edges
    try :
        node_id, count = np.unique(skel[ID].edges.flatten(), return_counts = True) # find all the nodes IDs and their count
    except:
        return [] # key ID error
    unique_nodes = node_id[count == 1] # take only the IDs that are present ones => endpoints
    end_points = []
    for elem in unique_nodes:
        points = list(skel[ID].vertices[elem])
        if points[0] < limit1 or points[0]>Z-limit1: # Don't take nodes close to the edges (dim of the cube / 2):
            continue
        if points[1] < limit2 or points[1]>Y-limit2: # Don't take nodes close to the edges (dim of the cube / 2):
            continue
        if points[2] < limit2 or points[2]>X-limit2: # Don't take nodes close to the edges (dim of the cube / 2):
            continue
        
        end_points.append(points)
    return end_points

def listOfCandidatesEndpoint(seg, eucl_treshold):
    list_candidates = []
    list_pairs_seen = []
    Z, Y, X = np.shape(seg)
    # 10, 100 for Mouse Cortex
    # 3, 1024 for SNEMI3D
    limit1 = 10; limit2 = 100; 

    with open('MouseCortex/Dicts/dict_seg_stats.pickle', 'rb') as handle:
        seg_ids_stats = pickle.load(handle)
    print('Number of Axons: ', len(seg_ids_stats))
    
    """
    with open("./SNEMI3D/Train/Dicts/dict_seg_stats.pickle", 'rb') as handle:
        seg_ids_stats = pickle.load(handle)
    """
 
    ids_check = []  # the ids on which we will compute the skeleton 
    count = 0
    for seg_id in seg_ids_stats:
        z_max = seg_ids_stats[seg_id][1]; z_min = seg_ids_stats[seg_id][2]
        y_max = seg_ids_stats[seg_id][3]; y_min = seg_ids_stats[seg_id][4]
        x_max = seg_ids_stats[seg_id][5]; x_min = seg_ids_stats[seg_id][6]
        
        deltaZ = z_max - z_min
        deltaX = x_max - x_min 
        deltaY = y_max - y_min
        if  deltaZ < limit1 and (deltaX < limit2 or deltaY < limit2): # just check those for now 
            continue
        ids_check.append(seg_id) # we keep this id
    print('Nbr of axons we consider: ', len(ids_check))

    print('Starting to Compute the skeleton')
    t1 = time.time()
    parralels = 16
    # if you want to compute the skeleton with skimage.morpgonly, use directly the function implemented and 
    # find the voxels that have only one neighbor
    skel = kimimaro.skeletonize(seg, teasar_params={
        'max_paths': 50 #None# 50 # None# max_paths = 50 for EM, 
         },
        # anisotropy=(30,16,16),
        object_ids=ids_check,
        parallel=parralels)
    print('Time to Compute the skeleton: ', time.time() - t1)

    for seg_id in ids_check:
        endpoints1 = findEndpoints(skel, seg_id, Z, Y, X)
        neighs_ids = seg_ids_stats[seg_id][8]
       
        for neigh_id in neighs_ids:
            if ((seg_id, neigh_id) in list_pairs_seen) or (
                    (neigh_id, seg_id) in list_pairs_seen):  # we already saw this pair
                continue
            if neigh_id not in ids_check:
                continue 
            list_pairs_seen.append((seg_id, neigh_id))
            endpoints2 = findEndpoints(skel, neigh_id, Z, Y, X)
          
            
            candidates = euclDist(endpoints1, endpoints2, eucl_treshold)
            if len(candidates) > 0:
                dz1 = seg_ids_stats[seg_id][1]- seg_ids_stats[seg_id][2]
                dy1 = seg_ids_stats[seg_id][3]- seg_ids_stats[seg_id][4]
                dx1 = seg_ids_stats[seg_id][5]- seg_ids_stats[seg_id][6]
                dz2 = seg_ids_stats[neigh_id][1]- seg_ids_stats[neigh_id][2]
                dy2 = seg_ids_stats[neigh_id][3]- seg_ids_stats[neigh_id][4]
                dx2 = seg_ids_stats[neigh_id][5]- seg_ids_stats[neigh_id][6]
                list_candidates.append([seg_id, neigh_id, seg_ids_stats[seg_id][0], seg_ids_stats[neigh_id][0], candidates,
                dz1, dz2, dy1, dy2, dx1, dx2])
    df = pd.DataFrame(list_candidates,
                      columns=["ID1-Seg", "ID2-Seg", "ID1-Pixels", "ID2-Pixels", "EndPoints",  "z1", "z2","y1","y2","x1","x2"])
    
    df.to_csv('./MouseCortex/candidatesNEW.csv')

if __name__ == '__main__':
    """
    Compute all the endpoints candidates. For an object, look at each of his neighbors
    We compute the skeleton of the object itself + its neighbors, and the endpoints based on the skeleton.
    For each pair of endpoints, if it is < to a treshold we add it to the list.
    Note:
    The files created for SNEMI dataset were created with the skeletonize function from skimage.morphology, which is 
    more precise but longer, and thus too long for the MouseCortex dataset. This file shows an implemenatation
    of the kimaro function for both
    """
    start_time = time.time()
    data_type = 'MouseCortex'
    tresh = 60
    if data_type == 'MouseCortex':
        D1 = '../../datasets/MouseCortex/axons_only/'  # image
        # Prediction from VCG file on only axons
        seg = np.load(D1 + 'seg_axons.npy') # We don't take candidates near the border
        print('Finished Loading')
    elif data_type == 'SNEMI3D':
        data = 'train'
        D1 = '../../datasets/SNEMI3D/axons_only/'+data+'/'  # image
        # Prediction from VCG file on only axons
        seg = np.load(D1 + 'seg1_axons.npy')
        gt = np.load(D1 + 'gT_axons.npy') 
        print('Finished Loading')
    
    listOfCandidatesEndpoint(seg=seg, eucl_treshold=tresh)
    