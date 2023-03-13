import numpy as np
import time
import pandas as pd
from utils import euclDist, smallSkelEndPointsv2
import pickle


def listOfCandidatesEndpoint(seg: np.ndarray, eucl_treshold: int, save: str, size: int) -> list:
    """
    Compute the Endpoints Candidates
    :param seg: segmentation
    :param eucl_treshold: treshold to merge
    :param save: save path
    :param size: limite size of the pixels to consider in each object
    :return: list of candidate endpoints
    """
    # Read the Dict:
    with open('dict_seg_stats.pickle', 'rb') as handle:
        seg_ids_stats = pickle.load(handle)

    list_candidates = []
    N = len(seg_ids_stats) #nbr of keys in the dict
    list_pairs_seen = []
    skeletons_dict = {}  # Computing the skeletons is the very long part, we will store them in a spart matrix each time we compute it to not compute it again if we see it again
    current_iter = 0
    for seg_id in seg_ids_stats: # go through the keys of the dict
        count_id = seg_ids_stats[seg_id][0]
        current_iter += 1
        #print(round(current_iter / N * 100, 3))
        if count_id < size:  # we don't check parts that are too small for now.
            continue
        z_max = seg_ids_stats[seg_id][1]; z_min = seg_ids_stats[seg_id][2]
        y_max = seg_ids_stats[seg_id][3]; y_min = seg_ids_stats[seg_id][4]
        x_max = seg_ids_stats[seg_id][5]; x_min = seg_ids_stats[seg_id][6]
        deltaz1 = z_max - z_min
        if deltaz1 < 3 :
            continue
        neighs_ids = seg_ids_stats[seg_id][8]

        if seg_id not in skeletons_dict:  # we never computed this skeleton before
            endPoints_target = smallSkelEndPointsv2(seg[z_min:z_max+1, y_min:y_max+1, x_min:x_max+1]
                                                    ,z_min,y_min,x_min,seg_id)

            skeletons_dict[seg_id] = endPoints_target  # we add the coordinates in a sparse format to the dictinnary
        else:
            endPoints_target = skeletons_dict[seg_id]  #

        for neigh_id in neighs_ids:
            if ((seg_id, neigh_id) in list_pairs_seen) or (
                    (neigh_id, seg_id) in list_pairs_seen):  # we already saw this pair
                continue
            if   seg_ids_stats[neigh_id][0] < size:  # to small, we don't check
                continue
            list_pairs_seen.append((seg_id, neigh_id))
            z_max = seg_ids_stats[neigh_id][1]; z_min = seg_ids_stats[neigh_id][2]
            y_max = seg_ids_stats[neigh_id][3]; y_min = seg_ids_stats[neigh_id][4]
            x_max = seg_ids_stats[neigh_id][5]; x_min = seg_ids_stats[neigh_id][6]
            deltaz2 = z_max - z_min  # min - max
            if deltaz2 < 3:
                continue
            if neigh_id not in skeletons_dict:  # we never computed this skeleton before
                endpoints_neigh = smallSkelEndPointsv2(seg[z_min:z_max + 1, y_min:y_max + 1, x_min:x_max + 1]
                                                        , z_min, y_min, x_min, neigh_id)
                skeletons_dict[neigh_id] = endpoints_neigh  # We keep the coordinates
            else:
                endpoints_neigh = skeletons_dict[neigh_id]  #

            # print('Time:', time.time() - start)
            candidates = euclDist(endPoints_target, endpoints_neigh, eucl_treshold)
            if len(candidates) > 0:
                count_id2 = seg_ids_stats[neigh_id][0]
                list_candidates.append([seg_id, neigh_id, count_id, count_id2, candidates, deltaz1, deltaz2])

    df = pd.DataFrame(list_candidates,
                      columns=["ID1-Seg", "ID2-Seg", "ID1-Pixels", "ID2-Pixels", "EndPoints", "Z1", "Z2"])
    #df.to_csv(save + '_' + str(eucl_treshold) + '.csv')
    print(df.head())
    print(len(df))
    return list_candidates


if __name__ == '__main__':
    """
    Compute all the endpoints candidates. For an object, look at each of his neighbors
    We compute the skeleton of the object itself + its neighbors, and the endpoints based on the skeleton.
    For each pair of endpoints, if it is < to a treshold we add it to the list.
    """
    merges = []
    start_time = time.time()
    data = 'train'
    limit_size = 0
    tresh = 50

    D1 = '../../datasets/SNEMI3D/axons_only/' + data + '/'  # image
    # Prediction from VCG file on only axons
    gt = np.load(D1 + 'gT_axons.npy')
    seg = np.load(D1 + 'seg1_axons.npy')
    print('Finished Loading')
    save_path = './T' + str(data[1:]) + '/EndPointsCandidates/list_candidatesfullZ'
    list_candidates_endpoints = listOfCandidatesEndpoint(seg=seg, eucl_treshold=tresh, save=save_path, size=limit_size)
