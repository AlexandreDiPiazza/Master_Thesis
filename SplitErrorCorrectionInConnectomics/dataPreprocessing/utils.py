import numpy as np

def segOnlyAxons(seg: np.ndarray, gT: np.ndarray, axon_ids: list):
    """
    base on the axon_ids of the gT, keep only voxels of seg that correspond to an axon in the GT
    :param seg: segmentation, which we will only keep axons
    :param gT: gT
    :param axon_ids: list of axon_ids in the gT
    :return:
    """
    all_ids = np.array([], dtype = 'uint64')
    for i,axon_id in enumerate(axon_ids):
            # find pred_ids
            #seg[gt==axon_id].. find the parts of the image in the seg corresponding to axons, and their predicted value
            #np.unique(...) find the seg_ids given to this part, and the count of this seg_id
            pred_ids, pred_counts = np.unique(seg[gT==axon_id], return_counts=True)
            pred_counts = pred_counts[pred_ids>0]
            pred_ids = pred_ids[pred_ids>0]
            #take all the ids of at the locations of groundtruth axons
            all_ids = np.append(all_ids, pred_ids).astype('uint64')

    # Remove duplicate, we only want the ids
    all_ids = np.unique(all_ids )
    mask_pred = np.isin(seg, test_elements = all_ids)
    axon_pred = seg*mask_pred
    return axon_pred


def segOnlyAxonsv2(seg: np.ndarray, gT: np.ndarray, axon_ids: list) -> np.ndarray:
    """
  base on the axon_ids of the gT, keep only voxels of seg that correspond to an axon in the GT
  :param seg: segmentation, which we will only keep axons
  :param gT: gT
  :param axon_ids: list of axon_ids in the gT
  :return: new seg_array
  """

    is_axon = seg * gT  # with this, all the pixels id's that are 0 in axon_gT are set to 0
    mask_axons = (is_axon != 0)  # all non zeros elements are True
    seg_axons_ids = np.unique(seg[mask_axons]) #take all the IDs where the axons_gt is not background

    final_mask = np.isin(seg, test_elements=seg_axons_ids) # take all these ids to keep
    axon_seg = seg * final_mask # take all the ids
    return axon_seg


def removeNonAxons(seg_axons: np.ndarray, gT_axons: np.ndarray ) -> np.ndarray :
    """
    :param seg_axons: segmentation containing mostly axons, but still some errors due to some objects having few pixels
                      corresponding to axons
    :param gT_axons: GT segmentation containing only axons
    :return: the corrected segmentation containing only real axons
    """
    indices = np.unique(seg_axons) # all the indices
    indices = indices[indices > 0] # we don't take the 0
    for index in indices:
        N = np.sum(seg_axons == index)  # nbr of pixels have this index in the segmentation
        count = np.sum((gT_axons == 0) * (seg_axons == index))  # nbr of pixels of this index that correspond to a 0 in the GT
        if N < 5000: #small parts, more noise so we have smaller %
            if round(count/N,3) > 0.9: # More than 99% don't correspond to a part of the axon, it should be deleted
                #print(index)
                #Replace all these index by 0 bc it's not considered an axon
                seg_axons[seg_axons==index] = 0
        else:
            if round(count/N,3) > 0.6: # More than 80% don't correspond to a part of the axon, it should be deleted
                #print(index)
                #Replace all these index by 0 bc it's not considered an axon
                seg_axons[seg_axons==index] = 0
    return seg_axons

def removeNonAxonsv2(seg_axons: np.ndarray, gT_axons: np.ndarray ) -> np.ndarray :
    """
    :param seg_axons: segmentation containing mostly axons, but still some errors due to some objects having few pixels
                      corresponding to axons
    :param gT_axons: GT segmentation containing only axons
    :return: the corrected segmentation containing only real axons
    """

    seg_temp = seg_axons * (gT_axons == 0 ) # this segmentation contains non zeros only outside of the axons

    ids1, counts1 = np.unique(seg_axons, return_counts=True) # all ids and count in original segmentation
    ids1 = ids1[counts1>0]
    counts1 = counts1[counts1>0]
    dict1 = {ids1[i]: counts1[i] for i in range(len(ids1))}
    ids2, counts2 = np.unique(seg_temp, return_counts=True) # all ids and count outside axons pixels in the gT
    ids2 = ids2[counts2 > 0]
    counts2 = counts2[counts2 > 0]
    dict2 = {ids2[j]: counts2[j] for j in range(len(ids2))}
    for id in list(ids2):
        if dict1[id] < 5000:
            if round(dict2[id] / dict1[id], 3) > 0.9:
                ids1 = ids1[ids1!=id]
        else:
            if round(dict2[id] / dict1[id], 3) > 0.6:
                ids1 = ids1[ids1!=id]
    final_mask = np.isin(seg_axons, test_elements=ids1)
    return seg_axons * final_mask


