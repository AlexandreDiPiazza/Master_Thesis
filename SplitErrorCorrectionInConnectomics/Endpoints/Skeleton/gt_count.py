import numpy as np 
import pickle

if __name__ == '__main__':
    """
    Simply compute the count of each ID in the GT, and save it as a dict
    """
    data_type = 'MouseCortex'
    if data_type == 'MouseCortex':
        data = 'Train'
        D1 = '../../datasets/MouseCortex/axons_only/'  # image
        gt = np.load(D1 +  'gT_axons.npy') 
 
    elif data_type == 'SNEMI3D':
        data = 'test'
        D1 = '../../datasets/SNEMI3D/axons_only/'+data+'/'  # image
        gt = np.load(D1 + 'gT_axons.npy') 
    unique, counts = np.unique(gt, return_counts = True)
    
    counts = counts[unique>0] # we don't care about the background
    unique = unique[unique>0]
    Dict = {}
    for ID, count in zip(unique, counts):
        Dict[ID] = count
    
  
    # save the dicts
    with open("./MouseCortex/Dicts/gt_count.pickle", "wb") as handle:
            pickle.dump(Dict, handle)
    #with open("./SNEMI3D/Test/Dicts/gt_count.pickle", "wb") as handle:
    #        pickle.dump(Dict, handle)
    
    
