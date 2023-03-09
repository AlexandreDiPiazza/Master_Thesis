import numpy as np
import pandas as pd
import yaml

def merge_chains(l: list) -> list: 
    """
    From a list of lists, merge the list that have a common eleement together
    taken from: https://stackoverflow.com/questions/4842613/merge-lists-that-share-common-elements
    """
    out = []
    while len(l)>0:
        first, *rest = l
        first = set(first)

        lf = -1
        while len(first)>lf:
            lf = len(first)

            rest2 = []
            for r in rest:
                if len(first.intersection(set(r)))>0:
                    first |= set(r)
                else:
                    rest2.append(r)     
            rest = rest2
        if list(first) != [] : #empty list
            out.append(list(first))
        l = rest
    return out

def create_chains(dataFrame):
    """
    From the results, we create the chains of eleements together
    """
    list_chains = [[]]
    for i in range(len(dataFrame)):
        ID1 = int(dataFrame.iloc[i]['ID1'])
        ID2 = int(dataFrame.iloc[i]['ID2'])
    
        if i == 0: # first elemen
            list_chains.append([ID1,ID2])
        else:
            new = True
            for j in range(len(list_chains)): 
                if (ID1 in list_chains[j]) or (ID2 in list_chains[j]): # this elem is part of a previous chain
                    new = False
                    if (ID1 in list_chains[j]) and (ID2 not in list_chains[j]): # first element part of the chain
                        list_chains[j].append(ID2) 
                    elif (ID2 in list_chains[j]) and (ID1 not in list_chains[j]):# second element paet of the chain
                        list_chains[j].append(ID1)
            if new == True: # this elemen is not a part of any chain
                list_chains.append([ID1,ID2]) 
    
    # At this point, list_chains contains duplicates, i.e chains that still need to be merged
    # if one chain is [1,3], the other is [5,8], and we have the candidates [8,3], instead of merging the two lists 
    # we will have two lists [1,3,8] and [5,8,3] but we would want : [1,3,5,8]
    # we still need to merge the lists that share a common element 
    final_list = merge_chains(list_chains)
    return final_list


if __name__ == '__main__':
    
    treshold = 0.8
    
    results_path = '../BinaryClassification/Models/Perfection/perfect_predictionsAxonEM.csv'
    save_path ='./corrections/MouseCortex/perfect_correction.npy'
    #results = pd.read_csv(results_path + '/list_treshold_0.72.csv')
    results = pd.read_csv(results_path)
    # List of all the merges done 
    list_merges = [(results.iloc[i]['ID1'], results.iloc[i]['ID2'] )   for i in range(len(results)) if results.iloc[i]['Pred'] > treshold]
    data_type = 'MouseCortex'
  
    data_type = 'MouseCortex'
    print(save_path)
    if data_type == 'SNEMI3D':
        data = 'test'
        D1 = '../datasets/SNEMI3D/axons_only/' + data + '/'  # image
        seg = np.load(D1 + 'seg1_axons.npy')
    elif data_type == 'MouseCortex':
        D1 = '../datasets/MouseCortex/axons_only/'  # image
        gt = np.load(D1 +  'gT_axons_corrected.npy')[375:,:,:] # first part is for test 
        seg = np.load(D1 + 'seg_axons.npy')[375:,:,:]


    print('Before Binary Classification, nbr of candidates', len(results))
    results = results[results['Pred']>treshold].reset_index()
    print('After, nbr of merges', len(results))
    
    
    
    results = pd.DataFrame(list_merges, columns =['ID1', 'ID2'])
    list_merges = create_chains(results)

    new_seg = seg.copy()
    print('Before, Nbr of Unique IDs: ', len(np.unique(new_seg)))
    for l in list_merges:
        new_ID = l[0]
        for ID in l[1:]:
            new_seg[new_seg == ID] = new_ID
    print('After, Nbr of Unique IDs: ', len(np.unique(new_seg)))
    np.save(save_path, new_seg)