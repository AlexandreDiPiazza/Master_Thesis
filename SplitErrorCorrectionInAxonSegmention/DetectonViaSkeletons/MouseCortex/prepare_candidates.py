import pandas as pd 
import numpy as np 



def remove_big_elems(df):
    """
    Remove broken pieces that are too small
    """
    print('Before: {}'.format(len(df)))
    limit1 = 30; limit2 = 300
    cond1 = (df['z1'] > limit1) | (df['x1']>limit2) | (df['y1']>limit2)
    cond2 = (df['z2'] > limit1) | (df['x2']>limit2) | (df['y2']>limit2)
    new_df = df[cond1 & cond2].reset_index(drop=True)
    print('After: {}'.format(len(new_df)))
    return new_df

def delete_false_splits(df): 
    """
    By exploring the GT data, we saw some GT split are actually not really split, BC it is the gt and the 
    target variable, we remove it 
    """
    false_splits = [
   (6422,7488), (6257,10054), (1612,6663),  (940,6799), (3789,4209), (700,7588), (7275,769), (1826,8715),
   (611,7452), (1605,7732), (5013,3331), (3725,6870), (1588, 8377), (7194, 10303), (890,6686), (3470,8169),
    (8442,5877)
    ]
    print('Nbr of splits to delete: {}'.format(len(false_splits)))
    print('Len Before: {}'.format(len(df)))
    for i in range(len(df)):
        ID1 = df.loc[i]['ID1-Seg']
        ID2 = df.loc[i]['ID2-Seg']
        for elem in false_splits:
            if (ID1 == elem[0] and ID2 == elem[1]) or (ID1 == elem[1] and ID2 == elem[0]):
                #print(elem, i )
                df = df.drop(index=i)
    df.reset_index(drop = True, inplace = True)
    print('Len After: {}'.format(len(df)))
    return df

def centerToConsider(l: list):
    """
    given a list of paire of endpoints, we choose the pair with the lowerst L2 norm, and compute
    the point at mean distance of the two endpoints.
    :param l: list of pairs of endPoints
    :return: center of the cube
    """
    mini = 1e3
    p = None
    for elem in l:
        L2 = computeEucl(elem[0], elem[1])
        if L2 < mini:
            mini = L2
            p = elem
    mean = (int(0.5 * (p[0][0] + p[1][0])), int(0.5 * (p[0][1] + p[1][1])), \
            int(0.5 * (p[0][2] + p[1][2])))
    return mean
def computeEucl(e1,e2):
    L2 = (5*(e1[0]-e2[0]))**2+(e1[1]-e2[1])**2+(e1[2]-e1[2])**2
    return np.sqrt(L2)

if __name__ == '__main__':
    # We load the data
    split_errors = pd.read_csv('list_split_errors.csv')#list_split_errors_Test_no_duplicatesZShortRange.csv')
    candidates = pd.read_csv('candidates_fin.csv', converters={'EndPoints': pd.eval})
    
    # We remove too small pieces from 
    big_split_errors = remove_big_elems(split_errors) 
    big_candidates = remove_big_elems(candidates)
    
    # We delete false split from the GT
    final_big_split_errors = delete_false_splits(big_split_errors)
    
    # take candidates not too close to lower edge
    big_candidates['cube_center'] = big_candidates['EndPoints'].apply(centerToConsider)
    big_candidates['keep'] = big_candidates['cube_center'].apply(lambda x: True if x[0]>15 else False)
    candidates_fin = big_candidates[big_candidates['keep']==True].reset_index(drop = True)
    
    #Split into lower and upper image, i.e candidates in z < 375 and candidates in z > 375
    candidates_fin['training_data'] = candidates_fin['cube_center'].apply(lambda x: True if x[0]<375 else False)
    candidates_train = candidates_fin[candidates_fin['training_data'] == True].copy().reset_index(drop = True)
    candidates_test = candidates_fin[candidates_fin['training_data'] == False].copy().reset_index(drop = True)
    print(candidates_test['EndPoints'].head(10))
    """
    #Baseline.
    tre = 
    for i in range(len(candidates_test))
        mini = 1e5
        endpoints_pairs = candidates_test.iloc[i]['EndPoints'][0]
        for pairs in endoints_pairs:
            l2 = np.sqrt((5 * (X1[0] - X2[0])) ** 2 + (X1[1] - X2[1]) ** 2 + (X1[2] - X2[2]) ** 2)
    
    print('Before:', len(final_big_split_errors))
    """
    # We add elements that we missed in the GT
    new_elements_GT = [{'ID1-Seg': 6422, 'ID2-Seg': 7488},{'ID1-Seg': 6257, 'ID2-Seg': 10054},{'ID1-Seg': 5476, 'ID2-Seg': 6315},{'ID1-Seg': 940, 'ID2-Seg': 6799},
                    {'ID1-Seg': 3889, 'ID2-Seg': 9525},{'ID1-Seg': 11074, 'ID2-Seg': 1629},{'ID1-Seg': 10429, 'ID2-Seg': 2718},{'ID1-Seg': 6538, 'ID2-Seg': 3372},
                    {'ID1-Seg': 2673, 'ID2-Seg': 10026},{'ID1-Seg': 7308, 'ID2-Seg': 8142},{'ID1-Seg': 3789, 'ID2-Seg': 4209}, {'ID1-Seg': 4669, 'ID2-Seg': 8944},
                    {'ID1-Seg': 8017, 'ID2-Seg': 6239}]
    
    
    for new_elem_GT in new_elements_GT: # we add it to the GT
        # create a new dataframe with the new row
        new_df = pd.DataFrame(new_elem_GT, index=[0])
        # concatenate the original dataframe with the new dataframe
        final_big_split_errors = pd.concat([final_big_split_errors, new_df], ignore_index=True)
    print('After: ', len(final_big_split_errors))
        
    
    
    #final_big_split_errors.to_csv('final_list_split_errors.csv') 
    #candidates_train.to_csv('./Train/candidates_train.csv')
    #candidates_test.to_csv('./Test/candidates_test.csv')

    
    
    
    
    
    
    
    
    
    
    
    
    