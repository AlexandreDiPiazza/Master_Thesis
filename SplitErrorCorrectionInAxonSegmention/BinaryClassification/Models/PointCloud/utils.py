import torch
import numpy as np
from scipy.spatial.transform import Rotation

def take_all_coords(array: np.ndarray, nbr_coords) -> torch.tensor:
    """
    Take the coords for obj1 and obj2 (2 channels) for each image
    return a N*3*coordinates tensor
    """
    nbr_samples = np.shape(array)[0]
    final_array = np.zeros(shape=(nbr_samples, 4, 2 * nbr_coords))  # 2*nbr_coords bc we take nbr_coords for each object, and they are 2
    for i in range(nbr_samples):
        coords1 = take_coords(array[i, 0, :, :, :], value=1,
                              nbr_coords=nbr_coords)  # take coords of pixels 1 for first object
        coords2 = take_coords(array[i, 1, :, :, :], value=1,
                              nbr_coords=nbr_coords)  # take coords of pixels 1 for second object
       
        final_array[i, 0:3, :nbr_coords] = coords1  # fill with coordinates obj1
        final_array[i, 3, :nbr_coords] = 0  # Add dimmension, fill with 0 for first object
        final_array[i, 0:3, nbr_coords:] = coords2  # fill with coordinates obj2
        final_array[i, 3, nbr_coords:] = 1  # add dimmension, fill with 1 for second object
        # Normalize between 0 and 1
        normalize = False
        if normalize == True:
            mini = np.min(final_array[i, 0:3, :], axis=1)
            maxi = np.max(final_array[i, 0:3, :], axis=1)
            final_array[i, 0:3, :] = (final_array[i, 0:3, :] - mini[:, None]) / (maxi[:, None] - mini[:, None])
       
        rotation = True
        if rotation == True: # Perform a rotation of the PC
            angle_z = np.random.randint(low=0, high=90)
            angle_y = np.random.randint(low=0, high=90)
            angle_x = np.random.randint(low=0, high=90)
           
            r = Rotation.from_euler('xyz', [angle_z, angle_y, angle_x], degrees = True)
            rotation_matrix = r.as_matrix()
            rotation_center =  np.array([[0.5, 0.5, 0.5]]).T # All our points are between 0 and 1, if we do directly a rotation some points will be negative
            rotated_points = np.abs(np.round(np.dot(rotation_matrix, final_array[i,0:3,:]-rotation_center)+ rotation_center, decimals = 6))
            final_array[i,0:3,:] = rotated_points
            
            
           
        # shuffle the columns, not necessary as PC is invariant to permutations
        final_array[i, 0:4, :] = final_array[i, 0:4, :][:, np.random.permutation(final_array[i, 0:4, :].shape[1])]
    return torch.tensor(final_array)

def take_all_coordsV2(array: np.ndarray, target: np.ndarray, nbr_coords, up_sample0, up_sample1, PC_params) -> torch.tensor:
    """
    Take the coords for obj1 and obj2 (1 channels) for each image
    return a N*3*coordinates tensor
    up_sample0: nbr of data_augmentations to do when target = 0 
    up_sample1: nbr of data_augmentations to do when target = 1
    """
    nbr_samples = np.shape(array)[0]
    zeros = np.sum(target==0)
    ones = np.sum(target == 1)
    final_array = np.zeros(shape=(zeros*up_sample0 + ones*up_sample1, 4, 2 * nbr_coords))  # 2*nbr_coords bc we take nbr_coords for each object, and they are 2
    final_target = np.zeros(shape=(zeros*up_sample0 + ones*up_sample1,1))
    im_ori = np.zeros(shape=(4,2 * nbr_coords))
    count = 0 
    for i in range(nbr_samples):
        
        tar = target[i] #target
        if tar == 0:
            augmentations = up_sample0 # How many transformations to do 
        elif tar == 1:
            augmentations = up_sample1
            
        coords_first  = np.argwhere(array[i, 0, :, :, :] == 1)  # shape n,3
        coords_second = np.argwhere(array[i, 0, :, :, :] == -1)  # shape n,3
        #print(np.mean(coords_first))
        #print(np.mean(coords_second))
        for l in range(augmentations):   
        
            coords1 = sample(coords_first, nbr_coords).T
            coords2 = sample(coords_second, nbr_coords).T
           
            im_ori[0:3, :nbr_coords] = coords1
            im_ori[3, : nbr_coords] = 0
            im_ori[0:3, nbr_coords:] = coords2
            im_ori[3,nbr_coords:] = 1
            # Normalize between 0 and 1
            normalize = True
            if normalize == True:
                mini = np.min(im_ori, axis=1)
                maxi = np.max(im_ori, axis=1)
                im_ori = (im_ori - mini[:, None]) / (maxi[:, None] - mini[:, None])
            
            im_trans = im_ori[0:3,:].copy() # copy the coords
            im_trans_labels = im_ori[3,:].copy() # copy the label of the coords
            if l == 0: #we keep one sample without noise
                rotation = True
                translation = False
                scaling = False
                jittering = False
                flip = True
            else: 
                translation = PC_params['translation']
                scaling = PC_params['scaling']
                jittering = PC_params['jittering']
                rotation = True
                flip = True
                
            if flip == True:
                z_flip = 1 ; y_flip = 1; x_flip = 1 #no flips
                if np.random.uniform()<0.5: #z_flip
                    z_flip = -1
                if np.random.uniform()<0.5:
                    y_flip = -1
                if np.random.uniform()<0.5:
                    x_flip = -1
                flips = np.array([z_flip, y_flip, x_flip])
                # create the scaling matrix
                flip_matrix = np.diag(flips)
                rotation_center =  np.array([[0.5, 0.5, 0.5]]).T
                # apply the scaling transformation to the point cloud
                im_trans = np.clip((flip_matrix @ (im_trans-rotation_center)) + rotation_center, a_min=0, a_max=1)
               
            if rotation == True: # Perform a rotation of the PC
                angle_z = np.random.randint(low=0, high=90) 
                angle_y = np.random.randint(low=0, high=90)
                angle_x = np.random.randint(low=0, high=90)
              
                r = Rotation.from_euler('xyz', [angle_z, angle_y, angle_x], degrees = True)
                rotation_matrix = r.as_matrix()
                rotation_center =  np.array([[0.5, 0.5, 0.5]]).T # All our points are between 0 and 1, if we do directly a rotation some points will be negative
                im_trans =  np.clip(np.dot(rotation_matrix, im_trans-rotation_center)+ rotation_center, a_min = 0, a_max = 1) #apply rotation
        
            
            if scaling == True: # Perform a scaling of the PC
                scaling_factor_x = np.random.uniform(low=0.95, high=1.05)
                scaling_factor_y = np.random.uniform(low=0.95, high=1.05)
                scaling_factor_z = np.random.uniform(low=0.95, high=1.05)
                scale = np.array([[scaling_factor_x], [scaling_factor_y], [scaling_factor_z]])
                im_trans = im_trans * scale
                im_trans = np.clip(im_trans, a_min=0, a_max=1) # Ensure points remain positive and between 0 and 1
                
            if translation == True:
                rd = np.random.uniform()
                if rd > 0.8 : 
                # We do only one translation in one of the directions: 
                    translation_x = np.random.uniform(low=-0.1, high=0.1)
                    translation_y = 0
                    translation_z = 0
                elif rd > 0.6 : 
                    translation_x = 0
                    translation_y = np.random.uniform(low=-0.1, high=0.1)
                    translation_z = 0
                elif rd > 0.4 : 
                    translation_x = 0
                    translation_y = 0
                    translation_z = np.random.uniform(low=-0.1, high=0.1)
                else: # no rotation
                    translation_x = 0
                    translation_y = 0
                    translation_z = 0
                translation = np.array([[translation_x], [translation_y],[translation_z]])  
                im_trans = im_trans - translation
                im_trans = np.clip(im_trans, a_min=0, a_max=1)
            
            if jittering == True: # Perform a jittering of the PC
                jittering_factor = np.random.uniform(low=-0.00005, high=0.00005, size=(3,2*nbr_coords))
                im_trans = im_trans + jittering_factor
                im_trans = np.clip(im_trans, a_min=0, a_max=1) # Ensure points remain positive and between 0 and 1
            final_array[count,0:3,:] = im_trans
            final_array[count, 3,: ] = im_trans_labels
            #final_array[count, 0:4, :] = final_array[count, 0:4, :][:, np.random.permutation(final_array[count, 0:4, :].shape[1])] 
            final_target[count] = tar
            count +=1
            
    
    return torch.tensor(final_array),torch.tensor(final_target)

def take_coords(array: np.ndarray, value: int, nbr_coords: int) -> np.ndarray:
    """
    Compute the 3D points of the array equal to value
    We take only nbr_coords such points
    Each coord is in the order:
    input: shape Z*Y*Y
    output: shape 3*nbr_coords
    """
    # the coordinates of np.argwhere start from the upper left, [0,0] is upper left, first row first col
    # order, first axis, second axis, third axis, Z,Y,X in our case, starting from upper left. [0,0,0]
    coords = np.argwhere(array == value)  # shape n,3
    #print('tk', np.mean(coords))
    sample_coords = sample(coords, nbr_coords)  # sample coordinates by taking randomly nbr_coords
    return sample_coords.T  # for our format, we want shape coordsxn

def sample_rows(arr, m):
    n, d = arr.shape
    if m < n:
        indices = random.sample(range(n), m)
        return arr[indices, :]
    else:
        indices = list(range(n)) * ((m + n - 1) // n)
        indices = indices[:m]
        return arr[indices, :]
        

def sample(coords: np.ndarray, n: int) -> np.ndarray:
    """
    Sample by taking n random rows from coords.
    if there are more rows than the nbr we want:
        we just shuffle and take n rows
    else:
        we take all the rows, and then take other rows until we have n, with replacement

    """
    size_each_coord = np.shape(coords)[1]
    np.random.shuffle(coords)  # shuffle coordinates

    if n <= len(coords):  # we shuffle and take the first n elements
        #print('Here', np.mean(coords, axis = 0), np.mean(coords[:n],axis=0), np.mean(coords[:n,:], axis = 0))
        return coords[:n]

    else:  # we take all the coords first, and then sample with replacement until we reach n points
        final_coords = np.zeros(shape=(n, size_each_coord))
        final_coords[:len(coords)] = coords  # fill with the first elements
        for i in range(len(coords), n):
            random_idx = np.random.randint(low=0, high=len(coords))
            final_coords[i, :] = coords[random_idx, :]  # sample with replacement
        return final_coords