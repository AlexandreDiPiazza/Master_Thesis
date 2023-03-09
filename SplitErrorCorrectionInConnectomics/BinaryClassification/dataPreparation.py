import numpy as np
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
import torch


class Dataset():
    def __init__(self, X_train, y_train, X_val, y_val, aug_0, aug_1, channels, pointCloud_bool):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.aug_0 = aug_0
        self.aug_1 = aug_1
        self.channels = channels
        self.pointCloud_bool = pointCloud_bool

    def augmentDataTrain(self):
        # split into train_val
        X_train = self.X_train
        y_train = self.y_train
        Z, Y, X = np.shape(X_train[0, 0, :, :, :])

        my_transforms = transforms.Compose([
            # SYMMETRIE PAR RAPPORT AU PLAN Z,Y. (valeurs z,y change pas, mais changement pour les x)
            transforms.RandomHorizontalFlip(p=0.5),
            # SYMMETRIE PAR RAPPORT AU PLAN Z,X. (valeurs z,x change pas, mais changement pour les y)
            transforms.RandomVerticalFlip(p=0.5),
        ])
        N = len(y_train)
        zeros = torch.sum(y_train == 0)
        ones = torch.sum(y_train == 1)
        # 3 augmentations for the 0, 5 augmentations for the ones.
        X_aug = torch.zeros(size=(2*(zeros * self.aug_0 + ones * self.aug_1), self.channels, Z, Y, X),
                            dtype=torch.int8)  # Containing the augmentations
        y_aug = []
        count = 0
        for i in range(N):
            im_ori = X_train[i, :, :, :, :].clone()
            target = y_train[i]
            # We do 5 augmentations for the ones, and 3 for the zeros.
            if target == 1:
                augmentations = self.aug_1
            else:
                augmentations = self.aug_0
            for j in range(augmentations):  # different nbr of augmentations for 1s and 0s
                im = torch.clone(im_ori)
                rand = torch.rand(size=(11,))
                if rand[0] > 0.5:
                    # Rotation de 90.First transform to do bc of the noise
                    im = transforms.functional.rotate(img=im, angle=90)
                # parameters for the affine transformation
                scale = 0.4 * rand[2] + 0.8  # 0.2, 0.9
                tr1 = 0
                tr2 = 0
                shear1 = int(rand[5] * 20 - 10);
                shear2 = int(rand[6] * 20 - 10)
                if rand[7] > 0.5:
                    im = transforms.functional.affine(im, angle=20, translate=[tr1, tr2], scale=scale,
                                                      shear=[shear1, shear2])
                if rand[1] > 0.5:
                    im = torch.from_numpy(np.array(im)[:, ::-1, :, :].copy())  # ZFlip
                im = my_transforms(im)
                if rand[9] < 0.25: 
                    noise = 1*(torch.rand(size = (Z,Y,X)) < (rand[8] * 0.01 + 0.99)) # create a matrix of 0 or 1, with the % of 0 random
                    im = im * noise # set some values to 0 
        
                # im is (1,1,Z,Y,X), we convert it to (1,2,Z,Y,X) binary
                if self.channels == 2:
                    im1, im2 = self.to2Channels(im, ax=0)
                    im3, im4 = self.to2Channels(im, ax=1) #other order
                    X_aug[count, 0, :, :, :] = im1
                    X_aug[count, 1, :, :, :] = im2
                    X_aug[count+1, 0, :, :, :] = im3
                    X_aug[count+1, 1, :, :, :] = im4
                elif self.channels == 1:
                    im1 = self.to1Channel(im, ax=0)
                    im2 = self.to1Channel(im, ax=1) #other order
                    X_aug[count, 0, :, :, :] = im1
                    X_aug[count+1, 0, :, :, :] = im2
                    
                y_aug.append(target)# we do two appends bc we have 2 images added
                y_aug.append(target) 
                count += 2
        if self.channels == 2:
            X_train_fin, y_train_fin = self.dataTo2Channels(X_train, y_train)
        elif self.channels == 1:
            X_train_fin, y_train_fin = self.dataTo1Channel(X_train, y_train)
        
        X_train_aug = torch.cat((X_train_fin, X_aug), 0)  # Concatenates the Train and the Augmentation X
        y_aug_tens = torch.tensor(y_aug)
        y_train_aug = torch.cat((y_train_fin, y_aug_tens), 0)  # Conctatenae the train and the Augmentation y
        y_train_aug = torch.reshape(y_train_aug, shape=(len(y_train_aug), 1))  # reshape for NN format
        if self.pointCloud_bool == True: 
            return X_train_aug.type(torch.int8),  y_train_aug.type(torch.float32) #y neds torch.float32 for the networ,, while X_train is changed later for the coords
        else:
            return X_train_aug.type(torch.float32),  y_train_aug.type(torch.float32) 
            
    
    def prepareValData(self, PointCloud):
        """
        Due to lack of data, we also here add data augmentations to the validation set, just rotation, etc that don't damadge the image ( eg no noise addition)
        However, we don't upsample the negative as it is done in the training augmentation
        We don't do augmentations here for the PointCloud, as they are done after taking the coordinates.
        """
        X_val = self.X_val
        y_val = self.y_val
        
        Z, Y, X = np.shape(X_val[0, 0, :, :, :])

        my_transforms = transforms.Compose([
            # SYMMETRIE PAR RAPPORT AU PLAN Z,Y. (valeurs z,y change pas, mais changement pour les x)
            transforms.RandomHorizontalFlip(p=0.5),
            # SYMMETRIE PAR RAPPORT AU PLAN Z,X. (valeurs z,x change pas, mais changement pour les y)
            transforms.RandomVerticalFlip(p=0.5),
        ])
        N = len(y_val)
        zeros = torch.sum(y_val == 0)
        ones = torch.sum(y_val == 1)
        # 3 augmentations for the 0, 5 augmentations for the ones.
        if PointCloud == True:
            augmentations = 0  #No augmentation for PointCloud
        else:
            augmentations = 20 
        X_aug = torch.zeros(size=(N*2*augmentations, self.channels, Z, Y, X),
                            dtype=torch.int8)  # Containing the augmentations
        y_aug = []
        count = 0
        for i in range(N):
            im_ori = torch.clone(X_val[i, :, :, :, :])
            target = y_val[i]
            for j in range(augmentations):  # different nbr of augmentations for 1s and 0s
                rand = torch.rand(size=(11,))
                im = torch.clone(im_ori)
                if rand[0] > 0.5:
                    # Rotation de 90.First transform to do bc of the noise
                    im = transforms.functional.rotate(img=im, angle=90)
                # parameters for the affine transformation
                if self.pointCloud_bool == False: 
                    # Translation along z axis
                    if rand[10] < 0 : 
                        tempo = torch.zeros(size = (1,Z,Y,X), dtype = torch.int8)
                        z_trans = torch.randint(low=-2, high = 3, size = (1,)).item() 
                        if z_trans == 2: #higher by 2
                            tempo[:,:16,:,:] = im[:,2:,:,:]
                        elif z_trans == 1: # higher by 1
                            tempo[:,:17,:,:] = im[:,1:,:,:]
                        elif z_trans == -2: # lower by 2
                            tempo[:,2:,:,:] = im[:,:16,:,:]
                        elif z_trans == -1: # lower by 1
                            tempo[:,1:,:,:] = im[:,:17,:,:]
                        else: # no change
                            tempo[:,:,:,:] = im[:,:,:,:]
                        im = tempo.clone()
                
                scale = 0.4 * rand[2] + 0.8  # 0.2, 0.9
                tr1 = 0
                tr2 = 0
                shear1 = int(rand[5] * 20 - 10);
                shear2 = int(rand[6] * 20 - 10)
                if rand[7] > 0.5:
                    im = transforms.functional.affine(im, angle=20, translate=[tr1, tr2], scale=scale,
                                                      shear=[shear1, shear2])
                if rand[1] > 0.5:
                    im = torch.from_numpy(np.array(im)[:, ::-1, :, :].copy())  # ZFlip
                im = my_transforms(im)
                # im is (1,1,Z,Y,X), we convert it to (1,2,Z,Y,X) binary
                if self.channels == 2:
                    im1, im2 = self.to2Channels(im, ax=0)
                    im3, im4 = self.to2Channels(im, ax=1) #other order
                    X_aug[count, 0, :, :, :] = im1
                    X_aug[count, 1, :, :, :] = im2
                    X_aug[count+1, 0, :, :, :] = im3
                    X_aug[count+1, 1, :, :, :] = im4
                elif self.channels == 1:
                    im1 = self.to1Channel(im, ax=0)
                    im2 = self.to1Channel(im, ax=1) #other order
                    X_aug[count, 0, :, :, :] = im1
                    X_aug[count+1, 0, :, :, :] = im2
                    
                y_aug.append(target)# we do two appends bc we have 2 images added
                y_aug.append(target) 
                count += 2
 
        if self.channels == 2:
            X_val_fin, y_val_fin = self.dataTo2Channels(X_val, y_val)
        elif self.channels == 1:
            X_val_fin, y_val_fin = self.dataTo1Channel(X_val, y_val)

        X_val_aug = torch.cat((X_val_fin, X_aug), 0)  # Concatenates the Train and the Augmentation X
        y_aug_tens = torch.tensor(y_aug)
        y_val_aug = torch.cat((y_val_fin, y_aug_tens), 0)  # Conctatenae the train and the Augmentation y
        y_val_aug = torch.reshape(y_val_aug, shape=(len(y_val_aug), 1))  # reshape for NN format
        
        if self.pointCloud_bool == True: 
            return X_val_aug.type(torch.int8),  y_val_aug.type(torch.float32) #y neds torch.float32 for the networ,, while X_train is changed later
        else:
            return X_val_aug.type(torch.float32),  y_val_aug.type(torch.float32) #y neds torch.float32 for the networ,, while X_train is changed later
        

    def dataTo2Channels(self, tensor_X: torch.Tensor, tensor_Y: torch.Tensor) -> torch.Tensor:
        """
        X: Takes as input a Tensor of size (N,1,Z,Y,X) and returns a (2*N,2,Z,Y,X) tensor,
        making the 2 Channels trandormation for each element, with the two orders
        Y: takes as input a Tensor of size(N,1) and outputs a size of (2*N,1)
        """
        N, _, Z, Y, X = tensor_X.shape
        new_tensor_X = torch.zeros(size=(2*N, 2, Z, Y, X), dtype=torch.int8)  # New tensor with 2 channels
        new_tensor_Y = torch.zeros(size=(2 * N,), dtype=torch.int8)  # New tensor with 2 channels
        count = 0
        for k in range(N):
            im = tensor_X[k, :, :, :, :]
            target = tensor_Y[k]
            im1, im2 = self.to2Channels(
                im, ax = 0)  # create the two binary channels from the first channel having 1 and 2 labels
            im3, im4 = self.to2Channels(
                im, ax=1)  # create the two binary channels from the first channel having 1 and 2 labels
            new_tensor_X[count, 0, :, :, :] = im1
            new_tensor_X[count, 1, :, :, :] = im2
            new_tensor_X[count+1, 0, :, :, :] = im3
            new_tensor_X[count+1, 1, :, :, :] = im4
            new_tensor_Y[count] = target
            new_tensor_Y[count+1] = target
            count +=2
        return new_tensor_X, new_tensor_Y

    def to2Channels(self, tensor: torch.Tensor, ax: int) -> torch.Tensor:
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
        
    def dataTo1Channel(self, tensor_X: torch.Tensor, tensor_Y: torch.Tensor) -> torch.Tensor:
        """
        X: Takes as input a Tensor of size (N,1,Z,Y,X) and returns a (2*N,1,Z,Y,X) tensor,
        with the values -1 and 1 for each object
        Y: takes as input a Tensor of size(N,1) and outputs a size of (2*N,1)
        """
        N, _, Z, Y, X = tensor_X.shape
        new_tensor_X = torch.zeros(size=(2*N, 1, Z, Y, X), dtype=torch.int8)  # New tensor with 2 channels
        new_tensor_Y = torch.zeros(size=(2 * N,), dtype=torch.int8)  # New tensor with 2 channels
        count = 0
        for k in range(N):
            im = tensor_X[k, :, :, :, :]
            target = tensor_Y[k]
            im1 = self.to1Channel(im, ax = 0)  # create the two binary channels from the first channel having 1 and 2 labels
            im2 = self.to1Channel(im, ax=1)  # create the two binary channels from the first channel having 1 and 2 labels
            new_tensor_X[count, 0, :, :, :] = im1
            new_tensor_X[count+1, 0, :, :, :] = im2
            new_tensor_Y[count] = target
            new_tensor_Y[count+1] = target
            count +=2
        return new_tensor_X, new_tensor_Y

    def to1Channel(self, tensor: torch.Tensor, ax: int) -> torch.Tensor:
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

