import torch
from torch import nn


class SiameseCNN(nn.Module):
    def __init__(self):
        super(SiameseCNN, self).__init__()
        self.flatten = nn.Flatten()
        self.conv_layer1 = self._conv_layer_set(1, 8, pool_kernel = (1,2,2), k1 = (3,3,3), k2 =(3,3,3))
        self.conv_layer2 = self._conv_layer_set(8, 16,pool_kernel = (1,2,2), k1 = (3,3,3), k2 =(3,3,3))
        self.conv_layer3 = self._conv_layer_set(16,32, pool_kernel = (2,2,2), k1 = (3,3,3), k2 =(3,3,3))
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
      
        self.activation = nn.LeakyReLU(negative_slope=0.001, inplace=True),
        self.Dropout = nn.Dropout(p=0.2)
        self.Linear1 = nn.Linear(21600 , 512, bias=True)
        self.Linear2 = nn.Linear(512,1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def _conv_layer_set(self, in_c, out_c, pool_kernel, k1, k2):
        conv_layer = nn.Sequential(
            nn.Conv3d(in_channels=in_c, out_channels=out_c, kernel_size=k1, stride=(1, 1, 1),
                      padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None),
            nn.LeakyReLU(negative_slope=0.001, inplace=False),
            nn.Conv3d(in_channels=out_c, out_channels=out_c, kernel_size=k2, stride=(1, 1, 1),
                      padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None),
            nn.LeakyReLU(negative_slope=0.001, inplace=True),
            nn.MaxPool3d(kernel_size=pool_kernel, stride=None, padding=0, dilation=1, return_indices=False,
                         ceil_mode=False),
            nn.Dropout(p=0.2)
        )
        return conv_layer

    def forward(self, x):
        batch_size = x.shape[0]
        Z = x.shape[2] ; Y = x.shape[3] ; X = x.shape[4]
        # First image of the Siamese
        out1 = self.conv_layer1(x[:,0,:,:,:].view(batch_size,1,Z,Y,X)) # we take the first channel
        out1 = self.conv_layer2(out1)
        out1 = self.conv_layer3(out1)
        out1 = self.flatten(out1)
        out1 = out1.view(out1.size(0), -1)
        
        # Second image of the Siamese
        out2 = self.conv_layer1(x[:,1,:,:,:].view(batch_size,1,Z,Y,X)) # we take the second channel
        out2 = self.conv_layer2(out2)
        out2 = self.conv_layer3(out2)
        out2 = self.flatten(out2)
        out2 = out2.view(out2.size(0), -1)
     
        #We concatenate the two
        out = torch.abs(out1-out2)
       
        out = self.Linear1(out)
        out = self.Dropout(out)
        out = self.Linear2(out)
        out = self.sigmoid(out)
        return out