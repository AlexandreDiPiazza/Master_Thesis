from torch import nn

class CNN(nn.Module):
    def __init__(self, channel):
        super(CNN, self).__init__()
        self.first_channel = channel
        self.flatten = nn.Flatten()
        self.conv_layer1 = self._conv_layer_set(self.first_channel, 8, pool_kernel = (1,2,2), k1 = (3,3,3), k2 =(3,3,3))
        self.conv_layer2 = self._conv_layer_set(8, 16,pool_kernel = (1,2,2), k1 = (3,3,3), k2 =(3,3,3))
        self.conv_layer3 = self._conv_layer_set(16,32, pool_kernel = (2,2,2), k1 = (3,3,3), k2 =(3,3,3))
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
      
        self.activation = nn.LeakyReLU(negative_slope=0.001, inplace=True),
        self.Dropout = nn.Dropout(p=0.2)
        self.Linear1 = nn.Linear(21600, 512, bias=True)
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
        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        out = self.conv_layer3(out)
        #out = self.conv_layer4(out)
        out = self.flatten(out)
        #out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.Linear1(out)
        out = self.Dropout(out)
        out = self.Linear2(out)
        out = self.sigmoid(out)
        return out
    
   