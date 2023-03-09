from torch import nn 
import timm

class TimmModel(nn.Module):
    def __init__(self, backbone, n_slice_per_c, in_chans, image_size, pretrained=False):
        super(TimmModel, self).__init__()
        self.n_slice_per_c = n_slice_per_c
        self.in_chans = in_chans
        self.image_size = image_size
        self.drop_rate = 0
        self.drop_rate_last = 0.3
        self.drop_path_rate = 0.3
        self.out_dim = 1
        self.sigmoid = nn.Sigmoid()
        self.encoder = timm.create_model(
            backbone,
            in_chans=in_chans,
            num_classes=self.out_dim,
            features_only=False,
            drop_rate=self.drop_rate,
            drop_path_rate=self.drop_path_rate,
            pretrained=pretrained
        )

        if 'efficient' in backbone:
            hdim = self.encoder.conv_head.out_channels
            self.encoder.classifier = nn.Identity()
        elif 'convnext' in backbone:
            hdim = self.encoder.head.fc.in_features
            self.encoder.head.fc = nn.Identity()

        self.lstm = nn.LSTM(hdim, 256, num_layers=2, dropout=self.drop_rate, bidirectional=True, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(self.drop_rate_last),
            nn.LeakyReLU(0.1),
            nn.Linear(256, self.out_dim),
        )
        self.final_loss = nn.LeakyReLU(0.1)
        self.final_layer = nn.Linear(self.n_slice_per_c,1, bias = True)

    def forward(self, x):  # (bs, nslice, ch, sz, sz)
        bs = x.shape[0]
        x = x.view(bs * self.n_slice_per_c, self.in_chans, self.image_size, self.image_size)
        feat = self.encoder(x)
        feat = feat.view(bs, self.n_slice_per_c, -1)
        feat, _ = self.lstm(feat)
        feat = feat.contiguous().view(bs * self.n_slice_per_c, -1)
        feat = self.head(feat)
        feat = feat.view(bs, self.n_slice_per_c).contiguous()
        feat = self.final_loss(feat)
        feat = self.final_layer(feat)
        feat = self.sigmoid(feat)
        return feat