"""
References
----------
[1] Tran, Du, et al. "Learning spatiotemporal features with 3d convolutional networks."
Proceedings of the IEEE international conference on computer vision. 2015.
"""

import torch.nn as nn


class C3D(nn.Module):
    """
    The C3D network as in video-caffe implementation.
    """

    def __init__(self, n_classes=101):
        super(C3D, self).__init__()

        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1), stride=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1), stride=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1), stride=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1), stride=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1), stride=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.fc6 = nn.Linear(2304, 2048)
        self.fc7 = nn.Linear(2048, 2048)
        self.fc8 = nn.Linear(2048, n_classes)

        self.dropout = nn.Dropout(p=0.5)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

        # Weight init
        for layer in self.modules():
            if isinstance(layer, nn.Conv3d):
                nn.init.normal_(layer.weight, std=0.01)
                nn.init.constant_(layer.bias, val=1)
            elif isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.005)
                nn.init.constant_(layer.bias, val=1)
        nn.init.normal_(self.fc8.weight, std=0.01)
        nn.init.constant_(self.fc8.bias, val=0)

    def forward(self, x):

        h = self.relu(self.conv1(x))
        h = self.pool1(h)

        h = self.relu(self.conv2(h))
        h = self.pool2(h)

        h = self.relu(self.conv3a(h))
        # h = self.relu(self.conv3b(h))
        h = self.pool3(h)

        h = self.relu(self.conv4a(h))
        # h = self.relu(self.conv4b(h))
        h = self.pool4(h)

        h = self.relu(self.conv5a(h))
        # h = self.relu(self.conv5b(h))
        h = self.pool5(h)

        h = h.view(-1, 2304)
        h = self.relu(self.fc6(h))
        h = self.dropout(h)
        h = self.relu(self.fc7(h))
        h = self.dropout(h)

        logits = self.fc8(h)
        return logits
        # probs = self.softmax(logits)
        #
        # return probs

    def extract(self, x, layer='fc6'):
        h = self.relu(self.conv1(x))
        h = self.pool1(h)
        if layer == 'conv1':
            return h

        h = self.relu(self.conv2(h))
        h = self.pool2(h)
        if layer == 'conv2':
            return h

        h = self.relu(self.conv3a(h))
        # h = self.relu(self.conv3b(h))
        h = self.pool3(h)
        if layer == 'conv3':
            return h

        h = self.relu(self.conv4a(h))
        # h = self.relu(self.conv4b(h))
        h = self.pool4(h)
        if layer == 'conv4':
            return h

        h = self.relu(self.conv5a(h))
        # h = self.relu(self.conv5b(h))
        h = self.pool5(h)

        h = h.view(-1, 2304)
        if layer == 'conv5':
            return h
        h = self.relu(self.fc6(h))
        h = self.dropout(h)
        if layer == 'fc6':
            return h
        h = self.relu(self.fc7(h))
        h = self.dropout(h)
        if layer == 'fc7':
            return h

        logits = self.fc8(h)
        if layer == 'fc8':
            return logits
        probs = self.softmax(logits)
        return probs
