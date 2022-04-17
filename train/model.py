import numpy as np
import h5py
import torch
import torch.utils.data as data
import torch.nn.functional as F
import torch.nn as nn

L = 32
# convolution window size in residual units
W = np.asarray([11, 11, 11, 11, 11, 11, 11, 11,
                21, 21, 21, 21, 41, 41, 41, 41])
# atrous rate in residual units
AR = np.asarray([1, 1, 1, 1, 4, 4, 4, 4,
                 10, 10, 10, 10, 25, 25, 25, 25])


class H5Dataset(data.Dataset):
    def __init__(self, file_path):
        super(H5Dataset, self).__init__()
        self.fp = file_path
        self.data = h5py.File(file_path, 'r', libver='latest')
        self.ctr = 0

    def __getitem__(self, idx):
        self.ctr += 1
        if self.ctr % 100000 == 0: # workaround to avoid memory issues
            self.data.close()
            self.data = h5py.File(self.fp, 'r', libver='latest')
        X = torch.Tensor(self.data['X' + str(idx)][:].T)
        Y = torch.Tensor(self.data['Y' + str(idx)][:].T)
        return (X, Y)

    def __len__(self):
        assert len(self.data) % 3 == 0
        return len(self.data) // 3


class ResBlock(nn.Module):
    def __init__(self, L, W, AR, pad=True):
        super(ResBlock, self).__init__()
        self.bn1 = nn.BatchNorm1d(L)
        s = 1  # stride
        # padding calculation: https://discuss.pytorch.org/t/how-to-keep-the-shape-of-input-and-output-same-when-dilation-conv/14338/2
        if pad:
            padding = int(1 / 2 * (1 - L + AR * (W - 1) - s + L * s))
        else:
            padding = 0
        self.conv1 = nn.Conv1d(L, L, W, dilation=AR, padding=padding)
        self.bn2 = nn.BatchNorm1d(L)
        self.conv2 = nn.Conv1d(L, L, W, dilation=AR, padding=padding)

    def forward(self, x):
        out = self.bn1(x)
        out = torch.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = torch.relu(out)
        out = self.conv2(out)
        out = out + x
        return out


class Pangolin(nn.Module):
    def __init__(self, L, W, AR):
        super(Pangolin, self).__init__()
        self.n_chans = L
        self.conv1 = nn.Conv1d(4, L, 1)  # in_channels, out_channels, kernel_size
        self.skip = nn.Conv1d(L, L, 1)
        self.resblocks, self.convs = nn.ModuleList(), nn.ModuleList()
        for i in range(len(W)):
            self.resblocks.append(ResBlock(L, W[i], AR[i]))
            if (((i + 1) % 4 == 0) or ((i + 1) == len(W))):
                self.convs.append(nn.Conv1d(L, L, 1))
        self.conv_last1 = nn.Conv1d(L, 2, 1)
        self.conv_last2 = nn.Conv1d(L, 1, 1)
        self.conv_last3 = nn.Conv1d(L, 2, 1)
        self.conv_last4 = nn.Conv1d(L, 1, 1)
        self.conv_last5 = nn.Conv1d(L, 2, 1)
        self.conv_last6 = nn.Conv1d(L, 1, 1)
        self.conv_last7 = nn.Conv1d(L, 2, 1)
        self.conv_last8 = nn.Conv1d(L, 1, 1)

    def forward(self, x):
        conv = self.conv1(x)
        skip = self.skip(conv)
        j = 0
        for i in range(len(W)):
            conv = self.resblocks[i](conv)
            if (((i + 1) % 4 == 0) or ((i + 1) == len(W))):
                dense = self.convs[j](conv)
                j += 1
                skip = skip + dense
        CL = 2 * np.sum(AR * (W - 1))
        skip = F.pad(skip, (-CL // 2, -CL // 2))
        out1 = F.softmax(self.conv_last1(skip), dim=1)
        out2 = torch.sigmoid(self.conv_last2(skip))
        out3 = F.softmax(self.conv_last3(skip), dim=1)
        out4 = torch.sigmoid(self.conv_last4(skip))
        out5 = F.softmax(self.conv_last5(skip), dim=1)
        out6 = torch.sigmoid(self.conv_last6(skip))
        out7 = F.softmax(self.conv_last7(skip), dim=1)
        out8 = torch.sigmoid(self.conv_last8(skip))
        return torch.cat([out1, out2, out3, out4, out5, out6, out7, out8], 1)


