import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pykdtree.kdtree import KDTree

device = 'cuda:0'


def norm_coord(coords, length=512):
    ''' Convert image from [0, 512] pixel length to [-1, 1] coords'''
    coords = coords / length
    coords -= 0.5
    coords *= 2
    return coords.detach().cpu().numpy()


class TreeMultiRandom(nn.Module):
    def __init__(self, sz=512, k=3, p=1, num_channels=8):
        super().__init__()

        self.k = k # number of neighboring points to search
        self.p = p
        self.sz = sz
        self.num_channels = num_channels

    def _find_idx(self, mask):
        img = torch.ones((self.sz, self.sz), device=device)
        holed_img = img * mask
        filled_idx = (holed_img != 0).nonzero()
        unfilled_idx = (holed_img == 0).nonzero()

        filled_idx_n = norm_coord(filled_idx, self.sz)  # num_coords, 2
        unfilled_idx_n = norm_coord(unfilled_idx, self.sz)  # num_coords, 2

        tree = KDTree(filled_idx_n)
        dist, idx = tree.query(unfilled_idx_n, k=self.k)

        idx = idx.astype(np.int32)
        dist = torch.from_numpy(dist).cuda()
        return idx, dist, filled_idx, unfilled_idx

    def _fill(self, holed_img, params):
        b, _, _ = holed_img.shape

        idx, dist, filled_idx, unfilled_idx = params
        # idx = num_coords, k
        # filled_idx = num_coords, 2
        vals = torch.zeros((b, dist.shape[0]), dtype=torch.float32, device=device)

        for i in range(self.k):
            # find coords of the points which are knn
            idx_select = filled_idx[idx[:, i]]  # num_coords, k

            # add value of those coords, weighted by their inverse distance
            vals += holed_img[:, idx_select[:, 0], idx_select[:, 1]] * (1.0 / dist[:, i]) ** self.p
        vals /= torch.sum((1.0 / dist) ** self.p, dim=1)

        holed_img[:, unfilled_idx[:, 0], unfilled_idx[:, 1]] = vals
        return holed_img

    def forward(self, coded, shutter_len):
        b, _, h, w = coded.shape

        stacked = torch.zeros((b, self.num_channels, h, w), device=device)

        if self.num_channels == 9 or self.num_channels == 4: # learn_all
            unique_lengths = torch.unique(shutter_len).type(torch.int8)
            for i, length in enumerate(unique_lengths):
                mask = (shutter_len == length)          # 512, 512
                holed_img = coded[:, 0, :, :] * mask    # 4, 512, 512 (remove empty axis)
                params = self._find_idx(mask)
                filled_img = self._fill(holed_img, params)
                stacked[:, i, :, :] = filled_img
        else:
            raise NotImplementedError('this has not been implemented')
        return stacked


class TileInterp(nn.Module):
    def __init__(self, shutter_name, tile_size, sz, interp='bilinear'):
        super().__init__()
        self.shutter_name = shutter_name
        self.tile_size = tile_size
        self.sz = sz
        self.interp = interp

    def forward(self, coded):
        b, _, _, _ = coded.shape
        full_stack = torch.zeros((b, self.tile_size ** 2, self.sz, self.sz), dtype=torch.float32, device=device)
        curr_channel = 0
        for i in range(self.tile_size):
            for j in range(self.tile_size):
                # 1,1,H/3,W/3 getting every measurement in the tile
                sampled_meas = coded[:, :, i::self.tile_size, j::self.tile_size]
                full_res = F.interpolate(sampled_meas, size=[self.sz, self.sz], mode='bilinear', align_corners=True)
                full_stack[:, curr_channel, ...] = full_res.squeeze(1)
                curr_channel += 1
        return full_stack
