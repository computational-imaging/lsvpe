import torch
import skimage.io
from tqdm import tqdm
import numpy as np
import os
import random

random.seed(1234)


def crop_center(img, size_x, size_y):
    _, y, x = img.shape
    startx = x // 2 - (size_x // 2)
    starty = y // 2 - (size_y // 2)
    return img[..., starty:starty + size_y, startx:startx + size_x]


def crop_based_on_idx(img, y, x):
    c, h, w = img.shape
    if h < y + 512:
        y = h - 512
    if w < x + 512:
        x = w - 512
    return img[..., y:y + 512, x:x + 512]


def create_nfs_seq(fpath, block_size, num_vids=None):
    all_frames = sorted(os.listdir(fpath))
    num_frames_to_use = np.minimum(len(all_frames), 3000)
    num_avail_frames = len(all_frames)

    if num_vids is None:
        num_vids = num_frames_to_use // block_size[0]
    start_idxs = []

    video_seq = []
    for i in range(num_vids):
        # random determine where to start the video
        start_idx = random.randint(0, num_avail_frames - block_size[0])
        start_idxs.append(start_idx)
        for j in range(block_size[0]):
            img = skimage.io.imread(f'{fpath}/{all_frames[start_idx + j]}')
            video_seq.append(img)
    return video_seq, num_vids, start_idxs


def convert_nfs_seq(video_seq, df, block_size=[8, 512, 512]):
    # data conversion and create an array containing all the frames
    assert len(video_seq) > 0, 'video seq must contain at least one frame'
    N = len(video_seq)
    C, H, W = 3, block_size[-2], block_size[-1]

    num_vids = len(df)
    video = torch.zeros([N, C, H, W], dtype=torch.float32)

    frame_count = 0
    for i in range(num_vids):
        for j in range(block_size[0]):
            im = torch.from_numpy(video_seq[i * 8 + j]).type(torch.float32)

            im = im.permute(2, 0, 1)  # permute from numpy format to torch format
            im = im / 255.0  # [0,1]
            im = crop_center(im, 512, 512)
            video[frame_count, :, :, :] = im
            frame_count += 1
    return video


def imsave(fname, x, dataformats='CDHW'):
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    if x.ndim == 4:
        if dataformats == 'CDHW':
            x = x.transpose(1, 2, 3, 0)

    if x.ndim == 3:
        x = x.transpose(1, 2, 0)
    skimage.io.imsave(fname, x, check_contrast=False)


def save_vid_dict(video, num_items_per_video=200, num_frames=8):
    vid_dict = {}
    for idx in tqdm(range(num_items_per_video)):
        # chop up block of frames to their idv 8 frame videos
        clip = video[idx * num_frames:idx * num_frames + num_frames, ...].clone()
        vid_dict[idx] = clip
    return vid_dict
