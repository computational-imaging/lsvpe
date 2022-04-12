from argparse import ArgumentParser
import torch
import glob
from dataprep.data_utils import *

REMOTE_DATA_PATH = '/media/data4b/cindy/data'
LOCAL_DATA_PATH = '/home/cindy/PycharmProjects/data'

REMOTE_CUSTOM_PATH = '/media/data4b/cindy/custom_data'
LOCAL_CUSTOM_PATH = '/home/cindy/PycharmProjects/custom_data'


def nfs_annotate(log_root, block_size, data_root, split):
    log_root = f'{log_root}_{block_size[0]}f'
    files = [f for f in glob.glob(f'{data_root}/*') if 'MACOS' not in f]
    save_dir = f'{log_root}/{split}'
    os.makedirs(save_dir, exist_ok=True)

    for i, fpath in enumerate(files):
        subject = fpath.split('/')[-1]
        subfolder_path = f'{fpath}/240/{subject}'

        print(subfolder_path)
        video_seq, num_vids, start_idxs = create_nfs_seq(subfolder_path, block_size, num_vids=20)
        nfs_annotate(video_seq, block_size, subject=subject, start_idxs=start_idxs)


def nfs_init(log_root, block_size, data_root, split):
    log_root = f'{log_root}_{block_size[0]}f'
    subject_folders = [f for f in glob.glob(f'{data_root}/nfs240/*') if 'MACOS' not in f]
    assert len(subject_folders) != 0

    print(len(subject_folders))

    save_dir = f'{log_root}/{split}'

    os.makedirs(save_dir, exist_ok=True)
    save_dict = {}
    subfolder_num = 0
    count = 0
    for i, fpath in enumerate(subject_folders):

        subject = fpath.split('/')[-1]
        count += 1
        print(subject)
        subfolder_path = f'{fpath}/240/{subject}'
        num_vids = len(subfolder_path)
        print(subfolder_path)

        video_seq = create_nfs_seq(subfolder_path, block_size)
        video = convert_nfs_seq(video_seq, subfolder_path, block_size, save_dir, subject=subject)

        print(f'Created video for {subfolder_num}, size={video.shape}')
        vid_dict = save_vid_dict(video, num_items_per_video=num_vids, num_frames=block_size[0])
        save_dict[subfolder_num] = vid_dict
        subfolder_num += 1

    print(f'Saved {count} videos')
    torch.save(save_dict, f'{save_dir}/nfs_block.pt')


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('-l', '--list', help='delimited list input', type=str, default='8,512,512')
    parser.add_argument('--remote', action='store_true')
    parser.add_argument('--dataset', type=str, choices=['nfs', 'gopro', ], default='nfs')

    args = parser.parse_args()

    log_root = f'{REMOTE_CUSTOM_PATH}'
    block_size = [int(item) for item in args.list.split(',')]

    if args.remote:
        log_root = f'{REMOTE_CUSTOM_PATH}/{args.dataset}_block_rgb_{block_size[-1]}'
        data_root = REMOTE_DATA_PATH
    else:
        log_root = f'{LOCAL_CUSTOM_PATH}/{args.dataset}_block_rgb_{block_size[-1]}'
        data_root = LOCAL_DATA_PATH

    nfs_init(log_root=log_root,
             block_size=block_size,
             data_root=data_root,
             split='train')
    nfs_init(log_root=log_root,
               block_size=block_size,
               data_root=data_root,
               split='test')


