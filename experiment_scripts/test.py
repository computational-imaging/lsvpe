import os, lpips
from argparse import ArgumentParser
import torch.optim
import numpy as np
import pandas as pd
from tqdm.autonotebook import tqdm
import torch.nn as nn
import json
import argparse
from src import dataloading, summary_utils, utils, models
import nets.losses as losses

utils.seed(65)

def process_args(args):
    args.remote = False
    args.local = True


def write_to_csv(df, fname, col_names):
    # if file does not exist write header
    if not os.path.isfile(fname):
        df.to_csv(fname, header=col_names)
    else:  # else it exists so append without writing the header
        df.to_csv(fname, mode='a', header=False)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--folder', type=str, default='21-10-31')
    args = parser.parse_args()
    log_root = f'/home/cindy/PycharmProjects/lsvpe/logs/{args.folder}/{args.folder}_unet'

    matched = {'short': ['Short', 1],
               'med': ['Medium', 2],
               'long': ['Long', 3],
               'uniform_scatter': ['Uniform Random-S', 4],
               'poisson_scatter': ['Poisson Random-S', 5],
               'nonad_scatter': ['Nonad-S', 6],
               'nonad_bilinear': ['Nonad-B', 7],
               'quad_scatter': ['Quad-S', 8],
               'quad_bilinear': ['Quad-B', 9],
               'lsvpe': ['L-SVPE', 10],
               'full': ['Full', 11],
               }

    exp_folders = sorted([f for f in os.listdir(log_root) if '.csv' not in f])
    lpips_fn = losses.LPIPS_1D()
    l1_loss = nn.L1Loss()

    for i, exp in enumerate(exp_folders):
        exp_root = f'{log_root}/{exp}'

        version_num = utils.find_version_number(exp_root)
        model_dir = f'{exp_root}/v_{version_num - 1}'  # get the latest version
        print(model_dir)

        with open(f'{exp_root}/args.json', 'rt') as f:
            t_args = argparse.Namespace()
            t_args.__dict__.update(json.load(f))
            args = parser.parse_args(namespace=t_args)
        process_args(args, exp_root)

        if exp not in matched.keys():
            index = 0
        else:
            index = matched[exp][1]  # get sorting index

        summaries_dir, checkpoints_dir = utils.make_subdirs(model_dir, make_dirs=False)

        # define model
        args.resume = '11-00'
        shutter = models.define_shutter(args.shutter, args, model_dir=model_dir)
        decoder = models.define_decoder(args.decoder, args)
        model = models.define_model(shutter, decoder, args, get_coded=False)

        model.cuda()
        model.eval()

        checkpoint = torch.load(f'{model_dir}/checkpoints/model_best.pth')
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
        except KeyError:
            model.load_state_dict(checkpoint)
        dfs = []

        if i == 0:
            val_dataloader = dataloading.loadValDataset(args)

        loss_postnet = []
        psnr_postnet = []
        ssim_postnet = []
        count = 0
        with torch.no_grad():
            for (avg, model_input, gt) in tqdm(val_dataloader):
                model_input = model_input.cuda()
                gt = gt.cuda()

                deblur = model(model_input)

                loss = lpips_fn(deblur.squeeze(0), gt.squeeze(0))
                loss_postnet.append(loss)
                p = summary_utils.get_psnr(deblur, gt)
                s = summary_utils.get_ssim(deblur, gt)

                psnr_postnet.append(p)
                ssim_postnet.append(s)

                single_column_names = ['exp',
                                       'index',
                                       'shutter',
                                       'decoder',
                                       'interpolation',
                                       'loss_reg',
                                       'PSNR',
                                       'SSIM',
                                       'LPIPS']
                df = pd.DataFrame(columns=single_column_names)
                series = pd.Series([exp,
                                    index,
                                    matched[exp][0],
                                    args.decoder,
                                    args.interp,
                                    args.reg,
                                    round(p, 3),
                                    round(s, 3),
                                    round(loss, 3)],
                                   index=df.columns)
                df = df.append(series, ignore_index=True)

                file_name = f'{log_root}/idv_results.csv'
                write_to_csv(df, file_name, single_column_names)
                count += 1

        mean_column_names = ['exp',
                             'shutter',
                             'decoder',
                             'interpolation',
                             'loss_reg',
                             'PSNR',
                             'SSIM',
                             'LPIPS']
        df = pd.DataFrame(columns=mean_column_names)
        results = [exp,
                   matched[exp][0],
                   args.decoder,
                   args.interp,
                   args.add_ff,
                   args.b_scale,
                   args.reg,
                   np.round(np.mean(psnr_postnet), decimals=3),
                   np.round(np.mean(ssim_postnet), decimals=3),
                   np.round(np.mean(loss_postnet), decimals=3),
                   ]
        series = pd.Series(results, index=df.columns)
        df = df.append(series, ignore_index=True)

        file_name = f'{log_root}/mean_results.csv'
        write_to_csv(df, file_name, mean_column_names)
