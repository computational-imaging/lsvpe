'''
Reads all the validation text files of the log folders,
compiles them into a csv
'''

import sys, os, lpips
from argparse import ArgumentParser
import torch.optim
import numpy as np
import pandas as pd
import json
import argparse

# Enable import from parent package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src_1.utils import find_version_number, make_subdirs

np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--date', type=str, default='08-26', help='format MM-DD')
    args = parser.parse_args()
    log_root = f'/home/cindy/coded-deblur/logs/21-10-31-baselines'

    if os.path.exists(f'{log_root}/results.csv'):
        os.remove(f'{log_root}/results.csv')

    folders = sorted([f for f in os.listdir(log_root) if '.csv' not in f])
    print(folders)
    for j, date_model in enumerate(folders):
        print(date_model)

        sub_root = f'{log_root}/{date_model}'

        experiments = sorted([f for f in os.listdir(sub_root) if '.csv' not in f])
        for i, exp in enumerate(experiments):
            exp_root = f'{sub_root}/{exp}'
            version_num = find_version_number(exp_root)
            model_dir = f'{exp_root}/v_{version_num - 1}'  # get the latest version

            with open(f'{exp_root}/args.json', 'rt') as f:
                t_args = argparse.Namespace()
                t_args.__dict__.update(json.load(f))
                args = parser.parse_args(namespace=t_args)
            args.remote = False
            if hasattr(args, 'n2n'):
                args.res2 = args.n2n
            elif not hasattr(args, 'res2'):
                args.res2 = False

            summaries_dir, checkpoints_dir = make_subdirs(model_dir, make_dirs=False)

            results = pd.read_csv(f'{model_dir}/checkpoints/val_results.csv')
            mean_column_names = ['date',
                                 'exp',
                                 'shutter',
                                 'decoder',
                                 'length',
                                 'residual',
                                 'PSNR',
                                 'SSIM',
                                 'LPIPS']
            if not hasattr(args, 'scale'):
                args.scale = False
            results = [date_model,
                       exp,
                       args.shutter,
                       args.decoder,
                       args.length,
                       args.residual,
                       results['PSNR'].iloc[0],
                       results['SSIM'].iloc[0],
                       results['LPIPS'].iloc[0],]
            df = pd.DataFrame(columns=mean_column_names)
            series = pd.Series(results,
                               index=df.columns)
            df = df.append(series, ignore_index=True)

            file_name = f'{log_root}/results.csv'

            if not os.path.isfile(file_name):
                df.to_csv(file_name, header=mean_column_names)
            else:  # else it exists so append without writing the header
                df.to_csv(file_name, mode='a', header=False)
