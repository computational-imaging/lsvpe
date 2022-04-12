import os, lpips
from argparse import ArgumentParser
import torch.optim
import pandas as pd
from tqdm.autonotebook import tqdm
import nets.losses as losses
from src import dataloading, summary_utils, utils, models
import numpy as np


utils.seed(123)
torch.set_num_threads(2)
device = 'cuda:0'


def main(args):
    args.exp_name = 'burst'
    args.local = True
    args.shutter = 'burst'
    args.decoder = 'none'

    log_root = f'/home/cindy/PycharmProjects/coded-deblur/logs'

    if os.path.exists(f'{log_root}/burst_idv_results.csv'):
        os.remove(f'{log_root}/burst_idv_results.csv')

    custom_loss_fn = losses.LPIPS_1D()
    args, exp_name = utils.modify_args(args)

    # define model
    shutter = models.define_shutter(args.shutter, args)
    decoder = None
    model = models.define_model(shutter, decoder, args, get_coded=False)

    model.eval()
    model.cuda()

    val_dataloader = dataloading.loadValDataset(args)

    loss_fn = utils.define_loss(args)

    single_column_names = ['exp',
                           'index',
                           'shutter',
                           'decoder',
                           'interpolation',
                           'loss_reg',
                           'PSNR',
                           'SSIM',
                           'LPIPS']
    with torch.no_grad():
        model.eval()

        val_losses = []
        val_psnrs = []
        val_lpips = []

        val_ssims = []
        for (avg, model_input, gt) in tqdm(val_dataloader):
            model_input = model_input.cuda()
            gt = gt.cuda()
            restored = model(model_input)
            val_loss = loss_fn(restored, gt)

            psnr = summary_utils.get_psnr(restored, gt)
            ssim = summary_utils.get_ssim(restored, gt)
            vlpips = custom_loss_fn(restored, gt)

            val_ssims.append(ssim)
            val_psnrs.append(psnr)
            val_losses.append(val_loss.item())
            val_lpips.append(vlpips)

            df = pd.DataFrame(columns=single_column_names)
            series = pd.Series(['burst_avg',
                                0,
                                'Burst',
                                'none',
                                'none',
                                0,
                                round(psnr, 3),
                                round(ssim, 3),
                                round(vlpips, 3)],
                               index=df.columns)
            df = df.append(series, ignore_index=True)

            file_name = f'{log_root}/burst_idv_results.csv'
            # if file does not exist write header
            if not os.path.isfile(file_name):
                df.to_csv(file_name, header=single_column_names)
            else:  # else it exists so append without writing the header
                df.to_csv(file_name, mode='a', header=False)

        mean_column_names = ['exp',
                             'shutter',
                             'decoder',
                             'interpolation',
                             'loss_reg',
                             'PSNR',
                             'SSIM',
                             'LPIPS']
        df = pd.DataFrame(columns=mean_column_names)
        results = ['00-00',
                   'Burst Avg',
                   'None',
                   args.interp,
                   args.reg,
                   np.round(np.mean(val_psnrs), decimals=3),
                   np.round(np.mean(val_ssims), decimals=3),
                   np.round(np.mean(val_lpips), decimals=3),
                   ]
        series = pd.Series(results, index=df.columns)
        df = df.append(series, ignore_index=True)

        file_name = f'{log_root}/burst_mean_results.csv'
        # if file does not exist write header
        if not os.path.isfile(file_name):
            df.to_csv(file_name, header=mean_column_names)
        else:  # else it exists so append without writing the header
            df.to_csv(file_name, mode='a', header=False)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_root',
                        type=str,
                        default='/home/cindy/PycharmProjects/custom_data')
    parser.add_argument('--log_root',
                        type=str,
                        default='/home/cindy/PycharmProjects/coded-deblur-publish/logs')
    parser.add_argument('--test', action='store_true',
                        help='dummy experiment name for just testing code')
    parser.add_argument('-b', '--block_size',
                        help='delimited list input for block size in format %,%,%',
                        default='8,512,512')
    parser.add_argument('--resume',
                        type=str,
                        default='00-00-00',
                        help='date of folder of exp to resume')
    parser.add_argument('--gt', type=int, default=0)
    parser.add_argument('--local', action='store_true')
    parser.add_argument('--scale', type=int, default=0)
    parser.add_argument('--exp_name', type=str, default='')
    parser.add_argument('--reg', type=float, default=100.0, help='regularization on lpips loss')
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--p', type=float, default=1.0)
    parser.add_argument('--max_epochs', type=int, default=6000)

    parser.add_argument('--mlr', help='model_lr', type=str, default='5e-4')
    parser.add_argument('--slr', help='shutter_lr', type=str, default='2e-4')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--steps_til_summary', type=int, default=8000)
    parser.add_argument('--epochs_til_checkpoint', type=int, default=500)
    parser.add_argument('--steps_til_ckpt', type=int, default=5000)
    parser.add_argument('--interp', type=str, required=True,
                        choices=['none', 'bilinear', 'scatter'])
    parser.add_argument('--init', type=str, choices=['even', 'ones', 'quad'], default='quad',
                        help='choose way to initialize learned shutters, '
                             'even=even probabilities on all options,'
                             'ones=use all ones,'
                             'quad=use quad structure')

    parser.add_argument('--loss', type=str, choices=['mpr', 'l1', 'l2_lpips', 'l2'], default='l2_lpips')
    parser.add_argument('--decoder', type=str,
                        choices=['unet', 'mpr', 'dncnn',],
                        default='unet')
    parser.add_argument('--shutter', type=str, required=True)
    parser.add_argument('--sched', type=str, default='reduce')

    args = parser.parse_args()
    args.block_size = [int(item) for item in args.block_size.split(',')]
    main(args)