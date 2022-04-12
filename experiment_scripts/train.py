import json
import os
import time
from argparse import ArgumentParser
from functools import partial
import numpy as np
import torch.optim
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm
import nets.losses as losses
from src import dataloading, summary_utils, utils, models


utils.seed(123)
torch.set_num_threads(2)
device = 'cuda:0'

def main(args):
    if 'learn' not in args.shutter:
        print('SETTING MLR TO 2E-4 since we are only training decoder')
        args.mlr = 2e-4
    args.slr = float(args.slr)
    args.mlr = float(args.mlr)

    if args.exp_name == '':
        response = input('You did not specify an exp_name, want to do that now? (type name or "N") : ')
        if response.lower() != 'n':
            args.exp_name = response.lower()

    lpips_fn = losses.LPIPS_1D()
    args, dir_name = utils.modify_args(args)

    # define model
    model_dir, version_num = utils.make_model_dir(dir_name, args.test, args.exp_name)

    shutter = models.define_shutter(args.shutter, args, model_dir=model_dir)
    decoder = models.define_decoder(args.decoder, args)
    model = models.define_model(shutter, decoder, args, get_coded=False)

    model.train()
    model.cuda()
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'Num trainable params: {params}')

    summaries_dir, checkpoints_dir = utils.make_subdirs(model_dir)
    writer = SummaryWriter(summaries_dir)
    optim = utils.define_optim(model, args)

    if args.date_resume != '00-00-00':
        print('Loading checkpoint')
        file = [f for f in os.listdir(f'{dir_name}/v_{version_num-1}/checkpoints') if 'best.pth' in f][0]
        fname = f'{dir_name}/v_{version_num - 1}/checkpoints/{file}'
        checkpoint = torch.load(fname)
        try:
            # some baselines won't have optim state dict, doesn't affect training much
            model.load_state_dict(checkpoint['model_state_dict'])
            # optim.load_state_dict(checkpoint['optim_state_dict'])
        except KeyError:
            model.load_state_dict(checkpoint)
    else:
        if not args.test:
            with open(f'{dir_name}/args.json', 'w') as f:
                json.dump(vars(args), f, indent=4)

    train_dataloader = dataloading.loadTrainingDataset(args)
    val_dataloader = dataloading.loadValDataset(args)
    summary_fn = partial(summary_utils.write_summary, args.batch_size, writer, args.shutter)

    scheduler = utils.define_schedule(optim)

    loss_fn = utils.define_loss(args)

    total_time_start = time.time()
    best_val_psnr = 0
    total_steps = 0

    with tqdm(total=len(train_dataloader) * args.max_epochs) as pbar:
        for epoch in range(args.max_epochs):
            for step, (avg, model_input, gt) in enumerate(train_dataloader):

                model_input = model_input.cuda()
                gt = gt.cuda()
                start_time = time.time()

                optim.zero_grad(set_to_none=True)
                restored = model(model_input, train=True)
                train_loss = loss_fn(restored, gt)

                train_loss.backward()
                optim.step()
                pbar.update(1)
                if not total_steps % args.steps_til_summary:
                    writer.add_scalar("total_train_loss", train_loss, total_steps)
                    summary_fn(model, model_input[:4], gt[:4], restored[:4], avg, total_steps, optim)
                    tqdm.write("Epoch %d, Total loss %0.6f, "
                               "iteration time %0.6f s, total time %0.6f min" % (
                                   epoch, train_loss, time.time() - start_time,
                                   (time.time() - total_time_start)/60.0))

                    if val_dataloader is not None:
                        with torch.no_grad():
                            model.eval()
                            val_losses = []
                            val_psnrs = []
                            val_lpips = []
                            val_ssims = []
                            for (avg, model_input, gt) in tqdm(val_dataloader):
                                model_input = model_input.cuda()
                                gt = gt.cuda()
                                restored = model(model_input, train=False)
                                val_loss = loss_fn(restored, gt)
                                psnr = summary_utils.get_psnr(restored, gt)
                                ssim = summary_utils.get_ssim(restored, gt)
                                vlpips = lpips_fn(restored, gt)

                                val_ssims.append(ssim)
                                val_psnrs.append(psnr)
                                val_losses.append(val_loss.item())
                                val_lpips.append(vlpips)

                            summary_utils.write_val_scalars(writer,
                                                            ['psnr', 'ssim', 'lpips', 'loss'],
                                                            [val_psnrs, val_ssims, val_lpips, val_losses],
                                                            total_steps)

                            if np.mean(val_psnrs) > best_val_psnr:
                                print(f'BEST PSNR: '
                                      f'{np.mean(val_psnrs)}, SSIM: {np.mean(val_ssims)}, LPIPS: {np.mean(val_lpips)}')
                                best_val_psnr = np.mean(val_psnrs)
                                utils.save_chkpt(model, optim, checkpoints_dir, epoch=epoch, best=True)
                                utils.save_best_metrics(args, total_steps, epoch, val_psnrs,
                                                        val_ssims, val_lpips, checkpoints_dir)
                        model.train()
                        scheduler.step(val_loss)
                total_steps += 1
        utils.save_chkpt(model, optim, checkpoints_dir, epoch=epoch, final=True)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_root',
                        type=str,
                        default='/home/cindy/PycharmProjects/custom_data')
    parser.add_argument('--log_root',
                        type=str,
                        default='/home/cindy/PycharmProjects/lsvpe/logs')
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