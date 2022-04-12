import torch
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid


def get_psnr(pred, gt):
    return 10 * torch.log10(1 / torch.mean((pred - gt) ** 2)).detach().cpu().numpy()


def get_ssim(pred, gt):
    ssims = []
    for i in range(pred.shape[0]):
        pred_i = pred[i, :, :, :].permute(1, 2, 0).detach().cpu().numpy()
        gt_i = gt[i, :, :, :].permute(1, 2, 0).detach().cpu().numpy()
        ssims.append(ssim(pred_i, gt_i, multichannel=True))
    return sum(ssims) / len(ssims)


def write_val_scalars(writer, names, values, total_steps):
    for name, val in zip(names, values):
        writer.add_scalar(f'val/{name}', np.mean(val), total_steps)


def write_summary(batch_size, writer, shutter_name, model, input, gt,
                  output, avg, total_steps, optim):
    coded = model.shutter(input)

    cat_input = coded[:, 0, :, :].unsqueeze(0)
    for i in range(1, coded.shape[1]):
        cat_input = torch.cat((cat_input, coded[:, i, ...].unsqueeze(0)), dim=0)

    grid = make_grid(cat_input,
                     scale_each=True, nrow=1, normalize=False).cpu().detach().numpy()
    writer.add_image(f"sensor image", grid, total_steps)

    result_gt = torch.cat((avg, output.cpu(), gt.cpu()), dim=0)
    grid = make_grid(result_gt,
                     scale_each=True,
                     nrow=batch_size,
                     normalize=False).cpu().detach().numpy()
    writer.add_image(f"avg_result_gt", grid, total_steps)

    psnr = get_psnr(output, gt)
    ssim = get_ssim(output, gt)
    writer.add_scalar(f"train/psnr", psnr, total_steps)
    writer.add_scalar(f"train/ssim", ssim, total_steps)
    writer.add_scalar("learning_rate", optim.param_groups[0]['lr'], total_steps)

    if 'learn' in shutter_name:
        fig = plt.figure()
        plt.bar(model.shutter.counts.keys(), model.shutter.counts.values())
        plt.ylabel('counts')
        writer.add_figure(f'lengths_freq', fig, total_steps)

        shutter = model.shutter.lengths.detach().cpu()

        fig = plt.figure()
        plt.imshow(shutter)
        plt.colorbar()
        writer.add_figure(f'train/learned_length', fig, total_steps)
