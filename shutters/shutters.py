import torch
import torch.nn as nn
import os
import shutters.shutter_utils as shutils
device = 'cuda:0'

repo_dir = '/home/cindy/PycharmProjects/coded-deblur-publish/shutters/shutter_templates'

def add_noise(img, exp_time=1, test=False):
    sig_shot_min = 0.0001
    sig_shot_max = 0.01
    sig_read_min = 0.001
    sig_read_max = 0.03
    if test: # keep noise levels fixed when testing
        sig_shot = (sig_shot_max - sig_shot_min) / 2
        sig_read = (sig_read_max - sig_read_min) / 2
    else:
        sig_shot = (sig_shot_min - sig_shot_max) * torch.rand(1, dtype=torch.float32, device=device) + sig_shot_max
        sig_read = (sig_read_min - sig_read_max) * torch.rand(1, dtype=torch.float32, device=device) + sig_read_max

    ratio = exp_time / 8

    # Scale image corresponding to exposure time
    img = img * ratio

    # Add shot noise, must detach or it'll mess with the computational graph
    shot = (img.detach() ** (1/2)) * sig_shot * torch.randn_like(img)

    # Add read noise. Short and long exposures should have the same read noise.
    read = sig_read * torch.randn_like(img)
    return img + shot + read


class Shutter:
    def __new__(cls, shutter_type, block_size, test=False, resume=False, model_dir='', init='even'):
        cls_out = {
            'short': Short,
            'long': Long,
            'quad': Quad,
            'med': Medium,
            'full': Full,
            'uniform': Uniform,
            'poisson': Poisson,
            'nonad': Nonad,
            'lsvpe': LSVPE,
        }[shutter_type]

        return cls_out(block_size, test, resume, model_dir, init)


class ShutterBase(nn.Module):
    def __init__(self, block_size, test=False, resume=False, model_dir='', init='even'):
        super().__init__()
        self.block_size = block_size
        self.test = test
        self.resume = resume
        self.model_dir = os.path.dirname(model_dir) # '21-11-14/21-11-14-net/short'

    def getLength(self):
        raise NotImplementedError('Must implement in derived class')

    def getMeasurementMatrix(self):
        raise NotImplementedError('Must implement in derived class')

    def forward(self, video_block, train=True):
        raise NotImplementedError('Must implement in derived class')

    def post_process(self, measurement, exp_time, test):
        measurement = torch.div(measurement, exp_time)       # 1 1 H W
        measurement = add_noise(measurement, exp_time=exp_time, test=test)
        measurement = torch.clamp(measurement, 0, 1)
        return measurement

    def count_instances(self, lengths, counts):
        flattened_lengths = lengths.reshape(-1, ).type(torch.int8)
        total_counts = torch.bincount(flattened_lengths).cpu()
        for k in range(1, len(total_counts)):
            counts[k] = total_counts[k]
        return counts


class Short(ShutterBase):
    def __init__(self, block_size, test=False, resume=False, model_dir='', init='even'):
        super().__init__(block_size, test, resume, model_dir, init)
        self.block_size = block_size

    def getLength(self):
        return torch.ones((1, 1, self.block_size[-2], self.block_size[-1]), dtype=torch.float32)

    def forward(self, video_block, train=True):
        measurement = video_block[:, :1, ...]
        measurement = self.post_process(measurement, exp_time=1, test=self.test)
        return measurement


class Medium(ShutterBase):
    def __init__(self, block_size, test=False, resume=False, model_dir='', init='even'):
        super().__init__(block_size, test, resume, model_dir, init)
        self.block_size = block_size

    def getLength(self):
        return torch.ones((1, 1, self.block_size[-2], self.block_size[-1]), dtype=torch.float32) * 4

    def forward(self, video_block, train=True):
        measurement = torch.sum(video_block[:, :4, ...], dim=1, keepdim=True)
        measurement = self.post_process(measurement, exp_time=4, test=self.test)
        return measurement


class Long(ShutterBase):
    def __init__(self, block_size, test=False, resume=False, model_dir='', init='even'):
        super().__init__(block_size, test, resume, model_dir, init)
        self.block_size = block_size

    def getLength(self):
        return torch.ones((1, 1, self.block_size[-2], self.block_size[-1]), dtype=torch.float32) * 8

    def forward(self, video_block, train=True):
        measurement = torch.sum(video_block[:, :, ...], dim=1, keepdim=True)
        measurement = self.post_process(measurement, exp_time=8, test=self.test)
        return measurement


class Full(ShutterBase):
    def __init__(self, block_size, test=False, resume=False, model_dir='', init='even'):
        super().__init__(block_size, test, resume, model_dir, init)

    def forward(self, video_block, train=True):
        measurement = video_block[:, :1, ...]
        short = self.post_process(measurement, exp_time=1, test=self.test)

        measurement = torch.sum(video_block[:, :4, ...], dim=1, keepdim=True)
        med = self.post_process(measurement, exp_time=4, test=self.test)

        measurement = torch.sum(video_block[:, :, ...], dim=1, keepdim=True)
        long = self.post_process(measurement, exp_time=8, test=self.test)
        return torch.cat((long, med, short), dim=1)


class Quad(ShutterBase):
    ''' Design consistent with Jiang et al. HDR reconstruction, exposure ratios are 1:4:8'''
    def __init__(self, block_size, test=False, resume=False, model_dir='', init='even'):
        super().__init__(block_size, test, resume, model_dir, init)

        self.shutter = torch.zeros(self.block_size, dtype=torch.float32, device=device)

        half_h = int(self.block_size[1] / 2)
        half_w = int(self.block_size[2] / 2)

        self.shutter[:, 0::2, 0::2] = torch.ones((8, half_h, half_w), device=device)
        self.shutter[:4, 1::2, 0::2] = torch.ones((4, half_h, half_w), device=device)
        self.shutter[:4, 0::2, 1::2] = torch.ones((4, half_h, half_w), device=device)
        self.shutter[:1, 1::2, 1::2] = torch.ones((1, half_h, half_w), device=device)
        self.shutter = self.shutter.unsqueeze(0)

        self.num_frames = torch.ones((self.block_size[1], self.block_size[2]), dtype=torch.float32, device=device)
        self.num_frames[0::2, 0::2] *= 8
        self.num_frames[1::2, 0::2] *= 4
        self.num_frames[0::2, 1::2] *= 4
        self.num_frames = self.num_frames.unsqueeze(0)

    def getLength(self):
        return self.num_frames

    def forward(self, video_block, train=True):
        measurement = torch.mul(self.shutter, video_block)          # 1 8 H W
        measurement = torch.sum(measurement, dim=1, keepdim=True)   # 1 1 H W
        measurement = self.post_process(measurement, exp_time=self.num_frames, test=self.test)
        return measurement


class Uniform(ShutterBase):
    def __init__(self, block_size, test=False, resume=False, model_dir='', init='even'):
        super().__init__(block_size, test, resume, model_dir, init)
        self.lengths = torch.randint(1, block_size[0] + 1, size=(block_size[1], block_size[2]), device=device)
        self.shutter = torch.zeros(block_size, device=device)

        for i in range(block_size[0] + 1):
            (y, x) = (self.lengths == i).nonzero(as_tuple=True)
            self.shutter[:i, y, x] = torch.ones((i, len(y)), device=device)

    def getLength(self):
        return self.lengths

    def forward(self, video_block, train=True):
        measurement = torch.mul(video_block, self.shutter)
        measurement = torch.sum(measurement, dim=1, keepdim=True)
        measurement = self.post_process(measurement, exp_time=self.lengths, test=self.test)
        return measurement


class Poisson(ShutterBase):
    def __init__(self, block_size, test=False, resume=False, model_dir='', init='even'):
        super().__init__(block_size, test, resume, model_dir, init)
        self.lengths = torch.load(f'{repo_dir}/poisson.pt') + 1.0
        if block_size[-1] != 512:
            self.lengths = self.lengths[..., :block_size[-2], :block_size[-1]]
        self.lengths = self.lengths.cuda()

        self.shutter = torch.zeros(block_size, device=device)
        for i in range(block_size[0] + 1):
            (y, x) = (self.lengths == i).nonzero(as_tuple=True)
            self.shutter[:i, y, x] = torch.ones((i, len(y)), device=device)

    def getLength(self):
        return self.lengths

    def forward(self, video_block, train=True):
        measurement = torch.mul(video_block, self.shutter)
        measurement = torch.sum(measurement, dim=1, keepdim=True)
        measurement = self.post_process(measurement, exp_time=self.lengths, test=self.test)
        return measurement


class Nonad(ShutterBase):
    def __init__(self, block_size, test=False, resume=False, model_dir='', init='even'):
        super().__init__(block_size, test, resume, model_dir, init)
        # load preinitialized shutter, must add 1 to get correct lengths between [1-8]
        self.lengths = torch.load(f'{repo_dir}/nonad.pt') + 1.0
        if block_size[-1] != 512:
            self.lengths = self.lengths[..., :block_size[-2], :block_size[-1]]
        self.lengths = self.lengths.cuda()

        self.shutter = torch.zeros(block_size, device=device)
        for i in range(1, block_size[0] + 1):
            (y, x) = (self.lengths == i).nonzero(as_tuple=True)
            self.shutter[:i, y, x] = torch.ones((i, len(y)), device=device)

    def getLength(self):
        return self.lengths

    def forward(self, video_block, train=True):
        measurement = torch.mul(video_block, self.shutter)
        measurement = torch.sum(measurement, dim=1, keepdim=True)
        measurement = self.post_process(measurement, exp_time=self.lengths, test=self.test)
        return measurement


class LSVPE(ShutterBase):
    def __init__(self, block_size, test=False, resume=False, model_dir='', init='even'):
        super().__init__(block_size, test, resume, model_dir, init)
        self.block_size = block_size

        if init == 'even':
            rand_end = self.block_size[0] * torch.rand((self.block_size[1], self.block_size[2]), dtype=torch.float32)
        elif init == 'ones':
            rand_end = torch.ones((self.block_size[1], self.block_size[2]), dtype=torch.float32)
        elif init == 'quad':
            rand_end = torch.ones((self.block_size[1], self.block_size[2]), dtype=torch.float32)
            rand_end[::2, ::2] *= 8.0
            rand_end[1::2, ::2] *= 4.0
            rand_end[::2, 1::2] *= 4.0
        else:
            raise NotImplementedError

        self.end_params = nn.Parameter(rand_end, requires_grad=True)

        self.time_range = torch.arange(0, self.block_size[0], dtype=torch.float32, device=device)[:, None, None]
        self.time_range = self.time_range.repeat(1, self.block_size[1], self.block_size[2])
        self.total_steps = 0
        self.lengths = torch.zeros((self.block_size[1], self.block_size[2]))
        self.total_steps = 0
        self.counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0}

    def getLength(self):
        end_params_int = torch.clamp(self.end_params, 1.0, self.block_size[0])
        shutter = shutils.less_than(self.time_range, end_params_int)
        self.lengths = torch.sum(shutter, dim=0)
        return self.lengths

    def forward(self, video_block, train=False):
        if train:
            self.total_steps += 1

        end_params_int = torch.clamp(self.end_params, 1.0, self.block_size[0])
        shutter = shutils.less_than(self.time_range, end_params_int)

        self.lengths = torch.sum(shutter, dim=0)

        if train and self.total_steps % 200 == 0:
            self.counts = self.count_instances(self.lengths, self.counts)

        measurement = torch.mul(video_block, shutter)
        measurement = torch.sum(measurement, dim=1, keepdim=True)
        measurement = self.post_process(measurement, exp_time=self.lengths, test=self.test)
        return measurement
