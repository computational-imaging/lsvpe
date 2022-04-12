import torch.nn as nn
import shutters.shutters as shutters
from nets.unet import UNet
from nets.mprnet import MPRNet_s2
from nets.transform_modules import TileInterp, TreeMultiRandom
from nets.dncnn import DnCNN, init_weights

device = 'cuda:0'


def define_shutter(shutter_type, args, test=False, model_dir=''):
    return shutters.Shutter(shutter_type=shutter_type, block_size=args.block_size,
                            test=test, resume=args.resume, model_dir=model_dir, init=args.init)


def define_model(shutter, decoder, args, get_coded=False):
    ''' Define any special interpolation modules in between encoder and decoder '''
    if args.shutter in ['short', 'med', 'long', 'full'] or decoder is None or args.interp is None:
        # print('***** No interpolation module added! *****')
        return Model(shutter, decoder, dec_name=args.decoder, get_coded=get_coded)
    elif args.shutter == 'quad' in args.shutter:
        if args.interp == 'bilinear':
            # print('***** Bilinear 2x2 interpolation module added! *****')
            return TileInterpModel(shutter, decoder,
                                   get_coded=get_coded,
                                   shutter_name=args.shutter,
                                   sz=args.block_size[-1])
        elif args.interp == 'scatter':
            # print('***** Scatter 2x2 interpolation module added! *****')
            return TreeModel(shutter,
                             decoder,
                             shutter_name=args.shutter,
                             get_coded=get_coded,
                             sz=args.block_size[-1],
                             k=args.k,
                             p=args.p,
                             num_channels=4)

    elif args.shutter == 'nonad' in args.shutter:
        if args.interp == 'bilinear':
            print('***** Bilinear 3x3 interpolation module added! *****')
            return TileInterpModel(shutter, decoder,
                                   get_coded=get_coded,
                                   shutter_name=args.shutter,
                                   sz=args.block_size[-1])
        elif args.interp == 'scatter':
            print('***** Scatter 3x3 interpolation module added! *****')
            return TreeModel(shutter,
                             decoder,
                             shutter_name=args.shutter,
                             get_coded=get_coded,
                             sz=args.block_size[-1],
                             k=args.k,
                             p=args.p,
                             num_channels=9)
    elif args.interp == 'scatter' and args.shutter in ['lsvpe', 'uniform', 'poisson']:
        return TreeModel(shutter,
                         decoder,
                         shutter_name=args.shutter,
                         get_coded=get_coded,
                         k=args.k,
                         p=args.p,
                         num_channels=9,
                         sz=args.block_size[-1])
    raise NotImplementedError('Interp + Shutter combo has not been implemented')


def define_decoder(model_name, args):
    if args.decoder == 'none':
        return None
    out_ch = 1
    if args.shutter == 'full':
        in_ch = 3
    elif args.shutter in ['short', 'med', 'long'] or args.interp is None:
        in_ch = 1
    elif 'quad' in args.shutter:
        in_ch = 4
    elif 'nonad' in args.shutter:
        in_ch = 9
    elif args.interp == 'scatter':
        in_ch = 9
    else:
        raise NotImplementedError
    if model_name == 'unet':
        return UNet(in_ch=in_ch, out_ch=out_ch, depth=6, wf=5, padding=True, batch_norm=False, up_mode='upconv')
    if model_name == 'mpr':
        return MPRNet_s2(in_c=in_ch)
    if model_name == 'dncnn':
        model = DnCNN(in_nc=in_ch, out_nc=1, nc=64, nb=17, act_mode='BR', shutter_name=args.shutter)
        init_weights(model, init_type='orthogonal', init_bn_type='uniform', gain=0.2)
        return model
    raise NotImplementedError('Model not specified correctly')


class TreeModel(nn.Module):
    def __init__(self, shutter, decoder, get_coded=False,
                 shutter_name=None, sz=512, k=3, p=1, num_channels=8):
        super().__init__()
        self.get_coded = get_coded
        self.shutter = shutter
        self.decoder = decoder
        self.shutter_name = shutter_name
        self.tree = TreeMultiRandom(sz=sz, k=k, p=p, num_channels=num_channels)

    def forward(self, input, train=True):
        coded = self.shutter(input, train=train)
        multi = self.tree(coded, self.shutter.getLength())
        x = self.decoder(multi)
        if self.get_coded:
            return x, coded
        return x

    def forward_using_capture(self, coded):
        multi = self.tree(coded, self.shutter.getLength())
        x = self.decoder(multi)
        if self.get_coded:
            return x, coded
        return x


class TileInterpModel(nn.Module):
    def __init__(self, shutter, decoder, get_coded=False,
                 shutter_name='nonad', sz=512, interp='bilinear'):
        super().__init__()
        self.get_coded = get_coded
        self.shutter = shutter
        self.decoder = decoder

        if 'nonad' in shutter_name:
            self.tile_size = 3
        elif 'quad' in shutter_name:
            self.tile_size = 2
        else:
            raise NotImplementedError
        self.interpolator = TileInterp(shutter_name=shutter_name, tile_size=self.tile_size, sz=sz, interp=interp)

    def forward(self, input, train=True):
        coded = self.shutter(input, train=train)
        multi = self.interpolator(coded)
        x = self.decoder(multi)
        if self.get_coded:
            return x, multi
        return x

    def forward_using_capture(self, coded):
        multi = self.interpolator(coded)
        x = self.decoder(multi)
        if self.get_coded:
            return x, multi
        return x


class Model(nn.Module):
    def __init__(self, shutter, decoder, dec_name=None, get_coded=False):
        super().__init__()
        self.get_coded = get_coded
        self.shutter = shutter
        self.decoder = decoder
        self.dec_name = dec_name

    def forward(self, input, train=True):
        coded = self.shutter(input, train=train)
        if not coded.requires_grad:
            ## needed for computing gradients wrt input for fixed shutters
            coded.requires_grad = True
        if self.decoder is None:
            if self.get_coded:
                return coded, coded
            return coded

        x = self.decoder(coded)
        if self.get_coded:
            return x, coded
        return x

    def forward_using_capture(self, coded):
        x = self.decoder(coded)
        if self.get_coded:
            return x, coded
        return x
