import argparse
from exp import Exp

import warnings
warnings.filterwarnings('ignore')


def create_parser():
    parser = argparse.ArgumentParser()
    # Set-up parameters
    parser.add_argument('--device', default='cuda', type=str, help='Name of device to use for tensor computations (cuda/cpu)')
    parser.add_argument('--res_dir', default='./results', type=str)
    parser.add_argument('--ex_name', default='Debug', type=str)
    parser.add_argument('--use_gpu', default=True, type=bool)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--seed', default=1, type=int)

    # dataset parameters
    parser.add_argument('--batch_size', default=2, type=int, help='Batch size')
    parser.add_argument('--val_batch_size', default=2, type=int, help='Batch size')
    parser.add_argument('--data_root', default='./data/smoking_frames') # todo
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--img_size', default=[256, 320], type=int, nargs='*')

    # model parameters
    parser.add_argument('--model', type=str, default='cycle_gan',
                        help='chooses which model to use. [cycle_gan | pix2pix | test | colorization]')
    parser.add_argument('--input_nc', type=int, default=3,
                        help='# of input image channels: 3 for RGB and 1 for grayscale')
    parser.add_argument('--output_nc', type=int, default=3,
                        help='# of output image channels: 3 for RGB and 1 for grayscale')
    parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in the last conv layer')
    parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in the first conv layer')
    parser.add_argument('--netD', type=str, default='basic',
                        help='specify discriminator architecture [basic | n_layers | pixel]. The basic model is a 70x70 PatchGAN. n_layers allows you to specify the layers in the discriminator')
    parser.add_argument('--netG', type=str, default='resnet_9blocks',
                        help='specify generator architecture [resnet_9blocks | resnet_6blocks | unet_256 | unet_128]') # todo
    parser.add_argument('--n_layers_D', type=int, default=3, help='only used if netD==n_layers')
    parser.add_argument('--norm', type=str, default='instance',
                        help='instance normalization or batch normalization [instance | batch | none]')
    parser.add_argument('--init_type', type=str, default='normal',
                        help='network initialization [normal | xavier | kaiming | orthogonal]')
    parser.add_argument('--init_gain', type=float, default=0.02,
                        help='scaling factor for normal, xavier and orthogonal.')
    parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
    # parser.add_argument('--in_shape', default=[3, 256, 320], type=int, nargs='*') # [10, 1, 64, 64] for mmnist, [4, 2, 32, 32] for taxibj
    # parser.add_argument('--hid_S', default=64, type=int)
    # parser.add_argument('--hid_T', default=256, type=int)
    # parser.add_argument('--N_S', default=4, type=int)
    # parser.add_argument('--N_T', default=8, type=int)
    # parser.add_argument('--groups', default=4, type=int)

    # Training parameters
    parser.add_argument('--is_train', default=True, type=bool)
    parser.add_argument('--epochs', default=500, type=int)
    parser.add_argument('--log_step', default=1, type=int)
    parser.add_argument('--lr', default=0.01, type=float, help='Learning rate')

    # resume
    parser.add_argument('--resume_path', default=None, type=str)
    return parser


if __name__ == '__main__':
    args = create_parser().parse_args()
    # args.in_shape = [3, args.img_size[0], args.img_size[1]]
    config = args.__dict__

    exp = Exp(args)
    exp.train(args)
