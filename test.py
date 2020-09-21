from models import PSNet as PSNet

import argparse
import time
import csv
import cv2
import numpy as np
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import custom_transforms
from utils import tensor2array
from loss_functions import compute_errors_test
from scannet import ScannetDataset

import os
from path import Path
from scipy.misc import imsave
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='Structure from Motion Learner training on KITTI and CityScapes Dataset',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--sequence-length', type=int, metavar='N', help='sequence length for training', default=2)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--pretrained-dps', dest='pretrained_dps', default=None, metavar='PATH',
                    help='path to pre-trained dpsnet model')
parser.add_argument('--seed', default=0, type=int, help='seed for random functions, and network initialization')
parser.add_argument('--output-dir', default='result', type=str,
                    help='Output directory for saving predictions in a big 3D numpy file')
parser.add_argument('--testlist', default='./dataset/scannet/test_split.txt', type=str,
                    help='Text file indicates input data')
parser.add_argument('--nlabel', type=int, default=64, help='number of label')
parser.add_argument('--mindepth', type=float, default=0.5, help='minimum depth')
parser.add_argument('--maxdepth', type=float, default=10, help='maximum depth')
parser.add_argument('--output-print', action='store_true', help='print output depth')
parser.add_argument('--print-freq', default=1, type=int,
                    metavar='N', help='print frequency')

parser.add_argument('--seq_len', default=3, type=int,
                    help='the length of video sequence')


def main():
    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    normalize = custom_transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                            std=[0.5, 0.5, 0.5])
    valid_transform = custom_transforms.Compose([custom_transforms.ArrayToTensor(), normalize])
    val_set = ScannetDataset(args.data, args.testlist,
                             mode='test',
                             n_frames=args.seq_len,
                             r=args.seq_len * 2,
                             transform=valid_transform)

    print('{} samples found in {} valid scenes'.format(len(val_set), len(val_set.scenes)))
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    dpsnet = PSNet(args.nlabel, args.mindepth).cuda()
    weights = torch.load(args.pretrained_dps)
    dpsnet.load_state_dict(weights['state_dict'])
    dpsnet.eval()

    output_dir = Path(args.output_dir)
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    with torch.no_grad():
        for ii, (tgt_img, ref_imgs, ref_poses, intrinsics, intrinsics_inv, tgt_depth, scene, tgt_filename) in enumerate(
                val_loader):

            tgt_img_var = Variable(tgt_img.cuda())
            ref_imgs_var = [Variable(img.cuda()) for img in ref_imgs]
            ref_poses_var = [Variable(pose.cuda()) for pose in ref_poses]
            intrinsics_var = Variable(intrinsics.cuda())
            intrinsics_inv_var = Variable(intrinsics_inv.cuda())
            tgt_depth_var = Variable(tgt_depth.cuda())

            # compute output
            pose = torch.cat(ref_poses_var, 1)
            start = time.time()
            output_depth = dpsnet(tgt_img_var, ref_imgs_var, pose, intrinsics_var, intrinsics_inv_var)
            elps = time.time() - start
            tgt_disp = args.mindepth * args.nlabel / tgt_depth
            output_disp = args.mindepth * args.nlabel / output_depth

            mask = (tgt_depth <= args.maxdepth) & (tgt_depth >= args.mindepth) & (tgt_depth == tgt_depth)

            output_disp_ = torch.squeeze(output_disp.data.cpu(), 1)
            output_depth_ = torch.squeeze(output_depth.data.cpu(), 1)

            for idx in range(tgt_img_var.shape[0]):
                scene_name = scene[idx]
                rgb_basename = tgt_filename[idx]
                _, img_ext = os.path.splitext(rgb_basename)

                pred_depth_dir = os.path.join(args.output_dir, scene_name, "pred_depth")

                if not os.path.exists(pred_depth_dir):
                    os.makedirs(pred_depth_dir)

                pred_depth = output_depth[idx]

                pred_depth = np.float16(pred_depth.squeeze(1).cpu().numpy())
                pred_depth_filepath = os.path.join(pred_depth_dir,
                                                   rgb_basename.replace("color" + img_ext, "pred_depth.npy"))
                np.save(pred_depth_filepath, pred_depth)

                pred_depth_color = colorize_depth(pred_depth.squeeze(1),
                                                  max_depth=5.0).permute(0, 2, 3, 1).squeeze().cpu().numpy()
                pred_depth_color_filepath = os.path.join(pred_depth_dir,
                                                         rgb_basename.replace("color" + img_ext, "pred_depth.jpg"))
                cv2.imwrite(pred_depth_color_filepath, cv2.cvtColor(np.uint8(pred_depth_color), cv2.COLOR_RGB2BGR))


def colorize_depth(input, max_depth, color_mode=cv2.COLORMAP_RAINBOW):
    input_tensor = input.detach().cpu().numpy()
    normalized = input_tensor / max_depth * 255.0
    normalized = normalized.astype(np.uint8)
    if len(input_tensor.shape) == 3:
        normalized_color = np.zeros((input_tensor.shape[0],
                                     input_tensor.shape[1],
                                     input_tensor.shape[2],
                                     3))
        for i in range(input_tensor.shape[0]):
            normalized_color[i] = cv2.applyColorMap(normalized[i], color_mode)
        return torch.from_numpy(normalized_color).permute(0, 3, 1, 2)
    if len(input_tensor.shape) == 2:
        normalized = cv2.applyColorMap(normalized, color_mode)
        return torch.from_numpy(normalized).permute(2, 0, 1)


if __name__ == '__main__':
    main()
