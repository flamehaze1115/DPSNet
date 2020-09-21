import numpy as np
import os
import cv2
import re
import csv
import glob
import random
import pickle

import torch.utils.data as data
import torch

from scipy import interpolate

import re


def _read_split_file(filepath):
    '''
    Read data split txt file provided for Robust Vision
    '''
    with open(filepath) as f:
        trajs = f.readlines()
    trajs = [x.strip() for x in trajs]
    return trajs


def fill_depth(depth):
    x, y = np.meshgrid(np.arange(depth.shape[1]).astype("float32"),
                       np.arange(depth.shape[0]).astype("float32"))
    xx = x[depth > 0]
    yy = y[depth > 0]
    zz = depth[depth > 0]

    grid = interpolate.griddata((xx, yy), zz.ravel(),
                                (x, y), method='nearest')
    return grid


def augument(images):
    # randomly shift gamma

    random_gamma = np.random.uniform(0.9, 1.1, size=1)
    images = 255.0 * ((images / 255.0) ** random_gamma)

    # randomly shift brightness
    random_brightness = np.random.uniform(0.8, 1.2, size=1)
    images *= random_brightness

    # randomly shift color
    random_colors = np.random.uniform(0.8, 1.2, size=[3])
    images *= np.reshape(random_colors, [1, 1, 1, 3])

    images = np.clip(images, 0.0, 255.0)

    return images


class ScannetDataset(data.Dataset):
    def __init__(self, dataset_path, split_txt=None, n_frames=3, r=10,
                 mode='train', reloadscan=False, transform=None):
        super(ScannetDataset, self).__init__()
        self.dataset_path = dataset_path
        self.n_frames = n_frames

        self.radius = r

        self.transform = transform

        self.reloadscan = reloadscan

        self.mode = mode  # train or test

        if os.path.exists(split_txt):
            self.scenes = _read_split_file(split_txt)
        else:
            self.scenes = sorted(os.listdir(self.dataset_path))

        if self.mode == 'train':
            self.build_dataset_index_train(r=r)
        elif self.mode == 'test':
            self.build_dataset_index_test(r=r)

        if self.mode == 'train':
            random.shuffle(self.dataset_index)

        self.cam_intr = torch.tensor([[577.87, 0, 319.5],
                                      [0, 577.87, 239.5],
                                      [0, 0, 1]]).to(torch.float32)

    def __len__(self):
        return len(self.dataset_index)

    def shape(self):
        return [self.n_frames, self.height, self.width]

    def read_sample_train(self, index):
        data_blob = self.dataset_index[index]
        num_frames = data_blob['n_frames']
        num_samples = self.n_frames - 1

        frameid = data_blob['id']

        intervals = np.random.randint(2, 6, size=[num_samples])
        inds = []

        first_idx = 0
        tmp = first_idx
        for it in intervals:
            tmp = tmp + it
            inds.append(tmp)

        inds.append(first_idx)
        inds = sorted(inds)  # the middle three should be target images

        images = []
        images_paths = []
        img_ids = []
        for i in inds:
            image = cv2.imread(data_blob['images'][i])

            img_id = re.findall(r'\d+', os.path.basename(data_blob['images'][i]))
            img_ids.append(img_id)

            images_paths.append(data_blob['images'][i])

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images.append(image)

        poses = []
        poses_paths = []
        pose_ids = []
        for i in inds:
            pose = np.loadtxt(data_blob['poses'][i], delimiter=' ').astype(np.float32)

            pose_id = re.findall(r'\d+', os.path.basename(data_blob['poses'][i]))
            pose_ids.append(pose_id)

            poses.append(pose)
            poses_paths.append(data_blob['poses'][i])

        depths = []
        dmasks = []
        depths_paths = []
        depth_ids = []
        for i in inds:
            depth = cv2.imread(data_blob['depths'][i], cv2.IMREAD_ANYDEPTH)

            depth_id = re.findall(r'\d+', os.path.basename(data_blob['depths'][i]))
            depth_ids.append(depth_id)

            depth = (depth.astype(np.float32)) / 1000.0

            dmask = (depth >= self.depth_min) & (depth <= self.depth_max) & (np.isfinite(depth))
            depth[~dmask] = 0

            ratio = np.sum(np.float32(dmask)) / (self.width * self.height)

            assert ratio > 0.5

            depths.append(depth)
            dmasks.append(dmask)
            depths_paths.append(data_blob['depths'][i])

        images = np.stack(images, axis=0).astype(np.float32)
        poses = np.stack(poses, axis=0).astype(np.float32)

        assert np.all(np.isfinite(poses))
        assert (img_ids == pose_ids) & (img_ids == depth_ids)

        depths = np.stack(depths, axis=0).astype(np.float32)
        dmasks = np.stack(dmasks, axis=0)

        return images, poses, depths, dmasks, frameid, images_paths, data_blob['scene']

    def read_sample_test(self, index):
        data_blob = self.dataset_index[index]
        num_frames = data_blob['n_frames']
        num_samples = self.n_frames - 1

        frameid = data_blob['id']

        inds = [i for i in range(self.n_frames)]  # the middle three should be target images

        images = []
        images_paths = []
        img_ids = []
        for i in inds:
            image = cv2.imread(data_blob['images'][i])

            img_id = re.findall(r'\d+', os.path.basename(data_blob['images'][i]))
            img_ids.append(img_id)

            images_paths.append(data_blob['images'][i])

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images.append(image)

        poses = []
        poses_paths = []
        pose_ids = []
        for i in inds:
            pose = np.loadtxt(data_blob['poses'][i], delimiter=' ').astype(np.float32)

            pose_id = re.findall(r'\d+', os.path.basename(data_blob['poses'][i]))
            pose_ids.append(pose_id)

            poses.append(pose)
            poses_paths.append(data_blob['poses'][i])

        depths = []
        dmasks = []
        depths_paths = []
        depth_ids = []
        for i in inds:
            depth = cv2.imread(data_blob['depths'][i], cv2.IMREAD_ANYDEPTH)

            depth_id = re.findall(r'\d+', os.path.basename(data_blob['depths'][i]))
            depth_ids.append(depth_id)

            depth = (depth.astype(np.float32)) / 1000.0

            dmask = (depth >= self.depth_min) & (depth <= self.depth_max) & (np.isfinite(depth))
            depth[~dmask] = 0

            ratio = np.sum(np.float32(dmask)) / (self.width * self.height)

            assert ratio > 0.5

            depths.append(depth)
            dmasks.append(dmask)
            depths_paths.append(data_blob['depths'][i])

        images = np.stack(images, axis=0).astype(np.float32)
        poses = np.stack(poses, axis=0).astype(np.float32)

        assert np.all(np.isfinite(poses))
        assert (img_ids == pose_ids) & (img_ids == depth_ids)

        depths = np.stack(depths, axis=0).astype(np.float32)
        dmasks = np.stack(dmasks, axis=0)

        return images, poses, depths, dmasks, frameid, images_paths, data_blob['scene']

    def split_tgt_ref(self, images, poses, depths, dmasks, images_paths):
        """
        here reference image means source images
        :param images:
        :param poses:
        :param depths:
        :param dmasks:
        :param images_paths:
        :return:
        """
        N_images = images.shape[0]
        tgt_img = images[N_images // 2]
        ref_imgs = [images[i] for i in range(N_images) if i != N_images // 2]
        pose_tgt = poses[N_images // 2]
        ref_poses = [poses[i] @ np.linalg.inv(pose_tgt) for i in range(N_images) if i != N_images // 2]
        tgt_depth = depths[N_images // 2]
        tgt_dmask = dmasks[N_images // 2]

        return tgt_img, ref_imgs, ref_poses, tgt_depth, os.path.basename(images_paths[N_images // 2])

    def __getitem__(self, index):

        flag = True
        while flag:
            try:
                if self.mode == "train":
                    images, poses, depths, dmasks, frameid, images_paths, scene = self.read_sample_train(index)
                elif self.mode == "test":
                    images, poses, depths, dmasks, frameid, images_paths, scene = self.read_sample_test(index)
                flag = False
            except:

                tmp = np.random.randint(0, self.__len__(), 1)[0]
                print("data load error!", index, "use:  ", tmp)
                index = tmp

        tgt_img, ref_imgs, ref_poses, tgt_depth, tgt_filename = self.split_tgt_ref(images, poses, depths, dmasks)

        if self.transform is not None:
            imgs, tgt_depth, intrinsics = self.transform([tgt_img] + ref_imgs, tgt_depth, np.copy(self.cam_intr))
            tgt_img = imgs[0]
            ref_imgs = imgs[1:]
        else:
            intrinsics = np.copy(self.cam_intr)

        if self.mode == "train":
            return tgt_img, ref_imgs, ref_poses, intrinsics, np.linalg.inv(intrinsics), tgt_depth
        else:
            return tgt_img, ref_imgs, ref_poses, intrinsics, np.linalg.inv(intrinsics), tgt_depth, scene, tgt_filename

    def _load_scan(self, scan, interval, if_dump=True):
        """

        :param scan:
        :param interval: 2 if train mode; 10 if test mode
        :return:
        """
        scan_path = os.path.join(self.dataset_path, scan)

        datum_file = os.path.join(scan_path, 'scene.npy')

        # really need to sample scene more densely (skip every 2 frames not 4)

        if (not os.path.exists(datum_file)) or self.reloadscan:
            print("load ", datum_file, self.reloadscan, type(self.reloadscan))
            imfiles = glob.glob(os.path.join(scan_path, 'pose', '*.txt'))
            ixs = sorted([int(os.path.basename(x).split('.')[0]) for x in imfiles])

            poses = []
            for i in ixs[::interval]:
                posefile = os.path.join(scan_path, 'pose', '%d.txt' % i)
                pose = np.loadtxt(posefile, delimiter=' ').astype(np.float32)

                if ~np.all(np.isfinite(pose)):
                    break
                else:
                    poses.append(posefile)

            images = []
            for i in ixs[::interval]:
                imfile = os.path.join(scan_path, 'rgb', '%d.jpg' % i)
                images.append(imfile)

            depths = []
            for i in ixs[::interval]:
                depthfile = os.path.join(scan_path, 'depth', '%d.png' % i)
                depths.append(depthfile)

            valid_num = len(poses)

            scene_info = {
                "images": images[:valid_num],
                "depths": depths[:valid_num],
                "poses": poses
            }

            if if_dump:
                np.save(datum_file, scene_info)
            return scene_info

        else:
            return np.load(datum_file, allow_pickle=True).item()

    def build_dataset_index_train(self, r=4, skip=12):
        self.dataset_index = []
        data_id = 0

        for scan in self.scenes:
            # print(scan, len(self.dataset_index))
            # if len(self.dataset_index) > 10000:
            #     break

            scanid = int(re.findall(r'scene(.+?)_', scan)[0])

            scene_info = self._load_scan(scan, interval=2)
            images = scene_info["images"]
            depths = scene_info["depths"]
            poses = scene_info["poses"]

            for i in range(r, len(images) - r, skip):
                training_example = {}
                training_example['depths'] = depths[i - r:i + r + 1]
                training_example['images'] = images[i - r:i + r + 1]
                training_example['poses'] = poses[i - r:i + r + 1]
                training_example['n_frames'] = 2 * r + 1
                training_example['id'] = data_id
                training_example['scene'] = scan

                self.dataset_index.append(training_example)
                data_id += 1

    def build_dataset_index_test(self):
        self.dataset_index = []
        data_id = 0

        for scan in self.scenes:

            scanid = int(re.findall(r'scene(.+?)_', scan)[0])

            scene_info = self._load_scan(scan, interval=10, if_dump=False)
            images = scene_info["images"]
            depths = scene_info["depths"]
            poses = scene_info["poses"]

            for i in range(0, len(images) - self.n_frames, 1):
                training_example = {}
                training_example['depths'] = depths[i: i + self.n_frames + 1]
                training_example['images'] = images[i: i + self.n_frames + 1]
                training_example['poses'] = poses[i: i + self.n_frames + 1]
                training_example['n_frames'] = self.n_frames
                training_example['id'] = data_id
                training_example['scene'] = scan

                self.dataset_index.append(training_example)
                data_id += 1
