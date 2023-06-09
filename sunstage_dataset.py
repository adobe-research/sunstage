import os
import cv2
import torch
import pickle
import numpy as np

from util import get_camera_dict, convert_pose, get_cam_param
from torch.utils.data import Dataset

class SunStageData(Dataset):
    def __init__(self, opt):
        data_dir = os.path.join(opt.data_dir, opt.obj_name)
        ignore_ids = []
        with open('{}/to_ignore.txt'.format(data_dir), 'r') as f:
            for line in f:
                if '-' not in line:
                    ignore_ids += [line.strip()]
                else:
                    line = line.strip()
                    line = line.split('-')
                    for i in range(int(line[0]), int(line[1]) + 1):
                        ignore_ids += ['{:04d}'.format(i)]
        with open('{}/video_sections.txt'.format(data_dir), 'r') as f:
            for line in f:
                line = line.split(':')
                if line[0] == '360':
                    n_360 = int(line[1].strip().split('-')[-1])
                    ids_360 = range(int(line[1].strip().split('-')[0]), n_360 + 1)
                elif line[0] == 'mv':
                    n_mv = int(line[1].strip().split('-')[-1])
                    ids_mv = range(int(line[1].strip().split('-')[0]), n_mv + 1)
                elif line[0] == 'bow':
                    n_bow = int(line[1].strip().split('-')[-1])
                    ids_bow = range(int(line[1].strip().split('-')[0]), n_bow + 1)
        ignore_ids = list(set(ignore_ids))
        ignore_ids.sort()

        n_360_valid = n_360
        for i in ignore_ids:
            if int(i) <= n_360:
                n_360_valid -= 1
        for i in ids_bow:
            ignore_ids += ['{:04d}'.format(i)]
        print(ignore_ids)
        print(ids_360, ids_mv, ids_bow)
        print(n_360, n_360_valid)

        ignore_ids = list(set(ignore_ids))
        ignore_ids.sort()
        self.n_360_valid = n_360_valid
        self.n_mv_full = n_360_valid // 20

        self.camera_dict = get_camera_dict(data_dir)
        valid_ids = []
        for i in range(len(self.camera_dict)):
            img_id = '{:04d}'.format(i + 1)
            if img_id not in ignore_ids:
                valid_ids += [img_id]
        self.valid_ids = valid_ids
        self.data_dir = data_dir
        self.landmarks2d = torch.load('{}/predictions.pth'.format(self.data_dir))

        self.img_info = {}
        for img_id in self.valid_ids:
            cam_R = self.get_camR(f'{img_id}.png')
            focal_length, principal_point, image_size = get_cam_param(img_id, self.camera_dict, self.data_dir)
            exp, pose = self.load_data(img_id)

            self.img_info[img_id] = {'cam_R': cam_R,
                                     'focal_length': focal_length,
                                     'principal_point': principal_point,
                                     'image_size': image_size,
                                     'exp': exp,
                                     'pose': pose,
                                     'full_lmk': int(img_id) in ids_360}

    def __len__(self):
        return len(self.valid_ids)

    def load_data(self, img_id):
        with open('{}/deca_out/{}/{}_geo.pkl'.format(self.data_dir, img_id, img_id), 'rb') as f:
            render_data = pickle.load(f)

        exp = torch.from_numpy(render_data['exp'])
        pose = torch.from_numpy(render_data['pose'])
        return exp, pose

    def load_mask(self, img_id):
        mask = cv2.imread('{}/deca_out/{}/{}_mask.png'.format(self.data_dir, img_id, img_id))
        mask = cv2.resize(mask, (224, 224), interpolation=cv2.INTER_NEAREST) / 255.
        mask = torch.from_numpy(mask.astype(np.float32))
        mask = mask.permute(2, 0, 1)[None, 0, ...]

        mask_skin = cv2.imread('{}/test_nohair/{}_parsing.png'.format(self.data_dir, img_id))
        mask_skin = cv2.resize(mask_skin, (224, 224), interpolation=cv2.INTER_NEAREST) / 255.
        mask_skin = torch.from_numpy(mask_skin.astype(np.float32))
        mask_skin = mask_skin.permute(2, 0, 1)[None, 0, ...]
        return 1 - mask, mask_skin

    def load_gt(self, img_id):
        img = cv2.imread('{}/deca_out/{}/{}.png'.format(self.data_dir, img_id, img_id))
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA) / 255.
        img = torch.from_numpy(img.astype(np.float32))
        img = img.permute(2, 0, 1)[None, [2, 1, 0], ...]
        return img

    def load_lmk_gt(self, img_id):
        return self.landmarks2d[int(img_id) - 1, :, :]

    def get_camR(self, img_name):
        img_info = self.camera_dict[img_name]
        pose = np.array(img_info['W2C']).reshape((4, 4))
        R = convert_pose(np.linalg.inv(pose[:3, :3]))
        R = torch.from_numpy(R).float()
        cam_R = R.unsqueeze(0)
        return cam_R

    def __getitem__(self, j):
        img_id = self.valid_ids[j]
        img_dict = self.img_info[img_id]

        mask_bg, mask_fg = self.load_mask(img_id)
        lmk_gt = self.load_lmk_gt(img_id)

        img_dict['img_id'] = img_id
        img_dict['img_gt'] = self.load_gt(img_id)
        img_dict['mask_bg'] = mask_bg
        img_dict['mask_fg'] = mask_fg
        img_dict['lmk_gt'] = lmk_gt
        return img_dict
