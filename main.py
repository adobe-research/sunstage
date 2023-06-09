import argparse
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np

data_dir = '/home/yifan1/Desktop/sunstage/data_mobile/dan_1'

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

from FLAME import FLAME

def tensor2image(tensor):
    image = tensor.detach().cpu().numpy()
    image = image*255.
    image = np.maximum(np.minimum(image, 255), 0)
    image = image.transpose(1,2,0)
    return image.astype(np.uint8).copy()


device = 'cuda:0'









def load_data(img_id):
    with open('{}/deca_out/{}/{}_geo.pkl'.format(data_dir, img_id, img_id), 'rb') as f:
        render_data = pickle.load(f)

    shape = torch.from_numpy(render_data['shape']).to(device)
    exp = torch.from_numpy(render_data['exp']).to(device)
    pose = torch.from_numpy(render_data['pose']).to(device)
    cam = torch.from_numpy(render_data['cam']).to(device)
    return shape, exp, pose, cam

import os
import torch

from pytorch3d.utils import ico_sphere
import numpy as np
from tqdm import tqdm

# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes, save_obj

from pytorch3d.loss import (
    chamfer_distance,
    mesh_edge_loss,
    mesh_laplacian_smoothing,
    mesh_normal_consistency,
)








colmap_cameras, colmap_images, colmap_points3D = read_model('{}/0'.format(data_dir), '.txt')
camera_dict = parse_camera_dict(colmap_cameras, colmap_images)

n_img = len(camera_dict)
print(camera_dict['0001.png']['K'])
print(get_cam_param('0001', camera_dict))





def load_match_uv(grids_dict, img_name, match_pts):
    match_pts = torch.from_numpy(match_pts * 2 - 1).float().to(device)
    grid = grids_dict[img_name]
    grid = grid.permute(0, 3, 1, 2)
    match_pts = match_pts[None, :, None, :]
    pts = F.grid_sample(grid, match_pts, align_corners=False)
    return pts


def proj_lmk_scale(verts, scale_factor, T_z, cam_R, camera, full_lmk=True):
    verts = verts.clone()
    verts[:, :, 1:] = -verts[:, :, 1:]
    verts[..., :2] = -verts[..., :2]
    verts = verts.permute(0, 2, 1)
    verts = torch.bmm(cam_R, verts)
    verts = scale_factor * verts
    verts += T_z
    verts = verts.permute(0, 2, 1)

    verts_view = camera.get_world_to_view_transform().transform_points(verts)
    # view to NDC transform
    to_ndc_transform = camera.get_ndc_camera_transform()
    projection_transform = camera.get_projection_transform().compose(to_ndc_transform)
    verts_ndc = projection_transform.transform_points(verts_view)
    verts_ndc[..., 2] = verts_view[..., 2]
    verts_ndc[..., :2] *= -1

    if full_lmk:
        return verts_ndc[0, :, :2]
    else:
        return verts_ndc[0, 17:, :2]




import cv2
from torch.utils.tensorboard import SummaryWriter

n_img = len(camera_dict)
Ts = torch.zeros((n_img, 2)).float().to(device).requires_grad_()
scale_factor = torch.cuda.FloatTensor([260000]).requires_grad_()
z_factor = torch.full((n_img, 1), 149000.0, device=device, requires_grad=True)

shape_mean = []
for i in range(1, n_img + 1):
    with open('{}/deca_out/{:04d}/{:04d}_geo.pkl'.format(data_dir, i, i), 'rb') as f:
        render_data = pickle.load(f)
    shape_mean += [torch.from_numpy(render_data['shape']).to(device)]
shape_mean = torch.mean(torch.cat(shape_mean, dim=0), dim=0, keepdim=True).requires_grad_()
exp_offset = torch.full((n_img, 50), 0.0, device=device, requires_grad=True)
pose_offset = torch.full((n_img, 6), 0.0, device=device, requires_grad=True)
pose_mask = torch.full((1, 6), 1.0, device=device)
pose_mask[0, 3] = 0.

shader_pers = BlendShader()
raster_settings = RasterizationSettings(
    image_size=224,
    blur_radius=0.0,
    faces_per_pixel=1,
    perspective_correct=False,
)
# Rasterization settings for silhouette rendering
sigma = 1e-4
raster_settings_silhouette = RasterizationSettings(
    image_size=224,
    blur_radius=np.log(1. / 1e-4 - 1.) * sigma,
    faces_per_pixel=50,
    perspective_correct=False,
)

optimizer = torch.optim.Adam([{'params': Ts, 'lr': 1e-4},
                              {'params': shape_mean, 'lr': 1e-4},
                              {'params': exp_offset, 'lr': 1e-4},
                              {'params': pose_offset, 'lr': 1e-4},
                              {'params': scale_factor, 'lr': 1e2},
                              {'params': z_factor, 'lr': 1e2}])

n_epoch = 2000
loss_max = 1e10
exp_name = 'dan_1_nosift_zps'
save_dir = os.path.join('/tmp/pycharm_project_307/align_output', exp_name)
os.makedirs(save_dir, exist_ok=True)
writer = SummaryWriter('runs_s1/{}'.format(exp_name))
for epoch in tqdm(range(n_epoch)):
    shuffle_indices = torch.randperm(n_img)
    loss_epoch = 0
    loss_mask_e = 0
    loss_lmk_e = 0
    loss_sift_e = 0
    for j in range(n_img):
        full_lmk = ((shuffle_indices[j] + 1) in ids_360) or ((shuffle_indices[j] + 1) in ids_bow)
        img_id = '{:04d}'.format(shuffle_indices[j].item() + 1)
        if img_id in ignore_ids:
            continue
        T_id = int(img_id) - 1

        T_z = torch.zeros((1, 1), dtype=torch.float32, device=device)
        cam_T = torch.cat((Ts[T_id:T_id + 1, :], T_z), dim=-1) * scale_factor

        img_name = img_id + '.png'
        img_info = camera_dict[img_name]
        pose = np.array(img_info['W2C']).reshape((4, 4))
        R = convert_pose(np.linalg.inv(pose[:3, :3]))
        R = torch.from_numpy(R).float().to(device)
        cam_R = R.unsqueeze(0)

        focal_length, principal_point, image_size = get_cam_param(img_id, camera_dict)
        camera = PerspectiveCameras(focal_length=focal_length, principal_point=principal_point, in_ndc=False,
                                    R=cam_R, T=cam_T, image_size=image_size, device=device)

        _, exp, pose, _ = load_data(img_id)
        exp += exp_offset[T_id:T_id + 1]
        pose += pose_offset[T_id:T_id + 1]
        pose *= pose_mask

        verts, _, lmk_3d = flame(shape_params=shape_mean, expression_params=exp, pose_params=pose)
        verts[:, :, 1:] = -verts[:, :, 1:]
        verts[..., :2] = -verts[..., :2]

        verts = verts.permute(0, 2, 1)
        verts = torch.bmm(cam_R, verts)
        T_z = torch.zeros((1, 2), dtype=torch.float32, device=device)
        T_z = torch.cat((T_z, z_factor[T_id:T_id + 1]), dim=-1).unsqueeze(-1)
        T_z = torch.bmm(cam_R, T_z)
        verts = scale_factor * verts
        verts += T_z
        verts = verts.permute(0, 2, 1)

        lmk = proj_lmk_scale(lmk_3d, scale_factor, T_z, cam_R, camera, full_lmk)
        lmk_gt = load_lmk_gt(img_id, full_lmk)

        mesh = Meshes(verts=verts.float(), faces=render.faces.expand(1, -1, -1).long())

        # Silhouette renderer
        renderer_silhouette = MeshRenderer(
            rasterizer=MeshRasterizer(
                raster_settings=raster_settings_silhouette,
                cameras=camera,
            ),
            shader=SoftSilhouetteShader()
        )

        # Render silhouette images.  The 3rd channel of the rendering output is
        # the alpha/silhouette channel
        silhouette_images = renderer_silhouette(mesh)
        silhouette_images = silhouette_images[..., 3]

        mask_bg, mask_fg = load_mask(img_id)
        mask_zero = torch.zeros_like(mask_bg)

        loss_mask = F.mse_loss(silhouette_images * mask_bg, mask_zero, reduction='sum') / mask_bg.sum() + \
                    F.mse_loss(silhouette_images * mask_fg, mask_fg, reduction='sum') / mask_fg.sum()
        loss_lmk = F.l1_loss(lmk, lmk_gt)

        loss = loss_mask + loss_lmk
        loss_epoch += loss
        loss_mask_e += loss_mask
        loss_lmk_e += loss_lmk

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    #         break
    loss_epoch /= n_img
    loss_mask_e /= n_img
    loss_lmk_e /= n_img
    loss_sift_e /= n_img

    writer.add_scalar('Loss/mask', loss_mask_e, epoch)
    writer.add_scalar('Loss/landmark', loss_lmk_e, epoch)
    writer.add_scalar('Loss/total', loss_epoch, epoch)

    writer.add_scalar('Intensity/T_z', z_factor[0], epoch)
    writer.add_scalar('Intensity/scale', scale_factor, epoch)

    if epoch % 50 == 0:
        torch.save(Ts.detach().cpu(), 'align_output/{}/Ts_{:04d}.pt'.format(exp_name, epoch))
        torch.save(shape_mean.detach().cpu(), 'align_output/{}/shape_mean_{:04d}.pt'.format(exp_name, epoch))
        torch.save(exp_offset.detach().cpu(), 'align_output/{}/exp_offset_{:04d}.pt'.format(exp_name, epoch))
        torch.save(pose_offset.detach().cpu(), 'align_output/{}/pose_offset_{:04d}.pt'.format(exp_name, epoch))
        torch.save(scale_factor.detach().cpu(), 'align_output/{}/scale_factor_{:04d}.pt'.format(exp_name, epoch))
        torch.save(z_factor.detach().cpu(), 'align_output/{}/z_factor_{:04d}.pt'.format(exp_name, epoch))
    if loss_epoch < loss_max:
        loss_max = loss_epoch
        torch.save(Ts.detach().cpu(), 'align_output/{}/Ts_best.pt'.format(exp_name))
        torch.save(shape_mean.detach().cpu(), 'align_output/{}/shape_mean_best.pt'.format(exp_name))
        torch.save(exp_offset.detach().cpu(), 'align_output/{}/exp_offset_best.pt'.format(exp_name))
        torch.save(pose_offset.detach().cpu(), 'align_output/{}/pose_offset_best.pt'.format(exp_name))
        torch.save(scale_factor.detach().cpu(), 'align_output/{}/scale_factor_best.pt'.format(exp_name))
        torch.save(z_factor.detach().cpu(), 'align_output/{}/z_factor_best.pt'.format(exp_name))
        print(epoch, loss_max.item())

#     break

plt.figure(figsize=(10, 10))
plt.imshow(plot_lmk(silhouette_images, lmk))
plt.axis("off")

plt.figure(figsize=(10, 10))
plt.imshow(plot_lmk(1 - mask_bg, lmk_gt))
plt.axis("off")