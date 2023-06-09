import os
import torch
import pickle
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from FLAME import FLAME

# Data structures and functions for rendering
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer.mesh.rasterizer import Fragments
from pytorch3d.renderer import (
    look_at_view_transform,
    PointLights,
    DirectionalLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    SoftSilhouetteShader,
    SoftPhongShader,
    TexturesVertex
)
from pytorch3d.renderer.cameras import PerspectiveCameras


def get_flame(deca_dir, device):
    cfg = argparse.ArgumentParser()
    cfg.deca_dir = deca_dir
    model_cfg = argparse.ArgumentParser()
    model_cfg.topology_path = os.path.join(cfg.deca_dir, 'data', 'head_template.obj')
    # texture data original from http://files.is.tue.mpg.de/tbolkart/FLAME/FLAME_texture_data.zip
    model_cfg.dense_template_path = os.path.join(cfg.deca_dir, 'data', 'texture_data_256.npy')
    model_cfg.fixed_displacement_path = os.path.join(cfg.deca_dir, 'data', 'fixed_displacement_256.npy')
    model_cfg.flame_model_path = os.path.join(cfg.deca_dir, 'data', 'generic_model.pkl')
    model_cfg.flame_lmk_embedding_path = os.path.join(cfg.deca_dir, 'data', 'landmark_embedding.npy')
    model_cfg.face_mask_path = os.path.join(cfg.deca_dir, 'data', 'uv_face_mask.png')
    model_cfg.face_eye_mask_path = os.path.join(cfg.deca_dir, 'data', 'uv_face_eye_mask.png')
    model_cfg.mean_tex_path = os.path.join(cfg.deca_dir, 'data', 'mean_texture.jpg')
    model_cfg.tex_path = os.path.join(cfg.deca_dir, 'data', 'FLAME_albedo_from_BFM.npz')
    model_cfg.tex_type = 'BFM'  # BFM, FLAME, albedoMM
    model_cfg.uv_size = 256
    model_cfg.param_list = ['shape', 'tex', 'exp', 'pose', 'cam', 'light']
    model_cfg.n_shape = 100
    model_cfg.n_tex = 50
    model_cfg.n_exp = 50
    model_cfg.n_cam = 3
    model_cfg.n_pose = 6
    model_cfg.n_light = 27
    model_cfg.use_tex = True
    model_cfg.jaw_type = 'aa'  # default use axis angle, another option: euler. Note that: aa is not stable in the beginning
    # face recognition model
    model_cfg.fr_model_path = os.path.join(cfg.deca_dir, 'data', 'resnet50_ft_weight.pkl')

    ## details
    model_cfg.n_detail = 128
    model_cfg.max_z = 0.01
    flame = FLAME(model_cfg).to(device)
    return flame


class BlendShader(nn.Module):
    """  Borrowed from https://github.com/facebookresearch/pytorch3d
    Notice:
        x,y,z are in image space, normalized
        can only render squared image now
    """

    def __init__(self):
        """
        use fixed raster_settings for rendering faces
        """
        super().__init__()

    def forward(self, fragments: Fragments, meshes: Meshes, attributes: torch.Tensor, **kwargs) -> torch.Tensor:
        pix_to_face, zbuf, bary_coords, dists = fragments.pix_to_face, fragments.zbuf, fragments.bary_coords, fragments.dists
        vismask = (pix_to_face > -1).float()
        D = attributes.shape[-1]
        attributes = attributes.clone();
        attributes = attributes.view(attributes.shape[0] * attributes.shape[1], 3, attributes.shape[-1])
        N, H, W, K, _ = bary_coords.shape
        mask = pix_to_face == -1
        pix_to_face = pix_to_face.clone()
        pix_to_face[mask] = 0
        idx = pix_to_face.view(N * H * W * K, 1, 1).expand(N * H * W * K, 3, D)
        pixel_face_vals = attributes.gather(0, idx).view(N, H, W, K, 3, D)
        pixel_vals = (bary_coords[..., None] * pixel_face_vals).sum(dim=-2)
        pixel_vals[mask] = 0  # Replace masked values in output.
        pixel_vals = pixel_vals[:, :, :, 0].permute(0, 3, 1, 2)
        pixel_vals = torch.cat([pixel_vals, vismask[:, :, :, 0][:, None, :, :]], dim=1)
        return pixel_vals

    def get_xyz(self, fragments: Fragments, meshes: Meshes, **kwargs) -> torch.Tensor:
        pix_to_face, zbuf, bary_coords, dists = fragments.pix_to_face, fragments.zbuf, fragments.bary_coords, fragments.dists
        verts_per_face = meshes.verts_packed()[meshes.faces_packed()]
        pixel_verts = interpolate_face_attributes(
            pix_to_face, bary_coords, verts_per_face
        )

        hit_z = torch.cat([pixel_verts[:, :, :, 0, :], zbuf], dim=-1)
        return hit_z


class SunStage1():
    def __init__(self, opt, n_img):
        data_dir = os.path.join(opt.data_dir, opt.obj_name)
        device = opt.device
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

        # load face

        _, faces, _ = load_obj(os.path.join(opt.deca_dir, 'data', 'head_template.obj'))
        self.faces = faces.verts_idx[None, ...].to(device)

        self.shader_pers = BlendShader()
        # Rasterization settings for silhouette rendering
        sigma = 1e-4
        self.raster_settings_silhouette = RasterizationSettings(
            image_size=224,
            blur_radius=np.log(1. / 1e-4 - 1.) * sigma,
            faces_per_pixel=50,
            perspective_correct=False,
        )

        self.opt = opt
        self.flame = get_flame(opt.deca_dir, device)
        self.pose_mask = pose_mask
        self.Ts = Ts
        self.shape_mean = shape_mean
        self.exp_offset = exp_offset
        self.pose_offset = pose_offset
        self.scale_factor = scale_factor
        self.z_factor = z_factor
        self.device = device
        self.optimizer = torch.optim.Adam([{'params': self.Ts, 'lr': 1e-4},
                                           {'params': self.shape_mean, 'lr': 1e-4},
                                           {'params': self.exp_offset, 'lr': 1e-4},
                                           {'params': self.pose_offset, 'lr': 1e-4},
                                           {'params': self.scale_factor, 'lr': 1e2},
                                           {'params': self.z_factor, 'lr': 1e2}])

    def get_camera(self, img_dict, T_id):
        T_z = torch.zeros((1, 1), dtype=torch.float32, device=self.device)
        cam_T = torch.cat((self.Ts[T_id:T_id + 1, :], T_z), dim=-1) * self.scale_factor

        camera = PerspectiveCameras(focal_length=img_dict['focal_length'][0],
                                    principal_point=img_dict['principal_point'][0],
                                    in_ndc=False,
                                    R=img_dict['cam_R'][0],
                                    T=cam_T,
                                    image_size=img_dict['image_size'][0],
                                    device=self.device)
        return camera

    def get_shape(self, img_dict, T_id):
        exp = img_dict['exp'][0]
        pose = img_dict['pose'][0]
        exp += self.exp_offset[T_id:T_id + 1]
        pose += self.pose_offset[T_id:T_id + 1]
        pose *= self.pose_mask

        verts, _, lmk_3d = self.flame(shape_params=self.shape_mean, expression_params=exp, pose_params=pose)
        return verts, lmk_3d

    def transform_verts(self, verts, cam_R, T_id):
        verts[:, :, 1:] = -verts[:, :, 1:]
        verts[..., :2] = -verts[..., :2]

        verts = verts.permute(0, 2, 1)
        verts = torch.bmm(cam_R, verts)
        T_z = torch.zeros((1, 2), dtype=torch.float32, device=self.device)
        T_z = torch.cat((T_z, self.z_factor[T_id:T_id + 1]), dim=-1).unsqueeze(-1)
        T_z = torch.bmm(cam_R, T_z)
        verts = self.scale_factor * verts
        verts += T_z
        verts = verts.permute(0, 2, 1)
        return verts

    def proj_lmk_scale(self, verts, cam_R, camera, T_id, full_lmk=True):
        verts = self.transform_verts(verts.clone(), cam_R, T_id)

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

    def render_s1(self, img_dict):
        img_id = img_dict['img_id'][0]
        T_id = int(img_id) - 1
        camera = self.get_camera(img_dict, T_id)
        verts, lmk_3d = self.get_shape(img_dict, T_id)
        verts = self.transform_verts(verts, img_dict['cam_R'][0], T_id)
        mesh = Meshes(verts=verts.float(), faces=self.faces.expand(1, -1, -1).long())

        # Silhouette renderer
        renderer_silhouette = MeshRenderer(
            rasterizer=MeshRasterizer(
                raster_settings=self.raster_settings_silhouette,
                cameras=camera,
            ),
            shader=SoftSilhouetteShader()
        )

        # Render silhouette images.  The 3rd channel of the rendering output is
        # the alpha/silhouette channel
        silhouette_images = renderer_silhouette(mesh)
        silhouette_images = silhouette_images[..., 3]

        return silhouette_images, lmk_3d

    def step_s1(self, img_dict, silhouette_images, lmk_3d):
        img_id = img_dict['img_id'][0]
        T_id = int(img_id) - 1
        camera = self.get_camera(img_dict, T_id)
        mask_bg, mask_fg = img_dict['mask_bg'][0], img_dict['mask_fg'][0]
        mask_zero = torch.zeros_like(mask_bg)

        loss_mask = F.mse_loss(silhouette_images * mask_bg, mask_zero, reduction='sum') / mask_bg.sum() + \
                    F.mse_loss(silhouette_images * mask_fg, mask_fg, reduction='sum') / mask_fg.sum()

        lmk = self.proj_lmk_scale(lmk_3d, img_dict['cam_R'][0], camera, T_id, img_dict['full_lmk'])
        lmk_gt = img_dict['lmk_gt'][0]
        if not img_dict['full_lmk']:
            lmk_gt = lmk_gt[17:, :]
        loss_lmk = F.l1_loss(lmk, lmk_gt)

        loss = loss_mask + loss_lmk

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def save(self, save_dir, n_epoch):
        save_dict = {'Ts': self.Ts.detach().cpu(),
                     'shape_mean': self.shape_mean.detach().cpu(),
                     'exp_offset': self.exp_offset.detach().cpu(),
                     'pose_offset': self.pose_offset.detach().cpu(),
                     'scale_factor': self.scale_factor.detach().cpu(),
                     'z_factor': self.z_factor.detach().cpu()}
        fname = os.path.join(save_dir, f'{self.opt.obj_name}_{n_epoch}_s1.pt')
        torch.save(save_dict, fname)
