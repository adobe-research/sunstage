import os
import cv2
import torch
import pickle
import kornia
import skimage
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import util
from util import VGGLoss
from FLAME import FLAME

# Data structures and functions for rendering
from pytorch3d.io import load_obj
from pytorch3d.ops import interpolate_face_attributes
from pytorch3d.structures import Meshes
from pytorch3d.renderer.mesh import rasterize_meshes
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


class Pytorch3dRasterizer(nn.Module):
    """  Borrowed from https://github.com/facebookresearch/pytorch3d
    Notice:
        x,y,z are in image space, normalized
        can only render squared image now
    """

    def __init__(self, image_size=224):
        """
        use fixed raster_settings for rendering faces
        """
        super().__init__()
        raster_settings = {
            'image_size': image_size,
            'blur_radius': 0.0,
            'faces_per_pixel': 1,
            'bin_size': None,
            'max_faces_per_bin': None,
            'perspective_correct': False,
        }
        raster_settings = util.dict2obj(raster_settings)
        self.raster_settings = raster_settings

    def forward(self, vertices, faces, attributes=None):
        fixed_vertices = vertices.clone()
        fixed_vertices[..., :2] = -fixed_vertices[..., :2]
        meshes_screen = Meshes(verts=fixed_vertices.float(), faces=faces.long())
        raster_settings = self.raster_settings
        pix_to_face, zbuf, bary_coords, dists = rasterize_meshes(
            meshes_screen,
            image_size=raster_settings.image_size,
            blur_radius=raster_settings.blur_radius,
            faces_per_pixel=raster_settings.faces_per_pixel,
            bin_size=raster_settings.bin_size,
            max_faces_per_bin=raster_settings.max_faces_per_bin,
            perspective_correct=raster_settings.perspective_correct,
        )
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

    def get_xyz(self, vertices, faces, scale=1):
        fixed_vertices = vertices.clone()
        meshes_screen = Meshes(verts=fixed_vertices.float(), faces=faces.long())
        raster_settings = self.raster_settings
        image_size = raster_settings.image_size * scale
        pix_to_face, zbuf, bary_coords, dists = rasterize_meshes(
            meshes_screen,
            image_size=image_size,
            blur_radius=raster_settings.blur_radius,
            faces_per_pixel=raster_settings.faces_per_pixel,
            bin_size=raster_settings.bin_size,
            max_faces_per_bin=raster_settings.max_faces_per_bin,
            perspective_correct=raster_settings.perspective_correct,
        )
        verts_per_face = meshes_screen.verts_packed()[meshes_screen.faces_packed()]
        pixel_verts = interpolate_face_attributes(
            pix_to_face, bary_coords, verts_per_face
        )

        hit_z = torch.cat([pixel_verts[:, :, :, 0, :], zbuf], dim=-1)
        return hit_z

    def get_mesh(self, vertices, faces):
        fixed_vertices = vertices.clone()
        fixed_vertices[..., :2] = -fixed_vertices[..., :2]
        meshes_screen = Meshes(verts=fixed_vertices.float(), faces=faces.long())
        return meshes_screen


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


class SunStage2():
    def __init__(self, opt, n_img):
        data_dir = os.path.join(opt.data_dir, opt.obj_name)
        device = opt.device
        self.opt = opt
        self.device = device

        disp_map = []
        albedo_map = []
        for i in range(1, n_img + 1):
            with open('{}/deca_out/{:04d}/{:04d}_geo.pkl'.format(data_dir, i, i), 'rb') as f:
                render_data = pickle.load(f)
            verts_disp = torch.from_numpy(render_data['verts_disp']).to(device)
            albedo = torch.from_numpy(render_data['albedo']).to(device)
            disp_map += [verts_disp]
            albedo_map += [albedo]
        albedo = torch.mean(torch.cat(albedo_map, dim=0), dim=0, keepdim=True)
        albedo = albedo.sign() * (albedo.abs() + 1e-10) ** 2.2
        albedo_inv = -torch.log((1 - albedo) / (albedo + 1e-10))
        disp_map = torch.full((1, 1, 256, 256), 0.0, device=device, requires_grad=True)

        disp_mask = cv2.imread('./data/sunstage/uv_eye_mouth.png')
        disp_mask = cv2.resize(disp_mask, (256, 256), cv2.INTER_NEAREST)
        disp_mask[disp_mask > 250] = 255
        disp_mask[disp_mask <= 250] = 0
        disp_mask = torch.from_numpy(disp_mask).float().to(device)
        disp_mask = disp_mask.permute(2, 0, 1)[None, :1, ...] / 255.
        pose_mask = torch.full((1, 6), 1.0, device=device)
        pose_mask[0, 3] = 0.

        self.load_s1()

        albedo_inv = albedo_inv.float().to(device).requires_grad_()
        sp_shininess = torch.full((1, 10), -2., device=device, requires_grad=True)
        sp_intensity = torch.full((1, 10), -2.7, device=device, requires_grad=True)
        env_color = torch.full((16, 32, 3), -4.0, device=device, requires_grad=True)
        light_param = torch.cuda.FloatTensor([0.0, 0.0, -1.0, .7, -4., 1., 1., 1., -4.]).requires_grad_()

        # load face and uv coord
        _, faces, aux = load_obj(os.path.join(opt.deca_dir, 'data', 'head_template.obj'))
        self.faces = faces.verts_idx[None, ...].to(device)
        uvcoords = aux.verts_uvs[None, ...]  # (N, V, 2)
        uvfaces = faces.textures_idx[None, ...]  # (N, F, 3)
        uvcoords = torch.cat([uvcoords, uvcoords[:, :, 0:1] * 0. + 1.], -1)  # [bz, ntv, 3]
        uvcoords = uvcoords * 2 - 1;
        uvcoords[..., 1] = -uvcoords[..., 1]
        self.face_uvcoords = util.face_vertices(uvcoords, uvfaces).to(device)
        self.uvcoords = uvcoords.to(device)
        self.uvfaces = uvfaces.to(device)
        self.load_uv_labels()

        # set up displacement map
        mask = skimage.io.imread('./data/sunstage/uv_face_eye_scoket_mask.png')
        mask[mask > 250] = 255
        mask[mask <= 250] = 0
        mask = torch.from_numpy(mask[:, :, 0])[None, None, :, :].contiguous().float() / 255.
        uv_face_eye_mask = F.interpolate(mask, [256, 256]).to(device)
        # uv eye mask
        self.uv_face_eye_mask = kornia.filters.gaussian_blur2d(uv_face_eye_mask, (5, 5), (2.5, 2.5))
        fixed_dis = np.load(os.path.join(opt.deca_dir, 'data', 'fixed_displacement_256.npy'))
        # uv eye fixed displacement
        self.fixed_uv_dis = torch.tensor(fixed_dis).float().to(device)
        # detailed face for detailed mesh
        dense_triangles = util.generate_triangles(256, 256)
        self.dense_faces = torch.from_numpy(dense_triangles).long()[None, :, :].to(device)

        # set up render
        self.set_raster_settings()
        self.shadow_rasterizer = Pytorch3dRasterizer(224).to(device)
        self.uv_rasterizer = Pytorch3dRasterizer(256).to(device)
        self.flame = get_flame(opt.deca_dir, device)
        self.pose_mask = pose_mask

        # vgg loss
        self.vgg_loss = VGGLoss()

        # set up optimizer
        self.light_param = light_param
        self.albedo_inv = albedo_inv
        self.sp_shininess = sp_shininess
        self.sp_intensity = sp_intensity
        self.env_color = env_color
        self.disp_map = disp_map
        self.optimizer = torch.optim.Adam([{'params': self.Ts, 'lr': 1e-4},
                                           {'params': self.shape_mean, 'lr': 1e-4},
                                           {'params': self.exp_offset, 'lr': 1e-4},
                                           {'params': self.pose_offset, 'lr': 1e-4},
                                           {'params': self.scale_factor, 'lr': 1e2},
                                           {'params': self.z_factor, 'lr': 1e2},
                                           {'params': self.light_param, 'lr': 1e-3},
                                           {'params': self.albedo_inv, 'lr': 1e-2},
                                           {'params': self.sp_shininess, 'lr': 1e-2},
                                           {'params': self.sp_intensity, 'lr': 1e-2},
                                           {'params': self.env_color, 'lr': 1e-3},
                                           {'params': self.disp_map, 'lr': 1e-4},])

    def load_s1(self):
        s1_dict = torch.load(f'{self.opt.s1_dir}/{self.opt.obj_name}_{self.opt.s1_epoch}_s1.pt')
        self.Ts = s1_dict['Ts'].to(self.device).requires_grad_()
        self.shape_mean = s1_dict['shape_mean'].to(self.device).requires_grad_()
        self.exp_offset = s1_dict['exp_offset'].to(self.device).requires_grad_()
        self.pose_offset = s1_dict['pose_offset'].to(self.device).requires_grad_()
        self.scale_factor = s1_dict['scale_factor'].to(self.device).requires_grad_()
        self.z_factor = s1_dict['z_factor'].to(self.device).requires_grad_()

    def set_raster_settings(self):
        self.raster_settings = RasterizationSettings(
            image_size=224,
            blur_radius=0.0,
            faces_per_pixel=1,
            perspective_correct=False,
        )
        self.raster_settings_shadow = RasterizationSettings(
            image_size=224 * 8,
            blur_radius=0.0,
            faces_per_pixel=1,
            perspective_correct=False,
        )
        # Rasterization settings for silhouette rendering
        sigma = 1e-4
        self.raster_settings_silhouette = RasterizationSettings(
            image_size=224,
            blur_radius=np.log(1. / 1e-4 - 1.) * sigma,
            faces_per_pixel=50,
            perspective_correct=False,
        )
        self.shader_pers = BlendShader()

    def load_uv_labels(self):
        uv_labels = []
        for i in range(1, 11):
            mask = cv2.imread('./data/sunstage/uv_seg/{:02d}.png'.format(i))
            mask[mask < 127] = 0
            mask[mask >= 127] = 255
            mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST) / 255.
            mask = cv2.GaussianBlur(mask, (5, 5), 0)
            uv_labels += [torch.from_numpy(mask[..., :1])]
        uv_labels = torch.cat(uv_labels, dim=-1).float().to(self.device).permute(2, 0, 1).unsqueeze(0)
        self.uv_labels = uv_labels / (torch.sum(uv_labels, dim=1, keepdim=True) + 1e-10)

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

    def world2uv(self, vertices):
        '''
        warp vertices from world space to uv space
        vertices: [bz, V, 3]
        uv_vertices: [bz, 3, h, w]
        '''
        batch_size = vertices.shape[0]
        face_vertices = util.face_vertices(vertices, self.faces.expand(batch_size, -1, -1))
        uv_vertices = self.uv_rasterizer(self.uvcoords.expand(batch_size, -1, -1),
                                         self.uvfaces.expand(batch_size, -1, -1), face_vertices)[:, :3]
        return uv_vertices

    def displacement2normal(self, uv_z, coarse_verts, coarse_normals):
        batch_size = uv_z.shape[0]
        uv_coarse_vertices = self.world2uv(coarse_verts)
        uv_coarse_normals = self.world2uv(coarse_normals)

        uv_z = uv_z * self.uv_face_eye_mask
        uv_detail_vertices = uv_coarse_vertices + uv_z * uv_coarse_normals + self.fixed_uv_dis[None, None, :,
                                                                             :] * uv_coarse_normals
        dense_vertices = uv_detail_vertices.permute(0, 2, 3, 1).reshape([batch_size, -1, 3])
        uv_detail_normals = util.vertex_normals(dense_vertices, self.dense_faces.expand(batch_size, -1, -1))
        uv_detail_normals = uv_detail_normals.reshape(
            [batch_size, uv_coarse_vertices.shape[2], uv_coarse_vertices.shape[3], 3]).permute(0, 3, 1, 2)
        uv_detail_normals = uv_detail_normals * self.uv_face_eye_mask + uv_coarse_normals * (1 - self.uv_face_eye_mask)
        return uv_detail_normals

    def get_shape(self, img_dict, T_id):
        exp = img_dict['exp'][0]
        pose = img_dict['pose'][0]
        exp += self.exp_offset[T_id:T_id + 1]
        pose += self.pose_offset[T_id:T_id + 1]
        pose *= self.pose_mask

        verts, _, lmk_3d = self.flame(shape_params=self.shape_mean, expression_params=exp, pose_params=pose)
        coarse_verts = verts.clone()
        coarse_normals = util.vertex_normals(coarse_verts, self.faces.expand(1, -1, -1))
        uv_z = 1e-2 * torch.tanh(self.disp_map)
        uv_detail_normals = self.displacement2normal(uv_z, coarse_verts, coarse_normals)
        return verts, lmk_3d, uv_detail_normals

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
        return verts, T_z

    def transform_normal(self, verts):
        transformed_normals = util.vertex_normals(verts.clone(), self.faces.expand(1, -1, -1))
        transformed_face_normals = util.face_vertices(transformed_normals, self.faces.expand(1, -1, -1))
        return transformed_face_normals

    def proj_lmk_scale(self, verts, cam_R, camera, T_id, full_lmk=True):
        verts, _ = self.transform_verts(verts.clone(), cam_R, T_id)

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

    def shading_ambient(self, normal_images, img_h, img_w):
        light_pos = F.normalize(self.light_param[0:3, None], dim=0).reshape(-1, 3, 1).permute(0, 2, 1)
        normal_images = normal_images.reshape(1, 3, -1)
        xyz, areas = util.gen_light_xyz(16, 32, envmap_radius=1.)
        xyz = xyz.reshape(1, -1, 3).to(self.device)
        areas = areas.reshape(1, 1, -1).to(self.device)
        # cosine term
        light_dot_normal = torch.bmm(xyz, normal_images)
        light_dot_normal = torch.clamp(light_dot_normal, 0., 1.)
        # sample color
        envmap_c = torch.exp(self.env_color)
        envmap_c = envmap_c.reshape(1, -1, 3).permute(0, 2, 1)
        envmap_c = envmap_c * areas
        # sum over envmap
        light_dot_normal = torch.bmm(envmap_c, light_dot_normal)
        l_amb = light_dot_normal.reshape(1, -1, img_h, img_w)
        return l_amb

    def shading_sun(self, normal_images, img_h, img_w):
        light_pos = F.normalize(self.light_param[0:3, None], dim=0).reshape(-1, 3, 1).permute(0, 2, 1)
        normal_images = normal_images.reshape(1, 3, -1)
        light_dot_normal = torch.bmm(light_pos, normal_images)
        light_dot_normal = torch.clamp(light_dot_normal, 0., 1.).reshape(1, -1, img_h, img_w)
        l_sun = torch.exp(self.light_param[3]) * light_dot_normal.expand(1, 3, img_h, img_w)
        return l_sun

    def shading_specular(self, normal_images, view_dir, uv_labels_images, img_h, img_w):
        light_pos = F.normalize(self.light_param[0:3, None], dim=0).unsqueeze(0).expand(-1, -1, 224 * 224)
        normal_images = normal_images.reshape(1, 3, -1)
        view_dir = F.normalize(view_dir, p=2.0, dim=-1).reshape(1, -1, 3).permute(0, 2, 1)
        half_v = F.normalize(view_dir + light_pos, p=2.0, dim=1)
        nh = torch.sum(half_v * normal_images, dim=1, keepdim=True)
        l_sp = torch.clamp(nh, 0., 1.).reshape(1, -1, img_h, img_w).expand(-1, 3, -1, -1)

        uv_labels_images = uv_labels_images[0].reshape(10, -1)
        shininess_map = torch.mm(self.sp_shininess, uv_labels_images).reshape(1, -1, img_h, img_w)
        intensity_map = torch.mm(self.sp_intensity, uv_labels_images).reshape(1, -1, img_h, img_w)
        n = torch.exp(6.5 * torch.sigmoid(shininess_map) + .5)
        l_sp = torch.exp(self.light_param[3]) * torch.sigmoid(intensity_map) * (l_sp ** n) * (n + 2)
        return l_sp

    def render_shadow_depth(self, trans_verts, light_pos, z_factor, scale_factor):
        light_pos = F.normalize(light_pos, dim=0)
        z = -light_pos
        y = torch.cuda.FloatTensor([0., 1., 1e-10])
        x = F.normalize(y.cross(z), dim=0)
        y = F.normalize(z.cross(x), dim=0)

        x = x[:, None]
        y = y[:, None]
        z = z[:, None]
        R = torch.cat((x, y, z), dim=1)
        R = R.T
        trans_verts = trans_verts.clone()
        trans_verts -= z_factor
        trans_verts /= scale_factor
        trans_verts = trans_verts.permute(0, 2, 1)

        R = R[None, ...]
        trans_verts = R.bmm(trans_verts)
        trans_verts = trans_verts.permute(0, 2, 1)

        trans_verts[..., :2] *= 5.
        trans_verts[..., 2] = trans_verts[..., 2] + 10

        xyz_d = self.shadow_rasterizer.get_xyz(trans_verts, self.faces.expand(1, -1, -1), 8)
        shadow_depth = xyz_d[..., -1:].permute(0, 3, 1, 2)
        return shadow_depth, R

    def render_shadow_map(self, xyz_d, shadow_depth, R, z_factor, scale_factor):
        xyz = xyz_d[..., :3]
        xyz -= z_factor
        xyz /= scale_factor
        xyz = xyz.permute(0, 3, 1, 2).reshape(1, 3, -1)
        xyz = R.bmm(xyz)
        xyz = xyz.permute(0, 2, 1).reshape(1, 224 * 8, 224 * 8, -1)
        xyz[..., -1] += 10

        z = xyz[..., -1:].permute(0, 3, 1, 2)
        xy = xyz[..., :2]
        shadow_z = F.grid_sample(shadow_depth, -xy * 5, mode='nearest', align_corners=True)

        shadow_map = torch.sigmoid((z - shadow_z * 1.0015) * 800.)
        shadow_map = shadow_map.expand(-1, 3, -1, -1)
        shadow_map = F.avg_pool2d(shadow_map, 8)
        return shadow_map

    def cast_shadow(self, verts, T_z, xyz_d):
        light_pos = F.normalize(self.light_param[0:3, None], dim=0)[:, 0]
        T_z = T_z.reshape(1, 1, 3)
        shadow_depth, R = self.render_shadow_depth(verts, light_pos, T_z, self.scale_factor)
        shadow_depth = torch.where(shadow_depth < 0., shadow_depth.max(), shadow_depth)
        shadow_map = self.render_shadow_map(xyz_d, shadow_depth, R, T_z, self.scale_factor)
        return shadow_map

    def shading_rgb(self, l_amb, l_sun, l_sp, shadow_map, albedo_images):
        shading_diffuse = (1. - shadow_map) * l_sun + l_amb
        shading_specular = (1. - shadow_map) * l_sp
        images = (torch.sigmoid(albedo_images) * shading_diffuse + shading_specular)
        return images

    def tune_mapping(self, images):
        images_L = images * torch.Tensor((0.2126, 0.7152, 0.0722))[None, :, None, None].float().to(self.device)
        images_L = torch.sum(images_L, dim=1, keepdim=True)
        images /= 1. + images_L
        images = images.sign() * (images.abs() + 1e-10) ** (1 / 2.2)
        images = torch.clamp(images, 0., 1.)
        return images

    def render_s2(self, img_dict):
        img_id = img_dict['img_id'][0]
        T_id = int(img_id) - 1
        # get camera
        camera = self.get_camera(img_dict, T_id)
        cam_R = img_dict['cam_R'][0].clone()
        cam_center = camera.get_camera_center()
        # get detailed geometry
        verts, lmk_3d, uv_detail_normals = self.get_shape(img_dict, T_id)
        verts, T_z = self.transform_verts(verts, img_dict['cam_R'][0], T_id)
        transformed_face_normals = self.transform_normal(verts)

        attributes = torch.cat([self.face_uvcoords.expand(1, -1, -1, -1),
                                transformed_face_normals],
                               -1)
        mesh = Meshes(verts=verts.float(), faces=self.faces.expand(1, -1, -1).long())

        # Silhouette renderer
        renderer_silhouette = MeshRenderer(
            rasterizer=MeshRasterizer(
                raster_settings=self.raster_settings_silhouette,
                cameras=camera,
            ),
            shader=SoftSilhouetteShader()
        )
        # Render silhouette images.
        silhouette_images = renderer_silhouette(mesh)
        silhouette_images = silhouette_images[..., 3]

        # rasterize rgb and shadow
        fragments = MeshRasterizer(raster_settings=self.raster_settings, cameras=camera)(mesh)
        rendering = self.shader_pers(fragments, mesh, attributes=attributes)
        view_dir = self.shader_pers.get_xyz(fragments, mesh)
        view_dir = cam_center - view_dir[..., :3]
        fragments = MeshRasterizer(raster_settings=self.raster_settings_shadow, cameras=camera)(mesh)
        xyz_d = self.shader_pers.get_xyz(fragments, mesh)
        alpha_images = rendering[:, -1, :, :][:, None, :, :].detach()

        # sample albedo
        uvcoords_images = rendering[:, :3, :, :];
        grid = (uvcoords_images).permute(0, 2, 3, 1)[:, :, :, :2]
        albedo_images = F.grid_sample(self.albedo_inv, grid, align_corners=False)
        detail_normals = F.grid_sample(uv_detail_normals, grid, align_corners=False)
        uv_labels_images = F.grid_sample(self.uv_labels, grid, align_corners=False)

        # transform normal
        detail_normals = detail_normals.reshape(1, 3, -1)
        detail_normals[:, 0, :] *= -1
        detail_normals[:, 2, :] *= -1
        detail_normals = torch.bmm(cam_R, detail_normals).reshape(1, 3, 224, 224) * alpha_images
        normal_images = F.normalize(detail_normals, dim=1)
        img_h, img_w = normal_images.shape[2], normal_images.shape[3]

        # shading ambient, directional, specular
        l_amb = self.shading_ambient(normal_images, img_h, img_w)
        l_sun = self.shading_sun(normal_images, img_h, img_w)
        l_sp = self.shading_specular(normal_images, view_dir, uv_labels_images, img_h, img_w)
        # cast shadow
        shadow_map = self.cast_shadow(verts, T_z, xyz_d)

        # shading rgb
        images = self.shading_rgb(l_amb, l_sun, l_sp, shadow_map, albedo_images) * alpha_images
        images = self.tune_mapping(images)

        return images, silhouette_images, lmk_3d

    def step_s2(self, img_dict, rendering, full_loss):
        images, silhouette_images, lmk_3d = rendering

        # RGB loss
        loss_color = F.mse_loss(images, img_dict['img_gt'][0])
        loss_vgg = self.vgg_loss(images, img_dict['img_gt'][0])
        # mask loss
        mask_bg, mask_fg = img_dict['mask_bg'][0], img_dict['mask_fg'][0]
        mask_zero = torch.zeros_like(mask_bg)
        loss_mask = F.mse_loss(silhouette_images * mask_bg, mask_zero, reduction='sum') / mask_bg.sum() + \
                    F.mse_loss(silhouette_images * mask_fg, mask_fg, reduction='sum') / mask_fg.sum()
        # keypoint loss
        img_id = img_dict['img_id'][0]
        T_id = int(img_id) - 1
        camera = self.get_camera(img_dict, T_id)
        lmk = self.proj_lmk_scale(lmk_3d, img_dict['cam_R'][0], camera, T_id, img_dict['full_lmk'])
        lmk_gt = img_dict['lmk_gt'][0]
        if not img_dict['full_lmk']:
            lmk_gt = lmk_gt[17:, :]
        loss_lmk = F.l1_loss(lmk, lmk_gt)
        # env loss
        envmap_c = torch.exp(self.env_color)
        envmap_cy = torch.roll(envmap_c, 1, 0)
        envmap_cx = torch.roll(envmap_c, 1, 1)
        loss_env_sm = F.mse_loss(envmap_cy[1:, :, :], envmap_c[1:, :, :]) + F.mse_loss(envmap_cx[:, 1:, :],
                                                                                       envmap_c[:, 1:, :])
        loss_env_sm *= 100
        envmap_c = envmap_c.reshape(1, -1, 3).permute(0, 2, 1)
        loss_env = F.mse_loss(envmap_c, torch.zeros_like(envmap_c)) + loss_env_sm

        # total loss
        loss_s1 = loss_mask + loss_lmk
        loss_s2 = loss_color + loss_vgg * .005 + loss_env * .01
        if full_loss or img_dict['full_lmk']:
            loss = loss_s2 + loss_s1 * 0.05
        else:
            loss = loss_s2 * 0. + loss_s1 * 0.05

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def save(self, save_dir, n_epoch):
        save_dict = {'Ts' : self.Ts.detach().cpu(),
                     'shape_mean' : self.shape_mean.detach().cpu(),
                     'exp_offset' : self.exp_offset.detach().cpu(),
                     'pose_offset' : self.pose_offset.detach().cpu(),
                     'scale_factor' : self.scale_factor.detach().cpu(),
                     'z_factor' : self.z_factor.detach().cpu(),
                     'light_param' : self.light_param.detach().cpu(),
                     'albedo_inv' : self.albedo_inv.detach().cpu(),
                     'sp_shininess' : self.sp_shininess.detach().cpu(),
                     'sp_intensity' : self.sp_intensity.detach().cpu(),
                     'env_color' : self.env_color.detach().cpu(),
                     'disp_map' : self.disp_map.detach().cpu()}
        fname = os.path.join(save_dir, f'{self.opt.obj_name}_{n_epoch}_s2.pt')
        torch.save(save_dict, fname)
