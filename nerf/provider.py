import os
import cv2
import glob
import json
import tqdm
import random
import numpy as np
from scipy.spatial.transform import Slerp, Rotation
import math
from camera_utils import create_cam2world_matrix, FOV_to_intrinsics
from training.volumetric_rendering import math_utils


import trimesh

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .utils import get_rays, safe_normalize

DIR_COLORS = np.array([
    [255, 0, 0, 255], # front
    [0, 255, 0, 255], # side
    [0, 0, 255, 255], # back
    [255, 255, 0, 255], # side
    [255, 0, 255, 255], # overhead
    [0, 255, 255, 255], # bottom
], dtype=np.uint8)

def visualize_poses(poses, dirs, size=0.1):
    # poses: [B, 4, 4], dirs: [B]

    axes = trimesh.creation.axis(axis_length=4)
    sphere = trimesh.creation.icosphere(radius=1)
    objects = [axes, sphere]

    for pose, dir in zip(poses, dirs):
        # a camera is visualized with 8 line segments.
        pos = pose[:3, 3]
        a = pos + size * pose[:3, 0] + size * pose[:3, 1] - size * pose[:3, 2]
        b = pos - size * pose[:3, 0] + size * pose[:3, 1] - size * pose[:3, 2]
        c = pos - size * pose[:3, 0] - size * pose[:3, 1] - size * pose[:3, 2]
        d = pos + size * pose[:3, 0] - size * pose[:3, 1] - size * pose[:3, 2]

        segs = np.array([[pos, a], [pos, b], [pos, c], [pos, d], [a, b], [b, c], [c, d], [d, a]])
        segs = trimesh.load_path(segs)

        # different color for different dirs
        segs.colors = DIR_COLORS[[dir]].repeat(len(segs.entities), 0)

        objects.append(segs)

    trimesh.Scene(objects).show()

def get_view_direction(thetas, phis, overhead, front):
    #                   phis [B,];          thetas: [B,]
    # front = 0         [0, front)
    # side (left) = 1   [front, 180)
    # back = 2          [180, 180+front)
    # side (right) = 3  [180+front, 360)
    # top = 4                               [0, overhead]
    # bottom = 5                            [180-overhead, 180]
    res = torch.zeros(thetas.shape[0], dtype=torch.long)
    # first determine by phis
    # res[(phis < front)] = 0
    # res[(phis >= front) & (phis < np.pi)] = 1
    # res[(phis >= np.pi) & (phis < (np.pi + front))] = 2
    # res[(phis >= (np.pi + front))] = 3
    res[(phis < front / 2) | (phis >= 2 * np.pi - front / 2)] = 0
    res[(phis >= front / 2) & (phis < np.pi - front / 2)] = 1
    res[(phis >= np.pi - front / 2) & (phis < np.pi + front / 2)] = 2
    res[(phis >= np.pi + front / 2) & (phis < 2 * np.pi - front / 2)] = 3
    # override by thetas
    res[thetas <= overhead] = 4
    res[thetas >= (np.pi - overhead)] = 5
    return res


def rand_poses(size, device, radius_range=[1, 1.5], theta_range=[0, 120], phi_range=[0, 360], return_dirs=False, angle_overhead=30, angle_front=60, jitter=False, uniform_sphere_rate=0.5):
    ''' generate random poses from an orbit camera
    Args:
        size: batch size of generated poses.
        device: where to allocate the output.
        radius: camera radius
        theta_range: [min, max], should be in [0, pi]
        phi_range: [min, max], should be in [0, 2 * pi]
    Return:
        poses: [size, 4, 4]
    '''
    def get_eg3d(size, device, radius_range, theta_range, phi_range, return_dirs, angle_overhead, angle_front, jitter, uniform_sphere_rate):
        theta_range = np.deg2rad(theta_range)
        phi_range = np.deg2rad(phi_range)
        angle_overhead = np.deg2rad(angle_overhead)
        angle_front = np.deg2rad(angle_front)
        
        radius = torch.rand(size, device=device) * (radius_range[1] - radius_range[0]) + radius_range[0]

        if random.random() < uniform_sphere_rate:
            unit_centers = F.normalize(
                torch.stack([
                    (torch.rand(size, device=device) - 0.5) * 2.0,
                    torch.rand(size, device=device),
                    (torch.rand(size, device=device) - 0.5) * 2.0,
                ], dim=-1), p=2, dim=1
            )
            thetas = torch.acos(unit_centers[:,1])
            phis = torch.atan2(unit_centers[:,0], unit_centers[:,2])
            phis[phis < 0] += 2 * np.pi
            # centers = unit_centers * radius.unsqueeze(-1)
        else:
            thetas = torch.rand(size, device=device) * (theta_range[1] - theta_range[0]) + theta_range[0]
            phis = torch.rand(size, device=device) * (phi_range[1] - phi_range[0]) + phi_range[0]

            # centers = torch.stack([
            #     radius * torch.sin(thetas) * torch.sin(phis),
            #     radius * torch.cos(thetas),
            #     radius * torch.sin(thetas) * torch.cos(phis),
            # ], dim=-1) # [B, 3]
        
        camera_origins = torch.zeros((size, 3), device=device)

        camera_origins[:, 0:1] = radius*torch.sin(phis) * torch.cos(math.pi-thetas)
        camera_origins[:, 2:3] = radius*torch.sin(phis) * torch.sin(math.pi-thetas)
        camera_origins[:, 1:2] = radius*torch.cos(phis)

        lookat_position = torch.tensor([0, 0, 0]).to(device)
        # forward_vectors = math_utils.normalize_vecs(-camera_origins)
        

        # jitters
        if jitter:
            camera_origins = camera_origins + (torch.rand_like(camera_origins) * 0.2 - 0.1)
            lookat_position = lookat_position + torch.randn_like(camera_origins) * 0.2
            
        forward_vectors = math_utils.normalize_vecs(lookat_position - camera_origins)
        
        poses = create_cam2world_matrix(forward_vectors, camera_origins)

        if return_dirs:
            dirs = get_view_direction(thetas, phis, angle_overhead, angle_front)
        else:
            dirs = None
            
        return poses, dirs
            
    theta_range = np.deg2rad(theta_range)
    phi_range = np.deg2rad(phi_range)
    angle_overhead = np.deg2rad(angle_overhead)
    angle_front = np.deg2rad(angle_front)
    
    radius = torch.rand(size, device=device) * (radius_range[1] - radius_range[0]) + radius_range[0]

    if random.random() < uniform_sphere_rate:
        unit_centers = F.normalize(
            torch.stack([
                (torch.rand(size, device=device) - 0.5) * 2.0,
                torch.rand(size, device=device),
                (torch.rand(size, device=device) - 0.5) * 2.0,
            ], dim=-1), p=2, dim=1
        )
        thetas = torch.acos(unit_centers[:,1])
        phis = torch.atan2(unit_centers[:,0], unit_centers[:,2])
        phis[phis < 0] += 2 * np.pi
        centers = unit_centers * radius.unsqueeze(-1)
    else:
        thetas = torch.rand(size, device=device) * (theta_range[1] - theta_range[0]) + theta_range[0]
        phis = torch.rand(size, device=device) * (phi_range[1] - phi_range[0]) + phi_range[0]

        centers = torch.stack([
            radius * torch.sin(thetas) * torch.sin(phis),
            radius * torch.cos(thetas),
            radius * torch.sin(thetas) * torch.cos(phis),
        ], dim=-1) # [B, 3]

    targets = 0

    # jitters
    if jitter:
        centers = centers + (torch.rand_like(centers) * 0.2 - 0.1)
        targets = targets + torch.randn_like(centers) * 0.2

    # lookat
    forward_vector = safe_normalize(centers - targets)
    up_vector = torch.FloatTensor([0, 1, 0]).to(device).unsqueeze(0).repeat(size, 1)
    right_vector = safe_normalize(torch.cross(forward_vector, up_vector, dim=-1))
    
    if jitter:
        up_noise = torch.randn_like(up_vector) * 0.02
    else:
        up_noise = 0

    up_vector = safe_normalize(torch.cross(right_vector, forward_vector, dim=-1) + up_noise)

    poses = torch.eye(4, dtype=torch.float, device=device).unsqueeze(0).repeat(size, 1, 1)
    poses[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), dim=-1)
    poses[:, :3, 3] = centers

    if return_dirs:
        dirs = get_view_direction(thetas, phis, angle_overhead, angle_front)
    else:
        dirs = None
    
    # poses, dirs = get_eg3d(size, device, radius_range, theta_range, phi_range, return_dirs, angle_overhead, angle_front, jitter, uniform_sphere_rate)
    return poses, dirs


def circle_poses(device, radius=1.25, theta=60, phi=0, return_dirs=False, angle_overhead=30, angle_front=60):
    def get_eg3d(device, radius=1.25, theta=60, phi=0, return_dirs=False, angle_overhead=30, angle_front=60):
        theta = np.deg2rad(theta)
        phi = np.deg2rad(phi)
        angle_overhead = np.deg2rad(angle_overhead)
        angle_front = np.deg2rad(angle_front)

        thetas = torch.FloatTensor([theta]).to(device)
        phis = torch.FloatTensor([phi]).to(device)

        camera_origins = torch.zeros((1, 3), device=device)

        camera_origins[:, 0:1] = radius*torch.sin(phis) * torch.cos(math.pi-thetas)
        camera_origins[:, 2:3] = radius*torch.sin(phis) * torch.sin(math.pi-thetas)
        camera_origins[:, 1:2] = radius*torch.cos(phis)

        lookat_position = torch.tensor([0, 0, 0]).to(device)
        # forward_vectors = math_utils.normalize_vecs(-camera_origins)
        forward_vectors = math_utils.normalize_vecs(lookat_position - camera_origins)
        
        poses = create_cam2world_matrix(forward_vectors, camera_origins)
        
        # print('pose: ', poses)

        return poses
    

    
    theta = np.deg2rad(theta)
    phi = np.deg2rad(phi)
    angle_overhead = np.deg2rad(angle_overhead)
    angle_front = np.deg2rad(angle_front)

    thetas = torch.FloatTensor([theta]).to(device)
    phis = torch.FloatTensor([phi]).to(device)

    centers = torch.stack([
        radius * torch.sin(thetas) * torch.sin(phis),
        radius * torch.cos(thetas),
        radius * torch.sin(thetas) * torch.cos(phis),
    ], dim=-1) # [B, 3]

    # lookat
    forward_vector = safe_normalize(centers)
    up_vector = torch.FloatTensor([0, 1, 0]).to(device).unsqueeze(0)
    right_vector = safe_normalize(torch.cross(forward_vector, up_vector, dim=-1))
    up_vector = safe_normalize(torch.cross(right_vector, forward_vector, dim=-1))

    poses = torch.eye(4, dtype=torch.float, device=device).unsqueeze(0)
    poses[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), dim=-1)
    poses[:, :3, 3] = centers

    if return_dirs:
        dirs = get_view_direction(thetas, phis, angle_overhead, angle_front)
    else:
        dirs = None
    
    # dirs = None
    # theta = 45
    # phi = 90
    # theta += .1
    # phi += .1
    
    # print('3d theta: ', theta)
    # print('3d phi: ', phi)
    
    
    # poses = get_eg3d(device, radius, theta, phi, return_dirs, angle_overhead, angle_front)
    return poses, dirs    
    

class NeRFDataset:
    def __init__(self, opt, device, type='train', H=256, W=256, size=100):
        super().__init__()
        
        self.opt = opt
        self.device = device
        self.type = type # train, val, test

        self.H = H
        self.W = W
        self.size = size

        self.training = self.type in ['train', 'all']
        
        self.cx = self.H / 2
        self.cy = self.W / 2

        self.near = self.opt.min_near
        self.far = 1000 # infinite

        # [debug] visualize poses
        # poses, dirs = rand_poses(100, self.device, radius_range=self.opt.radius_range, return_dirs=self.opt.dir_text, angle_overhead=self.opt.angle_overhead, angle_front=self.opt.angle_front, jitter=self.opt.jitter_pose, uniform_sphere_rate=1)
        # visualize_poses(poses.detach().cpu().numpy(), dirs.detach().cpu().numpy())


    def collate(self, index):

        def get_eg3d():
            ############## EG3D Style ####################
            B = len(index) # always 1
            
            # print('index: ', index)
            # print(1)

            if self.training:
                # random pose on the fly
                poses, dirs = rand_poses(B, self.device, radius_range=self.opt.radius_range, theta_range=self.opt.theta_range, return_dirs=self.opt.dir_text, angle_overhead=self.opt.angle_overhead, angle_front=self.opt.angle_front, jitter=self.opt.jitter_pose, uniform_sphere_rate=self.opt.uniform_sphere_rate)

                # random focal
                fov = random.random() * (self.opt.fovy_range[1] - self.opt.fovy_range[0]) + self.opt.fovy_range[0]
                fov_deg = 18.837
            else:
                # circle pose
                phi = (index[0] / self.size) * 360
                # phi = (1 / self.size) * 180
                # print('index: ', index[0]+1)
                # print('size: ', self.size)
                #poses, dirs = circle_poses(self.device, radius=self.opt.radius_range[1] * 1.2, theta=self.opt.val_theta, phi=phi, return_dirs=self.opt.dir_text, angle_overhead=self.opt.angle_overhead, angle_front=self.opt.angle_front)
                poses, dirs = circle_poses(self.device, radius=self.opt.val_radius, theta=self.opt.val_theta, phi=phi, return_dirs=self.opt.dir_text, angle_overhead=self.opt.angle_overhead, angle_front=self.opt.angle_front)
                # print('theta: ', self.opt.val_theta)
                # print()
                # fixed focal
                fov = (self.opt.fovy_range[1] + self.opt.fovy_range[0]) / 2
                fov_deg = 18.837

            focal = self.H / (2 * np.tan(np.deg2rad(fov) / 2))
            intrinsics = np.array([focal, focal, self.cx, self.cy])

            # intrinsic_mat = np.array([[focal/intrinsic_scale, 0, self.cx/intrinsic_scale],
            #                           [0, focal/intrinsic_scale, self.cy/intrinsic_scale],
            #                           [0, 0, 1]
            #                           ])
            
            # intrinsic_mat = torch.tensor([[[4.2634, 0.0000, 0.5000],
            #                             [0.0000, 4.2634, 0.5000],
            #                             [0.0000, 0.0000, 1.0000]]]).to(self.device)
            
            intrinsic_mat = FOV_to_intrinsics(fov, device=self.device)
            # print()
            
            projection = torch.tensor([
                [2*focal/self.W, 0, 0, 0], 
                [0, -2*focal/self.H, 0, 0],
                [0, 0, -(self.far+self.near)/(self.far-self.near), -(2*self.far*self.near)/(self.far-self.near)],
                [0, 0, -1, 0]
            ], dtype=torch.float32, device=self.device).unsqueeze(0)

            # mvp = projection @ torch.inverse(poses) # [1, 4, 4]
            mvp = None
            
            # sample a low-resolution but full image
            rays = get_rays(poses, intrinsics, self.H, self.W, -1)

            data = {
                'H': self.H,
                'W': self.W,
                'rays_o': rays['rays_o'],
                'rays_d': rays['rays_d'],
                'dir': dirs,
                'pose': poses,
                'mvp': mvp,
                'intrinsic': intrinsic_mat
            }
            
            return data
            # print(self.H, rays['rays_o'].shape)
            ############## EG3D Style ####################

        B = len(index) # always 1

        if self.training:
            # random pose on the fly
            poses, dirs = rand_poses(B, self.device, radius_range=self.opt.radius_range, theta_range=self.opt.theta_range, return_dirs=self.opt.dir_text, angle_overhead=self.opt.angle_overhead, angle_front=self.opt.angle_front, jitter=self.opt.jitter_pose, uniform_sphere_rate=self.opt.uniform_sphere_rate)

            # random focal
            fov = random.random() * (self.opt.fovy_range[1] - self.opt.fovy_range[0]) + self.opt.fovy_range[0]
        else:
            # circle pose
            phi = ((index[0] + 1) / self.size) * 90
            #poses, dirs = circle_poses(self.device, radius=self.opt.radius_range[1] * 1.2, theta=self.opt.val_theta, phi=phi, return_dirs=self.opt.dir_text, angle_overhead=self.opt.angle_overhead, angle_front=self.opt.angle_front)
            poses, dirs = circle_poses(self.device, radius=self.opt.val_radius, theta=self.opt.val_theta, phi=phi, return_dirs=self.opt.dir_text, angle_overhead=self.opt.angle_overhead, angle_front=self.opt.angle_front)

            # fixed focal
            fov = (self.opt.fovy_range[1] + self.opt.fovy_range[0]) / 2

        focal = self.H / (2 * np.tan(np.deg2rad(fov) / 2))
        intrinsics = np.array([focal, focal, self.cx, self.cy])

        intrinsic_mat = torch.tensor([[focal, 0, self.cx],
                                  [0, focal, self.cy],
                                  [0, 0, 1]
                                  ]).to(self.device)
        
        projection = torch.tensor([
            [2*focal/self.W, 0, 0, 0], 
            [0, -2*focal/self.H, 0, 0],
            [0, 0, -(self.far+self.near)/(self.far-self.near), -(2*self.far*self.near)/(self.far-self.near)],
            [0, 0, -1, 0]
        ], dtype=torch.float32, device=self.device).unsqueeze(0)

        mvp = projection @ torch.inverse(poses) # [1, 4, 4]
        
        # sample a low-resolution but full image
        rays = get_rays(poses, intrinsics, self.H, self.W, -1)
        # rays = get_rays(poses, intrinsics, self.H//2, self.W//2, -1)

        data = {
            'H': self.H,
            'W': self.W,
            'rays_o': rays['rays_o'],
            'rays_d': rays['rays_d'],
            'dir': dirs,
            'pose': poses,
            'mvp': mvp,
            'intrinsic': intrinsic_mat
        }
        # print(self.H, rays['rays_o'].shape)
        ############## EG3D Style ####################
        data = get_eg3d()
        return data


    def dataloader(self):
        loader = DataLoader(list(range(self.size)), batch_size=1, collate_fn=self.collate, shuffle=self.training, num_workers=0)
        return loader