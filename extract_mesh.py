import os
import argparse
import cv2
import glob
import imageio
import math
import torch
import numpy as np
import mcubes
from tqdm import tqdm
from collections import defaultdict

import trimesh
from utils.renderer import Renderer, WeakPerspectiveCamera
os.environ['PYOPENGL_PLATFORM'] = 'egl'

from utils import visualize_depth
from utils.util import load_pickle_file, write_pickle_file, read_json, write_json
from torchvision.utils import save_image, make_grid

from models.anim_nerf import batch_transform
from train import AnimNeRFSystem
from novel_view import get_cam_and_rays, get_smpl_params

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def create_grid(N, x_range, y_range, z_range):
    xmin, xmax = x_range
    ymin, ymax = y_range
    zmin, zmax = z_range
    x = np.linspace(xmin, xmax, N)
    y = np.linspace(ymin, ymax, N)
    z = np.linspace(zmin, zmax, N)
    grid = np.stack(np.meshgrid(x, y, z), -1)
    return grid

def mcubes_to_world(vertices, N, x_range, y_range, z_range):
    xmin, xmax = x_range
    ymin, ymax = y_range
    zmin, zmax = z_range
    vertices_ = vertices / N
    x_ = (ymax-ymin) * vertices_[:, 1] + ymin
    y_ = (xmax-xmin) * vertices_[:, 0] + xmin
    vertices_[:, 0] = x_
    vertices_[:, 1] = y_
    vertices_[:, 2] = (zmax-zmin) * vertices_[:, 2] + zmin
    return vertices_

@torch.no_grad()
def batched_inference(anim_nerf, points, chunk=32*32*64):
    bs, nv = points.shape[:2]

    sigmas = []
    for i in range(0, nv, chunk):
        xyz_chunk = points[:, i:i+chunk, :]
        dir_chunk = torch.zeros_like(xyz_chunk)
        rgb_chunk, sigma_chunk = anim_nerf(xyz_chunk, dir_chunk, use_fine=anim_nerf.use_fine)
        sigmas.append(torch.relu(sigma_chunk))
    sigmas = torch.cat(sigmas, 1)

    return sigmas

def get_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument('--ckpt_path', type=str, required=True,
                        help='pretrained checkpoint path to load')
    parser.add_argument('--frame_id', type=int, default=1,
                        help='frame_id for smpl and latent code')
    parser.add_argument('--cam_id', type=int, default=0,
                        help='cam_id for rays')
    parser.add_argument('--template', default=False, action='store_true',
                        help='if visualize template space')
    parser.add_argument('--orig_pose', default=False, action='store_true',
                        help='use optim pose')
    parser.add_argument('--chunk', type=int, default=32*32*64,
                        help='chunk size')
    parser.add_argument('--N_grid', type=int, default=256,
                        help='size of the grid on 1 side, larger=higher resolution')
    parser.add_argument('--x_range', nargs="+", type=float, default=[-1.2, 1.2],
                        help='x range of the object')
    parser.add_argument('--y_range', nargs="+", type=float, default=[-1.2, 1.2],
                        help='x range of the object')
    parser.add_argument('--z_range', nargs="+", type=float, default=[-1.2, 1.2],
                        help='x range of the object')
    parser.add_argument('--sigma_threshold', type=float, default=20.0,
                        help='threshold to consider a location is occupied')
    parser.add_argument('--dis_threshold', type=float, default=0.2,
                        help='distance threshold')
    parser.add_argument('--smooth', default=True, action='store_true',
                        help='if smooth the mesh')
    parser.add_argument('--vis', default=False, action='store_true',
                        help='if visualize')
    parser.add_argument('--n_views', type=int, default=120,
                        help='number of views')
    # parser.add_argument('--angle', type=int, default=0,
    #                     help='the view angle')

    return parser.parse_args()

if __name__ == "__main__":
    args = get_opts()
    system = AnimNeRFSystem.load_from_checkpoint(args.ckpt_path).to(device)
    system.anim_nerf.dis_threshold = args.dis_threshold
    hparams = system.hparams
    print(hparams)

    save_dir = os.path.join(hparams.outputs_dir, hparams.exp_name, 'mesh_{}_{}'.format(args.frame_id if not args.template else 'T', 'optim_pose' if not args.orig_pose and hparams.optim_body_params else 'orig_pose'))
    os.makedirs(save_dir, exist_ok=True)

    body_model_params_dir = os.path.join(hparams.root_dir, '{}s'.format(hparams.model_type))

    frame_id = args.frame_id
    cam_id = args.cam_id
    params_path = os.path.join(body_model_params_dir, "{:0>6}.pkl".format(frame_id))
    template_path = os.path.join(hparams.root_dir, '{}_template.pkl'.format(hparams.model_type))
    camera_path = os.path.join(hparams.root_dir, "cam{:0>3d}".format(cam_id), "camera.pkl")
    frame_idx, body_model_params, body_model_params_template = get_smpl_params(hparams, frame_id, params_path, template_path)
    cam, rays = get_cam_and_rays(hparams, camera_path)

    rays = rays.unsqueeze(0).to(device)
    for key in body_model_params:
        body_model_params[key] = body_model_params[key].unsqueeze(0).to(device)
    for key in body_model_params_template:
        body_model_params_template[key] = body_model_params_template[key].unsqueeze(0).to(device)

    frame_idx = frame_idx.to(device)

    if hparams.latent_dim > 0:
        latent_code = system.latent_codes(frame_idx)
        system.anim_nerf.set_latent_code(latent_code)
    
    if not args.orig_pose and hparams.optim_body_params and frame_idx.item() != -1:
        body_model_params = system.body_model_params(frame_idx)

    if args.template:
        body_model_params['global_orient'] = body_model_params_template['global_orient']
        body_model_params['body_pose'] = body_model_params_template['body_pose']
        body_model_params['transl'] = body_model_params_template['transl']
        body_model_params['betas'] = body_model_params_template['betas']

    system.anim_nerf.set_body_model(body_model_params, body_model_params_template)
    global_transform_inv = torch.inverse(system.anim_nerf.global_transform).detach().squeeze(0).cpu().numpy()
    rays = system.anim_nerf.convert_to_body_model_space(rays)
    system.anim_nerf.clac_ober2cano_transform()

    smpl_verts = system.anim_nerf.verts.detach().squeeze(0).cpu().numpy()
    smpl_faces = system.anim_nerf.body_model.faces
    save_path = os.path.join(save_dir, 'smpl.obj')
    mcubes.export_obj(smpl_verts, smpl_faces, save_path)

    N = args.N_grid
    grid = create_grid(args.N_grid, args.x_range, args.y_range, args.z_range)
    with torch.no_grad():
        points = torch.from_numpy(grid.reshape(-1, 3)).unsqueeze(0).float().to(device) # (1, N*N*N, 3)
        center = (system.anim_nerf.verts.max(dim=1)[0] + system.anim_nerf.verts.min(dim=1)[0]) / 2.
        points += center
        sigmas = batched_inference(system.anim_nerf, points, chunk=args.chunk)
    sigmas = sigmas.cpu().numpy()
    sigmas = np.maximum(sigmas, 0).reshape(N, N, N)
    sigmas = sigmas - args.sigma_threshold

    if args.smooth:
        sigmas = mcubes.smooth(sigmas)

    vertices, faces = mcubes.marching_cubes(-sigmas, 0.)
    vertices = mcubes_to_world(vertices, args.N_grid, args.x_range, args.y_range, args.z_range)
    vertices += center.squeeze(0).cpu().numpy()

    # vertices = (np.matmul(global_transform[:3, :3], vertices.transpose()) + global_transform[:3, 3:4]).transpose()

    save_path = os.path.join(save_dir, 'mesh.obj')
    mcubes.export_obj(vertices, faces, save_path)
    print("Saved to {}".format(save_path))
    # save_path = os.path.join(hparams.outputs_dir, hparams.exp_name, 'mesh_{}_{}.pkl'.format(frame_id if not args.template else 'T', 'optim_pose' if not args.orig_pose and hparams.optim_pose else 'orig_pose'))
    # write_pickle_file(save_path, {'global_transform': global_transform})
    
    if args.vis:
        os.makedirs(os.path.join(save_dir, 'images'), exist_ok=True)
        H, W = cam['height'], cam['width']
        focal = cam['camera_f']
        c = cam['camera_c']
        R = cam['R']
        t = cam['t']
        renderer = Renderer(resolution=(H, W))

        R = np.array([[1., 0., 0.], [0., -1., 0.], [0., 0., -1.]]) @ R
        t = t * [1, -1, -1]

        R = global_transform_inv[:3, :3] @ R
        t = global_transform_inv[:3, 3] + t
        renderer.set_camera(focal[0], focal[1], c[0], c[1], R, t)

        imgs_rendered = []
        for i in tqdm(range(args.n_views)):
            img_rendered = renderer.render(vertices, faces, angle=-i/args.n_views*360, axis=[0, 1, 0])
            cv2.imwrite(os.path.join(save_dir, 'images', '{:0>6d}.png'.format(i)), img_rendered)
            imgs_rendered.append(img_rendered)

        save_path = os.path.join(save_dir, '3d_rec.gif')
        imageio.mimsave(save_path, imgs_rendered, fps=30)
        print("Saved to {}".format(save_path))