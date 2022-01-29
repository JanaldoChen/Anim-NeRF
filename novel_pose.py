import os
import argparse
import sys
import cv2
import glob
import imageio
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict

import trimesh
from utils.renderer import Renderer, WeakPerspectiveCamera
os.environ['PYOPENGL_PLATFORM'] = 'egl'

from utils import visualize_depth
from utils.util import load_pickle_file
from torchvision.utils import save_image, make_grid

from models.anim_nerf import batch_transform
from train import AnimNeRFSystem
from novel_view import get_cam_and_rays, get_smpl_params

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_mixamo_smpl(actions_dir, action_type='0007', skip=1):
    result = load_pickle_file(os.path.join(actions_dir, action_type, 'result.pkl'))

    anim_len = result['anim_len']
    pose_array = result['smpl_array'].reshape(anim_len, -1)
    cam_array = result['cam_array']
    mocap = []
    for i in range(0, anim_len, skip):
        mocap.append({
            'cam': cam_array[i],
            'global_orient': pose_array[i, :3],
            'body_pose': pose_array[i, 3:72],
            'transl': np.array([cam_array[i, 1], cam_array[i, 2], 0])
            })

    return mocap

@torch.no_grad()
def batched_inference(volume_renderer, anim_nerf, rays, smpl_params, template_poses, latent_code, P=None, chunk=2048):
    bs, n_rays = rays.shape[:2]
    results = defaultdict(list)

    anim_nerf.set_body_model(smpl_params, template_poses)
    rays = anim_nerf.convert_to_body_model_space(rays)
    anim_nerf.clac_ober2cano_transform()
    if latent_code is not None:
        anim_nerf.set_latent_code(latent_code)
    
    if P is not None:
        rays[:, :, 0:3] = batch_transform(P, rays[:, :, 0:3], pad_ones=True)
        rays[:, :, 3:6] = batch_transform(P, rays[:, :, 3:6], pad_ones=False)

    for i in range(0, n_rays, chunk):
        rays_chunk = rays[:, i:i+chunk, :]
        rendered_ray_chunks = volume_renderer(anim_nerf, rays_chunk, perturb=0.0)
        for k, v in rendered_ray_chunks.items():
            results[k] += [v]
            
    for k, v in results.items():
        results[k] = torch.cat(v, 1)

    if 'rgbs_fine' in results:
        rgbs = results['rgbs_fine']
        alphas = results['alphas_fine']
        depths = results['depths_fine']
    else:
        rgbs = results['rgbs']
        alphas = results['alphas']
        depths = results['depths']
    W, H = hparams.img_wh
    img = rgbs.cpu().view(H, W, 3).permute(2, 0, 1) # (3, H, W)
    mask = alphas.cpu().view(H, W)
    depth = visualize_depth(depths.cpu().view(H, W))
    
    return img, mask, depth

def get_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument('--ckpt_path', type=str, required=True,
                        help='pretrained checkpoint path to load')
    parser.add_argument('--actions_dir', type=str, default='mocap/mixamo/',
                        help='actions floder to load')
    parser.add_argument('--action_type', type=str, default='0007',
                        help='action type')
    parser.add_argument('--frame_id', type=int, default=1,
                        help='frame_id for smpl and latent code')
    parser.add_argument('--cam_id', type=int, default=0,
                        help='cam_id for rays')
    parser.add_argument('--frame_skip', type=int, default=2,
                        help='frame skip')
    parser.add_argument('--dis_threshold', type=float, default=0.2,
                        help='distance threshold')
    parser.add_argument('--chunk', type=int, default=2048,
                        help='chunk size')

    return parser.parse_args()

if __name__ == "__main__":
    args = get_opts()
    system = AnimNeRFSystem.load_from_checkpoint(args.ckpt_path).to(device)
    system.anim_nerf.dis_threshold = args.dis_threshold
    hparams = system.hparams

    save_dir = os.path.join(hparams.outputs_dir, hparams.exp_name, 'novel_pose_{}'.format(args.action_type))
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'masks'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'depths'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'smpls_vis'), exist_ok=True)

    body_model_params_dir = os.path.join(hparams.root_dir, '{}s'.format(hparams.model_type))

    frame_id = args.frame_id
    cam_id = args.cam_id
    params_path = os.path.join(body_model_params_dir, "{:0>6}.pkl".format(frame_id))
    template_path = os.path.join(hparams.root_dir, '{}_template.pkl'.format(hparams.model_type))
    camera_path = os.path.join(hparams.root_dir, "cam{:0>3d}".format(cam_id), "camera.pkl")
    frame_idx, smpl_params_src, template_poses = get_smpl_params(hparams, frame_id, params_path, template_path)
    cam, rays = get_cam_and_rays(hparams, camera_path)

    rays = rays.unsqueeze(0).to(device)
    for key in smpl_params_src:
        smpl_params_src[key] = smpl_params_src[key].unsqueeze(0).to(device)
    for key in template_poses:
        template_poses[key] = template_poses[key].unsqueeze(0).to(device)
    
    smpl_params_src['betas'] = system.body_model_params.betas.weight
    smpl_params_src['transl'] = system.body_model_params.transl.weight.mean(0)

    frame_idx = frame_idx.to(device)

    if hparams.latent_dim > 0:
        latent_code = system.latent_codes(frame_idx)
    else:
        latent_code = None

    H, W = cam['height'], cam['width']
    focal = cam['camera_f']
    c = cam['camera_c']
    R = cam['R']
    t = cam['t']
    renderer = Renderer(resolution=(H, W))
    renderer.set_camera(focal[0], focal[1], c[0], c[1], R, t)

    mocap = load_mixamo_smpl(args.actions_dir, args.action_type, args.frame_skip)

    imgs_depths = []
    for i in tqdm(range(len(mocap))):
        body_pose_dim = 69 if hparams.model_type == 'smpl' else 63
        smpl_params = {
            'betas': smpl_params_src['betas'],
            'global_orient': torch.from_numpy(mocap[i]['global_orient']).float().unsqueeze(0).to(device),
            'body_pose': torch.from_numpy(mocap[i]['body_pose'][:body_pose_dim]).float().unsqueeze(0).to(device),
            'transl': smpl_params_src['transl'] + torch.from_numpy(mocap[i]['transl']).float().unsqueeze(0).to(device)
        }

        img, mask, depth = batched_inference(system.volume_renderer, system.anim_nerf, rays.clone(), smpl_params, template_poses, latent_code, chunk=args.chunk)
        img_depth = make_grid([img, depth], nrow=2)
        img_masked = torch.cat([img, mask.unsqueeze(0)], dim=0)
        save_image(img_masked, '{}/{}/{:0>6d}.png'.format(save_dir, 'images', i))
        save_image(mask, '{}/{}/{:0>6d}.png'.format(save_dir, 'masks', i))
        save_image(depth, '{}/{}/{:0>6d}.png'.format(save_dir, 'depths', i))
        imgs_depths.append((img_depth.permute(1, 2, 0).numpy()*255).astype(np.uint8))

        verts = system.anim_nerf.body_model(**smpl_params)['vertices'].detach().cpu().numpy()[0]
        img_rendered = renderer.render(verts, system.anim_nerf.body_model.faces)
        cv2.imwrite(os.path.join(save_dir, 'smpls_vis', '{:0>6d}.png'.format(i)), img_rendered)

    save_path = os.path.join(save_dir, 'novel_pose.gif')
    imageio.mimsave(save_path, imgs_depths, fps=30)
    print("Saved to {}".format(save_path))