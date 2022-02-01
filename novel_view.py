import os
import argparse
import cv2
import glob
import imageio
import math
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from utils import visualize_depth
from utils.util import load_pickle_file
from datasets.anim_nerf_dataset import gen_rays
from torchvision.utils import save_image, make_grid

from models.anim_nerf import batch_transform
from train import AnimNeRFSystem

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_smpl_params(hparams, frame_id, params_path, template_path):

    # frame_IDs = list(range(hparams.frame_start_ID, hparams.frame_end_ID+1, hparams.frame_skip))
    frame_IDs = hparams.frame_IDs
    frame_ids_index = {}
    for i, f_id in enumerate(frame_IDs):
        frame_ids_index[f_id] = i

    body_pose_dim = 69 if hparams.model_type == 'smpl' else 63
    params = load_pickle_file(params_path)
    body_model_params = {
        'betas': torch.from_numpy(params['betas']).float(),
        'global_orient': torch.from_numpy(params['global_orient']).float(),
        'body_pose': torch.from_numpy(params['body_pose'][:body_pose_dim]).float(),
        'transl': torch.from_numpy(params['transl']).float()
    }

    params_template = load_pickle_file(template_path)
    body_model_params_template = {
        'betas': torch.from_numpy(params_template['betas']).float(),
        'global_orient': torch.from_numpy(params_template['global_orient']).float(),
        'body_pose': torch.from_numpy(params_template['body_pose'][:body_pose_dim]).float(),
        'transl': torch.from_numpy(params_template['transl']).float()
    }

    if frame_id in frame_ids_index:
        frame_idx = torch.tensor([frame_ids_index[frame_id]])
    else:
        frame_idx = torch.tensor([-1])

    return frame_idx, body_model_params, body_model_params_template

def get_cam_and_rays(hparams, camera_path, near=0.1, far=10.0):
    cam = load_pickle_file(camera_path)

    # modify focal length to match size self.img_wh
    cam['camera_f'] = cam['camera_f'] * [hparams.img_wh[0]/cam['width'], hparams.img_wh[1]/cam['height']]
    cam['camera_c'] = cam['camera_c'] * [hparams.img_wh[0]/cam['width'], hparams.img_wh[1]/cam['height']]
    cam['height'], cam['width'] = hparams.img_wh[1], hparams.img_wh[0]

    R = cam['R']
    t = cam['t']
    focal = cam['camera_f']
    c = cam['camera_c']
    h = cam['height']
    w = cam['width']

    R_ = np.array([[1., 0., 0.], [0., -1., 0.], [0., 0., -1.]]) @ R
    t_ = np.array([1, -1, -1]) * t
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R_.transpose()
    pose[:3, 3] = R_.transpose() @ -t_
    c2w = torch.from_numpy(pose[:3, :4]).float()
    rays = gen_rays(c2w, h, w, focal, near, far, c)
    rays = rays.view(-1, 8)
    return cam, rays

@torch.no_grad()
def batched_inference(volume_renderer, anim_nerf, rays, body_model_params, body_model_params_template, latent_code, P=None, chunk=2048):
    bs, n_rays = rays.shape[:2]
    results = defaultdict(list)
    
    anim_nerf.set_body_model(body_model_params, body_model_params_template)
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
    parser.add_argument('--frame_id', type=int, default=1,
                        help='frame_id for smpl and latent code')
    parser.add_argument('--cam_id', type=int, default=0,
                        help='cam_id for rays')
    parser.add_argument('--template', default=False, action='store_true',
                        help='if visualize template space')
    parser.add_argument('--orig_pose', default=False, action='store_true',
                        help='use optim pose')
    parser.add_argument('--chunk', type=int, default=2048,
                        help='chunk size')
    parser.add_argument('--dis_threshold', type=float, default=0.2,
                        help='distance threshold')
    parser.add_argument('--betas_2th', type=float, default=0,
                        help='the 2th betas')
    parser.add_argument('--n_views', type=int, default=120,
                        help='number of views')
    parser.add_argument('--angle', type=int, default=0,
                        help='the view angle')

    return parser.parse_args()

if __name__ == "__main__":
    args = get_opts()
    system = AnimNeRFSystem.load_from_checkpoint(args.ckpt_path).to(device)
    system.anim_nerf.dis_threshold = args.dis_threshold
    hparams = system.hparams
    print(hparams)

    save_dir = os.path.join(hparams.outputs_dir, hparams.exp_name, 'novel_view_{}_{}_{}'.format(args.frame_id if not args.template else 'T', 'optim_pose' if not args.orig_pose else 'orig_pose', args.angle))
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'depths'), exist_ok=True)
    
    body_model_params_dir = os.path.join(hparams.root_dir, '{}s'.format(hparams.model_type))

    frame_id = args.frame_id
    cam_id = args.cam_id
    params_path = os.path.join(body_model_params_dir, "{:0>6}.pkl".format(frame_id))
    template_path = os.path.join(hparams.root_dir, '{}_template.pkl'.format(hparams.model_type))
    camera_path = os.path.join(hparams.root_dir, "cam{:0>3d}".format(cam_id), "camera.pkl")
    frame_idx, body_model_params, body_model_params_template = get_smpl_params(hparams, frame_id, params_path, template_path)
    cam, rays = get_cam_and_rays(hparams, camera_path)
    n_rays = rays.shape[0]
    
    rays = rays.unsqueeze(0).to(device)
    for key in body_model_params:
        body_model_params[key] = body_model_params[key].unsqueeze(0).to(device)
    for key in body_model_params_template:
        body_model_params_template[key] = body_model_params_template[key].unsqueeze(0).to(device)

    frame_idx = frame_idx.to(device)

    if hparams.latent_dim > 0:
        if frame_idx != -1:
            latent_code = system.latent_codes(frame_idx)
        else:
            latent_code = system.latent_codes(torch.zeros_like(frame_idx))
    else:
        latent_code = None
    
    if not args.orig_pose and frame_idx.item() != -1:
        body_model_params = system.body_model_params(frame_idx)

    if args.template:
        body_model_params['body_pose'] = body_model_params_template['body_pose']
    
    body_model_params['betas'][:, 1] += args.betas_2th

    imgs_depths = []
    for i in tqdm(range(args.n_views)):
        R_z = cv2.Rodrigues(np.array([-math.radians(args.angle), 0., 0.]))[0]
        R_y = cv2.Rodrigues(np.array([0., 2*np.pi*i/args.n_views, 0.]))[0]
        R_ = R_y @ R_z
        P = np.eye(4, dtype=np.float32)
        P[:3, :3] = R_
        P = torch.from_numpy(P).float().unsqueeze(0).expand(n_rays, -1, -1)
        P = P.unsqueeze(0).to(device)
        with torch.no_grad():
            img, mask, depth = batched_inference(system.volume_renderer, system.anim_nerf, rays.clone(), body_model_params, body_model_params_template, latent_code, P=P, chunk=args.chunk)
        img_depth = make_grid([img, depth], nrow=2)
        img_masked = torch.cat([img, mask.unsqueeze(0)], dim=0)
        save_image(img_masked, '{}/{}/{:0>6d}.png'.format(save_dir, 'images', i))
        save_image(depth, '{}/{}/{:0>6d}.png'.format(save_dir, 'depths', i))
        imgs_depths.append((img_depth.permute(1, 2, 0).numpy()*255).astype(np.uint8))

    save_path = os.path.join(save_dir, 'novel_view.gif')
    imageio.mimsave(save_path, imgs_depths, fps=30)
    print("Saved to {}".format(save_path))