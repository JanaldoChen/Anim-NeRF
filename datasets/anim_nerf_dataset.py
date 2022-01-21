import torch
from torch.utils.data import Dataset
import numpy as np
import os
import cv2
from torchvision import transforms as T

from utils.util import load_pickle_file

def get_pixelcoords(H, W, mask=None, subsampletype='foreground_pixel', subsamplesize=32, fore_rate=0.9, fore_erode=3):
    
    def sample(indx, indy, n_pixels):
            select_indexs = np.random.choice(indx.shape[0], n_pixels, replace=True)
            px = indx[select_indexs]
            py = indy[select_indexs]
            return px, py

    if subsampletype == 'pixel':
        indx, indy = np.meshgrid(
            np.arange(0, H),
            np.arange(0, W),
            indexing='ij'
            )
        indx = indx.flatten()
        indy = indy.flatten()
        px, py = sample(indx.flatten(), indy.flatten(), subsamplesize*subsamplesize)
        px = px.reshape(subsamplesize, subsamplesize)
        py = py.reshape(subsamplesize, subsamplesize)
    
    elif subsampletype == 'foreground_pixel':
        # foreground_pixels
        kernel = np.ones((fore_erode, fore_erode), np.uint8)
        mask_inside = cv2.erode(mask.copy(), kernel)
        mask_dilate1 = cv2.dilate(mask.copy(), kernel)
        kernel = np.ones((64, 64), np.uint8)
        mask_dilate2 = cv2.dilate(mask.copy(), kernel)
        mask_outside = mask_dilate2 - mask_dilate1

        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=2)
        indx, indy = np.where(mask_inside > 0)
        fore_pixels = int(subsamplesize*subsamplesize * fore_rate)
        fore_px, fore_py = sample(indx, indy, fore_pixels)
        indx, indy = np.where(mask_outside > 0)
        back_pixels = subsamplesize*subsamplesize - fore_pixels
        back_px, back_py = sample(indx, indy, back_pixels)
        px = np.concatenate((fore_px, back_px), axis=0).reshape(subsamplesize, subsamplesize)
        py = np.concatenate((fore_py, back_py), axis=0).reshape(subsamplesize, subsamplesize)

    else:
        px, py = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')

    pixelcoords = np.stack((px, py), axis=-1)
    return pixelcoords

def gen_ray_directions(H, W, focal, c=None):

    if c is None:
        c = [W*0.5, H*0.5]

    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H), indexing='ij')  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()

    directions = \
        torch.stack([(i-c[0])/focal[0], -(j-c[1])/focal[1], -torch.ones_like(i)], -1) # (H, W, 3)

    directions = directions / torch.norm(directions, dim=-1, keepdim=True)

    return directions

def gen_rays(c2w, H, W, focal, near, far, c=None):

    directions = gen_ray_directions(H, W, focal, c)
    
    # Rotate ray directions from camera coordinate to the world coordinate
    rays_d = directions @ c2w[:, :3].T # (H, W, 3)
    # The origin of all rays is the camera origin in world coordinate
    rays_o = c2w[:, 3].expand(rays_d.shape) # (H, W, 3)

    rays = torch.cat([rays_o, rays_d, 
                      near*torch.ones_like(rays_o[..., :1]),
                      far*torch.ones_like(rays_o[..., :1])], -1) # (h, w, 8)

    return rays
    

class AnimNeRFDatasets(Dataset):
    def __init__(self, root_dir, mode='train', cam_IDs=None, img_wh=(800, 800),
                 frame_start_ID=1, frame_end_ID=1, frame_skip=1, frame_ids_index=None, with_background=False, white_bkgd=True,
                 subsampletype='pixel', subsamplesize=32, model_type='smpl', fore_rate=0.9, fore_erode=3, **kwargs):
        self.root_dir = root_dir
        self.mode = mode
        self.cam_IDs = cam_IDs
        self.img_wh = img_wh
        self.with_background = with_background
        self.white_bkgd = white_bkgd
        self.subsampletype = subsampletype
        self.subsamplesize = subsamplesize
        self.model_type = model_type
        self.fore_rate = fore_rate
        self.fore_erode = fore_erode

        self.frame_IDs = list(range(frame_start_ID, frame_end_ID+1, frame_skip))
        self.num_frames = len(self.frame_IDs)
        
        if cam_IDs is None:
            self.num_cams = 1
        else:
            self.num_cams = len(cam_IDs)

        if frame_ids_index is None:
            frame_ids_index = {}
            for i, frame_id in enumerate(self.frame_IDs):
                frame_ids_index[frame_id] = i
        self.frame_ids_index = frame_ids_index
        
        self.dataset_size = self.num_frames * self.num_cams
        if self.mode == 'train':
            self.dataset_size *= (self.img_wh[0] * self.img_wh[1]) // (self.subsamplesize * self.subsamplesize)
            
        params_template = load_pickle_file(os.path.join(root_dir, '{}_template.pkl'.format(model_type)))
        self.body_model_params_template = {
            'betas_template': torch.from_numpy(params_template['betas']).float(),
            'global_orient_template': torch.from_numpy(params_template['global_orient']).float(),
            'body_pose_template': torch.from_numpy(params_template['body_pose']).float(),
            'transl_template': torch.from_numpy(params_template['transl']).float()
        }

        self.fg_points = torch.from_numpy(params_template['points'][params_template['distances'] < -0.02]).float()
        self.bg_points = torch.from_numpy(params_template['points'][params_template['distances'] > 0.10]).float()

        self.ToTensor = T.ToTensor()

    def __len__(self):
        return self.dataset_size

    def load_body_model_params(self, frame_ID):
        params_path = os.path.join(self.root_dir, "{}s".format(self.model_type), "{:0>6}.pkl".format(frame_ID))
        params = load_pickle_file(params_path)
        body_model_params = {
            'betas': torch.from_numpy(params['betas']).float(),
            'global_orient': torch.from_numpy(params['global_orient']).float(),
            'transl': torch.from_numpy(params['transl']).float()
        }
        if self.model_type == 'smpl':
            body_model_params.update({
                'body_pose': torch.from_numpy(params['body_pose']).float(),
            })
        elif self.model_type == 'smplh':
            body_model_params.update({
                'body_pose': torch.from_numpy(params['body_pose']).float(),
                'left_hand_pose': torch.from_numpy(params['left_hand_pose']).float(),
                'right_hand_pose': torch.from_numpy(params['right_hand_pose']).float(),
            })
        elif self.model_type == 'smplx':
            body_model_params.update({
                'body_pose': torch.from_numpy(params['body_pose']).float(),
                'expression': torch.from_numpy(params['expression']).float(),
                'left_hand_pose': torch.from_numpy(params['left_hand_pose']).float(),
                'right_hand_pose': torch.from_numpy(params['right_hand_pose']).float(),
                'jaw_pose': torch.from_numpy(params['jaw_pose']).float(),
            })
        else:
            raise ValueError(f'Unknown model type {self.model_type}, exiting!')

        return body_model_params

    def load_cam(self, cam_ID):
        camera_path = os.path.join(self.root_dir, "cam{:0>3d}".format(cam_ID), "camera.pkl")
        cam = load_pickle_file(camera_path)
        return cam

    def load_img_and_mask(self, frame_ID, cam_ID):
        image_path = os.path.join(self.root_dir, "cam{:0>3d}".format(cam_ID), "images", "{:0>6}.png".format(frame_ID))
        img = cv2.imread(image_path, -1)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        img, mask = img[..., :3], img[..., 3]
        return img, mask

    def image_transform(self, img, mask, cam, img_wh=None, undistort=True):
        if img_wh is None:
            img_wh = self.img_wh
        img = cv2.resize(img, img_wh)
        mask = cv2.resize(mask, img_wh)

        # modify focal length to match size self.img_wh
        cam['camera_f'] = cam['camera_f'] * [img_wh[0]/cam['width'], img_wh[1]/cam['height']]
        cam['camera_c'] = cam['camera_c'] * [img_wh[0]/cam['width'], img_wh[1]/cam['height']]
        cam['height'], cam['width'] = img_wh[1], img_wh[0]

        if undistort:
            K = np.eye(3)
            K[0][0] = cam['camera_f'][0]
            K[1][1] = cam['camera_f'][1]
            K[0][2] = cam['camera_c'][0]
            K[1][2] = cam['camera_c'][1]
            D = cam['camera_k'][:, np.newaxis]

            img = cv2.undistort(img, K, D)
            mask = cv2.undistort(mask, K, D)

        img = torch.from_numpy(img/255.).float().permute(2, 0, 1)
        mask = torch.from_numpy(mask/255.).float().unsqueeze(0)

        if not self.with_background:
            img = img * mask

        return img, mask, cam

    def get_rays(self, cam, near=0.1, far=10.0):
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
        return rays
    
    def get_points(self, num_points=128):
        fg_points = self.fg_points[torch.randint(0, self.fg_points.shape[0], (num_points,))]
        fg_points += torch.randn_like(fg_points) * 0.01
        bg_points = self.bg_points[torch.randint(0, self.bg_points.shape[0], (num_points, ))]
        bg_points += torch.randn_like(bg_points) * 0.01
        return fg_points, bg_points

    def __getitem__(self, idx):
        idx = idx % (self.num_frames * self.num_cams)
        frame_id = self.frame_IDs[idx % self.num_frames]
        cam_id = self.cam_IDs[idx // self.num_frames]

        cam = self.load_cam(cam_id)
        img, mask = self.load_img_and_mask(frame_id, cam_id)
        img, mask, cam = self.image_transform(img, mask, cam)

        if self.white_bkgd:
            img = img * mask + (1 - mask)

        rgbs, alphas = img.permute(1, 2, 0), mask.permute(1, 2, 0)
        rays = self.get_rays(cam)

        body_model_params = self.load_body_model_params(frame_id)
        fg_points, bg_points = self.get_points(num_points=128)

        if frame_id in self.frame_ids_index:
            frame_idx = self.frame_ids_index[frame_id]
        else:
            frame_idx = -1
        
        if self.mode == 'train':
            pixelcoords = get_pixelcoords(self.img_wh[1], self.img_wh[0], alphas.numpy(), subsampletype=self.subsampletype, subsamplesize=self.subsamplesize, fore_rate=self.fore_rate, fore_erode=self.fore_erode)
            pixelcoords = pixelcoords.reshape(-1, 2)
            rays = rays[pixelcoords[:, 0], pixelcoords[:, 1], :]
            rgbs = rgbs[pixelcoords[:, 0], pixelcoords[:, 1], :]
            alphas = alphas[pixelcoords[:, 0], pixelcoords[:, 1], :]
        else:
            rays = rays.view(-1, 8)
            rgbs = rgbs.view(-1, 3)
            alphas = alphas.view(-1, 1)

        sample = {
            'cam_id': cam_id,
            'frame_id': frame_id,
            'frame_idx': frame_idx,
            'rays': rays,
            'rgbs': rgbs,
            'alphas': alphas,
            'fg_points': fg_points,
            'bg_points': bg_points,
            **body_model_params,
            **self.body_model_params_template
            }

        return sample

if __name__ == '__main__':
    from tqdm import tqdm
    from torch.utils.data import DataLoader
    data_root = 'data'
    dataset_name = 'people_snapshot'
    people_ID = 'male-3-casual'
    frame_start_ID = 1
    frame_end_ID = 400
    frame_skip = 4
    cam_IDs = [0]
    dataset = AnimNeRFDatasets(root_dir=os.path.join(data_root, dataset_name, people_ID), mode='train', frame_start_ID=frame_start_ID, frame_end_ID=frame_end_ID, frame_skip=frame_skip, cam_IDs=cam_IDs)
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)

    for i, data in tqdm(enumerate(dataloader)):
        for key in data:
            try:
                if data[key].shape == torch.Size([1]):
                    print(key, data[key])
                else:
                    print(key, data[key].shape)
            except:
                print(key, data[key])