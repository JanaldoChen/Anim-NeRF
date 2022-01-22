import os
import h5py
import numpy as np
from tqdm import tqdm
import cv2
import shutil
import torch
import subprocess
import argparse

from utils.util import load_pickle_file, write_pickle_file
from smplx import body_models, create

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

def get_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_root', type=str, default='/home/janaldo/Janaldo_workspace/Datasets/people_snapshot_public',
                        help='people snapshot datasets root')
    parser.add_argument('--people_ID', type=str, default='male-3-casual',
                        help='people id')
    parser.add_argument('--gender', type=str, default='male',
                        help='gender')
    parser.add_argument('--output_dir', type=str, default='data/people_snapshot',
                        help='data to save')

    return parser.parse_args()


if __name__ == "__main__":
    args = get_opts()
    data_root = args.data_root
    people_ID = args.people_ID
    gender = args.gender
    output_dir = args.output_dir

    vid_file = os.path.join(data_root, people_ID, people_ID+'.mp4')
    images_dir = os.path.join(output_dir, people_ID, 'cam000', 'images')
    smpls_dir = os.path.join(output_dir, people_ID, 'smpls')
    if os.path.exists(images_dir):
        shutil.rmtree(images_dir)
    os.makedirs(images_dir, exist_ok=True)
    if os.path.exists(smpls_dir):
        shutil.rmtree(smpls_dir)
    os.makedirs(smpls_dir, exist_ok=True)

    command = ['ffmpeg', '-i', vid_file, '-f', 'image2', '-v', 'error', f'{images_dir}/%06d.png']
    subprocess.call(command)

    frame_IDs = os.listdir(images_dir)
    frame_IDs = [id.split('.')[0] for id in frame_IDs]
    frame_IDs.sort()

    camera_pkl = load_pickle_file(os.path.join(data_root, people_ID, 'camera.pkl'))
    camera = {
        'R': cv2.Rodrigues(camera_pkl['camera_rt'])[0],
        't': camera_pkl['camera_t'],
        'camera_f': camera_pkl['camera_f'],
        'camera_c': camera_pkl['camera_c'],
        'camera_k': camera_pkl['camera_k'],
        'height': camera_pkl['height'],
        'width': camera_pkl['width'],
    }
    write_pickle_file(os.path.join(output_dir, people_ID, 'cam000', 'camera.pkl'), camera)

    consensus_pkl = load_pickle_file(os.path.join(data_root, people_ID, 'consensus.pkl'))
    reconstructed_poses = h5py.File(os.path.join(data_root, people_ID, 'reconstructed_poses.hdf5'), 'r')
    masks = h5py.File(os.path.join(data_root, people_ID, 'masks.hdf5'), 'r')

    betas = consensus_pkl['betas']
    v_personal = consensus_pkl['v_personal']

    for frame_ID in tqdm(frame_IDs, desc=people_ID):
        img = cv2.imread(os.path.join(images_dir, frame_ID+'.png'))
        mask = masks['masks'][int(frame_ID) - 1]

        img_masked = np.concatenate((img, mask[:, :, np.newaxis]*255), axis=-1)
        cv2.imwrite(os.path.join(images_dir, frame_ID+'.png'), img_masked)

        pose = reconstructed_poses['pose'][int(frame_ID)-1]
        trans = reconstructed_poses['trans'][int(frame_ID)-1]
        smpl_params = {
            'betas': betas,
            'global_orient': pose[:3],
            'body_pose': pose[3:],
            'transl': trans,
            'v_personal': v_personal,
            'model_type': 'smpl',
            'gender': gender,
        }

        write_pickle_file(os.path.join(smpls_dir, frame_ID+'.pkl'), smpl_params)
    
    
