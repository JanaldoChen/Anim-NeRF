import os

import cv2
import pickle
import shutil
import argparse
import torch
import numpy as np
from tqdm import tqdm

import sys
sys.path.append('third_party/RobustVideoMatting')
from model import MattingNetwork
from inference_utils import ImageSequenceReader, ImageSequenceWriter

device = torch.device('cuda:0')
EXTS = ['jpg', 'jpeg', 'png']

def main(args):

    segmentor = MattingNetwork(variant='resnet50').eval().to(device)
    segmentor.load_state_dict(torch.load(args.ckpt_path))

    images_folder = args.images_folder
    output_folder = args.output_folder

    frame_IDs = os.listdir(images_folder)
    frame_IDs = [id.split('.')[0] for id in frame_IDs if id.split('.')[-1] in EXTS]
    frame_IDs.sort()

    frame_IDs = frame_IDs[:4][::-1] + frame_IDs

    rec = [None] * 4                                       # Initial recurrent 
    downsample_ratio = 1.0                                 # Adjust based on your video.   

    for i in tqdm(range(len(frame_IDs))):
        frame_ID = frame_IDs[i]
        img_path = os.path.join(images_folder, '{}.png'.format(frame_ID))
        img_masked_path = os.path.join(output_folder, '{}.png'.format(frame_ID))
        img = cv2.imread(img_path)
        src = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        src = torch.from_numpy(src).float() / 255.
        src = src.permute(2, 0, 1).unsqueeze(0)
        with torch.no_grad():
            fgr, pha, *rec = segmentor(src.to(device), *rec, downsample_ratio)  # Cycle the recurrent states.
        pha = pha.permute(0, 2, 3, 1).cpu().numpy().squeeze(0)
        mask = (pha > 0.5).astype(np.int32)
        mask = (mask * 255).astype(np.uint8)
        img_masked = np.concatenate([img, mask], axis=-1)
        cv2.imwrite(img_masked_path, img_masked)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_folder', type=str,
                        help='the images folder for segmentation')
    parser.add_argument('--output_folder', type=str,
                        help='the output folder to save results')
    parser.add_argument('--ckpt_path', type=str, default='third_party/RobustVideoMatting/checkpoints/rvm_resnet50.pth',
                        help='the checkpoints for rvm')
    
    args = parser.parse_args()

    main(args)