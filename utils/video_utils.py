# -*- coding: utf-8 -*-
# @Time    : 2019-08-02 18:31
# @Author  : Zhixin Piao 
# @Email   : piaozhx@shanghaitech.edu.cn

import os
import glob
import cv2
import shutil
import subprocess
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm
import numpy as np


def auto_unzip_fun(x, f):
    return f(*x)

def video_to_images(vid_file, img_folder=None, return_info=False):
    if img_folder is None:
        img_folder = osp.join('/tmp', osp.basename(vid_file).replace('.', '_'))

    os.makedirs(img_folder, exist_ok=True)

    command = ['ffmpeg',
               '-i', vid_file,
               '-f', 'image2',
               '-v', 'error',
               f'{img_folder}/%06d.png']
    print(f'Running \"{" ".join(command)}\"')
    subprocess.call(command)

    print(f'Images saved to \"{img_folder}\"')

    img_shape = cv2.imread(osp.join(img_folder, '000001.png')).shape

    if return_info:
        return img_folder, len(os.listdir(img_folder)), img_shape
    else:
        return img_folder

def images_to_video(img_folder, output_vid_file):
    os.makedirs(img_folder, exist_ok=True)

    command = [
        'ffmpeg', '-y', '-i', f'{img_folder}/%06d.png', '-profile:v', 'baseline',
        '-level', '3.0', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-an', '-v', 'error', output_vid_file,
    ]

    print(f'Running \"{" ".join(command)}\"')
    subprocess.call(command)
    

def make_video(output_mp4_path, img_path_list, save_frames_dir=None, fps=45):
    """
    output_path is the final mp4 name
    img_dir is where the images to make into video are saved.
    """

    first_img = cv2.imread(img_path_list[0])
    h, w = first_img.shape[:2]

    pool_size = 40
    tmp_avi_video_path = '%s.avi' % output_mp4_path
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    videoWriter = cv2.VideoWriter(tmp_avi_video_path, fourcc, fps, (w, h))
    args_list = [(img_path,) for img_path in img_path_list]
    with Pool(pool_size) as p:
        for img in tqdm(p.imap(partial(auto_unzip_fun, f=cv2.imread), args_list), total=len(args_list)):
            videoWriter.write(img)
    videoWriter.release()

    if save_frames_dir:
        for i, img_path in enumerate(img_path_list):
            shutil.copy(img_path, '%s/%.8d.jpg' % (save_frames_dir, i))

    # os.system("ffmpeg -y -i %s -vcodec h264 %s > /dev/null 2>&1" % (tmp_avi_video_path, output_mp4_path))
    os.system("ffmpeg -i %s -b:v 5.0M %s" % (tmp_avi_video_path, output_mp4_path))
    os.system("rm %s" % (tmp_avi_video_path))


def fuse_image(img_path_list, row_num, col_num):
    assert len(img_path_list) == row_num * col_num

    img_list = [cv2.imread(img_path) for img_path in img_path_list]

    row_imgs = []
    for i in range(row_num):
        col_imgs = img_list[i * col_num: (i + 1) * col_num]
        col_img = np.concatenate(col_imgs, axis=1)
        row_imgs.append(col_img)

    fused_img = np.concatenate(row_imgs, axis=0)
    return fused_img


def fuse_video(video_frames_path_list, output_mp4_path, row_num, col_num, fps=24):
    assert len(video_frames_path_list) == row_num * col_num

    frame_num = len(video_frames_path_list[0])
    first_img = cv2.imread(video_frames_path_list[0][0])
    h, w = first_img.shape[:2]
    fused_h, fused_w = h * row_num, w * col_num

    args_list = []
    for frame_idx in range(frame_num):
        fused_frame_path_list = [video_frames[frame_idx] for video_frames in video_frames_path_list]
        args_list.append((fused_frame_path_list, row_num, col_num))

    pool_size = 40
    tmp_avi_video_path = '%s.avi' % output_mp4_path
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    # for args in args_list:
    #     fuse_image(*args)
    # exit()

    videoWriter = cv2.VideoWriter(tmp_avi_video_path, fourcc, fps, (fused_w, fused_h))
    with Pool(pool_size) as p:
        for img in tqdm(p.imap(partial(auto_unzip_fun, f=fuse_image), args_list), total=len(args_list)):
            videoWriter.write(img)
    videoWriter.release()

    os.system("ffmpeg -y -i %s -vcodec h264 %s > /dev/null 2>&1" % (tmp_avi_video_path, output_mp4_path))
    # os.system("rm %s" % (tmp_avi_video_path))
