from urllib import parse
import torch
import os
import argparse
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import imageio
from torch.utils.data import DataLoader, dataloader
from torchvision.utils import save_image

# datasets
from datasets import dataset_dict
# optimizer, scheduler, visualization
from utils import *
# metrics
from models.evaluator import Evaluator

from train import AnimNeRFSystem
from train import AnimNeRFData
from utils.util import load_pickle_file

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument('--ckpt_path', type=str, required=True,
                        help='pretrained checkpoint path to load')
    parser.add_argument('--result_dir', type=str, default='./results', 
                        help='the results folder to save images')
    parser.add_argument('--vis', default=False, action='store_true',
                        help='if visualize')

    return parser.parse_args()

if __name__ == "__main__":
    args = get_opts()
    system = AnimNeRFSystem.load_from_checkpoint(args.ckpt_path, strict=False).to(device)
    hparams = system.hparams
    data = AnimNeRFData(hparams)
    data.setup()
    dataloader = data.test_dataloader()
    print(hparams)

    save_dir = os.path.join(args.result_dir, hparams.exp_name)
    os.makedirs(save_dir, exist_ok=True)
    W, H = hparams.img_wh

    evaluator =  Evaluator().to(device)

    test_psnr = []
    test_ssim = []
    test_lpips = []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            for key in batch:
                batch[key] = batch[key].to(device)
            frame_id, cam_id, frame_idx, rays, rgbs, alphas, body_model_params, body_model_params_template, _, _ = system.decode_batch(batch)
            if system.hparams.latent_dim > 0:
                if frame_idx != -1:
                    latent_code = system.latent_codes(frame_idx)
                else:
                    latent_code = system.latent_codes(torch.zeros_like(frame_idx))
            else:
                latent_code = None
            if system.hparams.optim_body_params and frame_idx != -1:
                body_model_params = system.body_model_params(frame_idx)
            # else:
            #     body_model_params['betas'] = system.body_model_params.betas(torch.zeros_like(frame_idx))
            results = system.forward(rays, body_model_params, body_model_params_template, latent_code=latent_code)

            if 'rgbs_fine' in results:
                img = results['rgbs_fine'].view(-1, H, W, 3).permute(0, 3, 1, 2) # (1, 3, H, W)
            else:
                img = results['rgbs'].view(-1, H, W, 3).permute(0, 3, 1, 2) # (1, 3, H, W)

            img_gt = rgbs.view(-1, H, W, 3).permute(0, 3, 1, 2) # (1, 3, H, W)

            test_metrics = evaluator(img, img_gt)
            test_psnr.append(test_metrics['psnr'].item())
            test_ssim.append(test_metrics['ssim'].item())
            test_lpips.append(test_metrics['lpips'].item())

            cam_id, frame_id = cam_id.item(), frame_id.item()
            print("[{}/{}] Camera ID {:0>3d} Frame ID {:0>6d}: psnr={:.2f}, ssim={:.4f}, lpips={:.4f}".format(i, len(hparams.frame_IDs), cam_id, frame_id, test_metrics['psnr'].item(), test_metrics['ssim'].item(), test_metrics['lpips'].item()))
            if args.vis:
                os.makedirs(os.path.join(save_dir, 'cam{:0>3d}'.format(cam_id)), exist_ok=True)
                save_image(img, os.path.join(save_dir, 'cam{:0>3d}'.format(cam_id), '{:0>6d}.png'.format(frame_id)))

    print("[{}] Test PSNR: {}".format(hparams.exp_name, np.mean(test_psnr)))
    print("[{}] Test SSIM: {}".format(hparams.exp_name, np.mean(test_ssim)))
    print("[{}] Test LPIPS: {}".format(hparams.exp_name, np.mean(test_lpips)))
            

