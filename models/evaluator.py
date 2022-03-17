import torch
import torch.nn as nn
import torch.nn.functional as F

import lpips
from torchmetrics.functional.image.ssim import structural_similarity_index_measure
from torchmetrics.functional.image.psnr import peak_signal_noise_ratio

class Evaluator(nn.Module):
    def __init__(self):
        super(Evaluator, self).__init__()
        self.psnr = peak_signal_noise_ratio
        self.ssim = structural_similarity_index_measure
        self.lpips = lpips.LPIPS(net='alex')

    def forward(self, img, img_gt):
        
        img_pnsr = self.psnr(img, img_gt, data_range=1.0)
        img_ssim = self.ssim(img, img_gt)
        img_lpips = self.lpips(img, img_gt)

        result = {
            'psnr': img_pnsr,
            'ssim': img_ssim,
            'lpips': img_lpips,
        }

        return result