import cv2
import argparse
import numpy as np
from PIL import Image
import torchvision.transforms as T

import torch
import torch.nn as nn
import torch.nn.functional as F

# optimizer
from torch.optim import SGD, Adam
# scheduler
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR, LambdaLR

# def get_optimizer(hparams, models):
#     eps = 1e-8
#     parameters = []
#     for model in models:
#         parameters += list(model.parameters())

#     if hparams.optimizer == 'sgd':
#         optimizer = SGD(parameters, lr=hparams.lr, 
#                         momentum=hparams.momentum, weight_decay=hparams.weight_decay)
#     elif hparams.optimizer == 'adam':
#         optimizer = Adam(parameters, lr=hparams.lr, eps=eps, 
#                          weight_decay=hparams.weight_decay)
#     else:
#         raise ValueError('optimizer not recognized!')

#     return optimizer

def get_optimizer(hparams, parameters):
    eps = 1e-8
    if hparams.optimizer.type == 'sgd':
        optimizer = SGD(parameters, lr=hparams.lr, 
                        momentum=hparams.optimizer.momentum, weight_decay=hparams.optimizer.weight_decay)
    elif hparams.optimizer.type == 'adam':
        optimizer = Adam(parameters, lr=hparams.lr, eps=eps, 
                         weight_decay=hparams.optimizer.weight_decay)
    else:
        raise ValueError('optimizer not recognized!')

    return optimizer

def get_scheduler(hparams, optimizer):
    eps = 1e-8
    if hparams.scheduler.type == 'steplr':
        scheduler = MultiStepLR(optimizer, milestones=hparams.scheduler.decay_step, 
                                gamma=hparams.scheduler.decay_gamma)
    elif hparams.scheduler.type == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=hparams.max_epochs, eta_min=eps)
    elif hparams.scheduler.type == 'poly':
        scheduler = LambdaLR(optimizer, lambda epoch: (1-epoch/hparams.max_epochs)**hparams.scheduler.poly_exp)
    else:
        raise ValueError('scheduler not recognized!')

    return scheduler

# def get_scheduler(hparams, optimizer):
#     eps = 1e-8
#     if hparams.lr_scheduler == 'steplr':
#         scheduler = MultiStepLR(optimizer, milestones=hparams.decay_step, 
#                                 gamma=hparams.decay_gamma)
#     elif hparams.lr_scheduler == 'cosine':
#         scheduler = CosineAnnealingLR(optimizer, T_max=hparams.max_epochs, eta_min=eps)
#     elif hparams.lr_scheduler == 'poly':
#         scheduler = LambdaLR(optimizer, lambda epoch: (1-epoch/hparams.max_epochs)**hparams.poly_exp)
#     else:
#         raise ValueError('scheduler not recognized!')

#     return scheduler

def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def extract_model_state_dict(ckpt_path, model_name='model', prefixes_to_ignore=[]):
    checkpoint = torch.load(ckpt_path)
    checkpoint_ = {}
    if 'state_dict' in checkpoint: # if it's a pytorch-lightning checkpoint
        checkpoint = checkpoint['state_dict']
    for k, v in checkpoint.items():
        if not k.startswith(model_name+'.'):
            continue
        k = k[len(model_name)+1:]
        for prefix in prefixes_to_ignore:
            if k.startswith(prefix):
                print('ignore', k)
                break
        else:
            checkpoint_[k] = v
    return checkpoint_

def load_ckpt(model, ckpt_path, model_name='model', prefixes_to_ignore=[]):
    model_dict = model.state_dict()
    checkpoint_ = extract_model_state_dict(ckpt_path, model_name, prefixes_to_ignore)
    model_dict.update(checkpoint_)
    model.load_state_dict(model_dict)

def load_hparams(ckpt_path):
    checkpoint = torch.load(ckpt_path)
    hparams = checkpoint['hyper_parameters']
    hparams = argparse.Namespace(**hparams)
    return hparams

def visualize_depth_numpy(depths, cmap=cv2.COLORMAP_JET, near=4.5, far=6.5):
    x = np.clip(depths, near, far)
    x = (x-near)/(far-near+1e-8) # normalize to 0~1
    x = (255*x).astype(np.uint8)
    x = cv2.applyColorMap(x, cmap)
    return x

# def visualize_depth(depths, cmap=cv2.COLORMAP_JET, near=4.5, far=6.5):
#     """
#     depths: (H, W)
#     """
#     x = depths.numpy()
#     x = np.clip(x, near, far)
#     x = (x-near)/(far-near+1e-8) # normalize to 0~1
#     x = (255*x).astype(np.uint8)
#     x_ = Image.fromarray(cv2.applyColorMap(x, cmap))
#     x_ = T.ToTensor()(x_) # (3, H, W)
#     return x_

def visualize_depth(depths, cmap=cv2.COLORMAP_JET):
    """
    depths: (H, W)
    """
    x = depths.numpy()
    x = np.nan_to_num(x)
    ma = np.max(x)
    mi = min(np.min(x), ma-2.)
    x = (x-mi)/(ma-mi+1e-8) # normalize to 0~1
    x = (255*x).astype(np.uint8)
    x_ = Image.fromarray(cv2.applyColorMap(x, cmap))
    x_ = T.ToTensor()(x_) # (3, H, W)
    return x_

def visualize_alpha(alphas, cmap=cv2.COLORMAP_JET):
    """
    alphas: (H, W)
    """
    x = alphas.numpy()
    x = np.clip(x, 0., 1.)
    x = (255*x).astype(np.uint8)
    x_ = Image.fromarray(cv2.applyColorMap(x, cmap))
    x_ = T.ToTensor()(x_) # (3, H, W)
    return x_

def visualize(img_gt, img_pred, depth):
    vis_list = []             
    img_pred = img_pred.squeeze(0).cpu() # (3, H, W)
    img_gt = img_gt.squeeze(0).cpu() # (3, H, W)
    depth = visualize_depth(depth.squeeze(0).cpu())
    depth = F.interpolate(depth.unsqueeze(0), size=img_gt.shape[-2:]).squeeze(0)
    vis_list += [img_gt, img_pred, depth] # (3, 3, H, W)
    res_vis = torch.stack(vis_list) # (4, 3, H, W)
    return res_vis