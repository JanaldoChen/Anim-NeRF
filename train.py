import os
import sys
import math
import numpy as np
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from config import get_cfg
# models
from models.volume_rendering import VolumeRenderer
from models.anim_nerf import AnimNeRF
from models.body_model_params import BodyModelParams
# losses
# datasets
from datasets import dataset_dict
# optimizer, scheduler, visualization
from utils import *
from utils.util import load_pickle_file


# pytorch-lightning
from torchmetrics.functional import psnr
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import TestTubeLogger


class AnimNeRFData(LightningDataModule):
    def __init__(self, hparams):
        super(AnimNeRFData, self).__init__()
        self.hparams = hparams
    
    def setup(self, stage=None):
        dataset = dataset_dict[self.hparams.dataset_name]

        if self.hparams.deformation_dim + self.hparams.apperance_dim > 0 or self.hparams.optim_body_params:
            frame_ids_index = {}
            for i, frame_id in enumerate(self.hparams.frame_IDs):
                frame_ids_index[frame_id] = i
        else:
            frame_ids_index = None
        
        kwargs = {'root_dir': self.hparams.root_dir,
               'img_wh': tuple(self.hparams.img_wh),
               'frame_start_ID': self.hparams.train.frame_start_ID,
               'frame_end_ID': self.hparams.train.frame_end_ID,
               'frame_skip': self.hparams.train.frame_skip,
               'subsampletype': self.hparams.train.subsampletype,
               'subsamplesize': self.hparams.train.subsamplesize,
               'model_type': self.hparams.model_type,
               'cam_IDs': self.hparams.train.cam_IDs
               }
        self.train_dataset = dataset(mode='train', frame_ids_index=frame_ids_index, **kwargs)

        kwargs = {'root_dir': self.hparams.root_dir,
               'img_wh': tuple(self.hparams.img_wh),
               'frame_start_ID': self.hparams.val.frame_start_ID,
               'frame_end_ID': self.hparams.val.frame_end_ID,
               'frame_skip': self.hparams.val.frame_skip,
               'model_type': self.hparams.model_type,
               'cam_IDs': self.hparams.val.cam_IDs
               }
        self.val_dataset = dataset(mode='val', frame_ids_index=frame_ids_index, **kwargs)

        kwargs = {'root_dir': self.hparams.root_dir,
               'img_wh': tuple(self.hparams.img_wh),
               'frame_start_ID': self.hparams.test.frame_start_ID,
               'frame_end_ID': self.hparams.test.frame_end_ID,
               'frame_skip': self.hparams.test.frame_skip,
               'model_type': self.hparams.model_type,
               'cam_IDs': self.hparams.test.cam_IDs
               }
        self.test_dataset = dataset(mode='val', frame_ids_index=frame_ids_index, **kwargs)
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          shuffle=True,
                          num_workers=self.hparams.train.num_workers,
                          batch_size=self.hparams.train.batch_size,
                          pin_memory=False)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          shuffle=False,
                          num_workers=self.hparams.val.num_workers,
                          batch_size=self.hparams.val.batch_size, # validate one image (H*W rays) at a time
                          pin_memory=False)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          shuffle=False,
                          num_workers=self.hparams.test.num_workers,
                          batch_size=self.hparams.test.batch_size, # validate one image (H*W rays) at a time
                          pin_memory=False)

class AnimNeRFSystem(LightningModule):
    def __init__(self, hparams):
        super(AnimNeRFSystem, self).__init__()
        self.hparams = hparams
        
        self.anim_nerf = AnimNeRF(
            model_path=self.hparams.model_path,
            model_type=self.hparams.model_type,
            gender=self.hparams.gender,
            freqs_xyz=self.hparams.freqs_xyz,
            freqs_dir=self.hparams.freqs_dir,
            use_view=self.hparams.use_view,
            k_neigh=self.hparams.k_neigh,
            use_knn=self.hparams.use_knn,
            use_unpose=self.hparams.use_unpose,
            unpose_view=self.hparams.unpose_view,
            use_deformation=self.hparams.use_deformation,
            pose_dim=self.hparams.pose_dim,
            deformation_dim=self.hparams.deformation_dim,
            apperance_dim=self.hparams.apperance_dim,
            use_fine=self.hparams.n_importance>0 or self.hparams.n_depth>0,
            share_fine=self.hparams.share_fine,
            dis_threshold=self.hparams.dis_threshold,
            query_inside=self.hparams.query_inside,
            )

        self.models = [self.anim_nerf]

        if self.hparams.deformation_dim > 0 or self.hparams.apperance_dim > 0:
            self.hparams.latent_dim = self.hparams.deformation_dim + self.hparams.apperance_dim
            self.latent_codes = nn.Embedding(self.hparams.num_frames, self.hparams.latent_dim)
            self.latent_codes.weight.data.normal_(0, 0.1)
            self.models += [self.latent_codes]

        self.body_model_params = BodyModelParams(self.hparams.num_frames, model_type=self.hparams.model_type)
        self.load_body_model_params()
        if self.hparams.optim_body_params:
            optim_params = self.body_model_params.param_names
            for param_name in optim_params:
                self.body_model_params.set_requires_grad(param_name, requires_grad=True)
        self.models += [self.body_model_params]

        self.volume_renderer = VolumeRenderer(n_coarse=self.hparams.n_samples, n_fine=self.hparams.n_importance, n_fine_depth=self.hparams.n_depth, share_fine=self.hparams.share_fine, white_bkgd=self.hparams.white_bkgd)

    def load_body_model_params(self):
        body_model_params = {param_name: [] for param_name in self.body_model_params.param_names}
        if self.hparams.train.cam_IDs is not None:
            body_model_params_dir = os.path.join(self.hparams.root_dir, 'cam{:0>3d}'.format(self.hparams.train.cam_IDs[0]), '{}s'.format(self.hparams.model_type))
        else:
            body_model_params_dir = os.path.join(self.hparams.root_dir, '{}s'.format(self.hparams.model_type))
            
        for frame_id in self.hparams.frame_IDs:
            params = load_pickle_file(os.path.join(body_model_params_dir, "{:0>6}.pkl".format(frame_id)))
            for param_name in body_model_params.keys():
                body_model_params[param_name].append(torch.from_numpy(params[param_name]).float().unsqueeze(0))
        for param_name in body_model_params.keys():
            body_model_params[param_name] = torch.cat(body_model_params[param_name], dim=0)
            self.body_model_params.init_parameters(param_name, body_model_params[param_name], requires_grad=False) 

    @torch.no_grad()
    def decode_batch(self, batch):
        frame_id = batch['frame_id']
        cam_id = batch['cam_id']
        frame_idx = batch['frame_idx']
        rays = batch['rays'] # (bs, n_rays, 8)
        rgbs = batch['rgbs'] # (bs, n_rays, 3)
        alphas = batch['alphas'] # (bs, n_rays, 1)
        body_model_params = {
            'betas': batch['betas'],
            'global_orient': batch['global_orient'],
            'body_pose': batch['body_pose'],
            'transl': batch['transl']
        }
        body_model_params_template = {
            'betas': batch['betas_template'],
            'global_orient': batch['global_orient_template'],
            'body_pose': batch['body_pose_template'],
            'transl': batch['transl_template']
        }
        fg_points = batch['fg_points'] # (bs, num_points, 3)
        bg_points = batch['bg_points'] # (bs, num_points, 3)
        
        return frame_id, cam_id, frame_idx, rays, rgbs, alphas, body_model_params, body_model_params_template, fg_points, bg_points

    def forward(self, rays, body_model_params, body_model_params_template, latent_code=None, perturb=1.0):
        bs, n_rays = rays.shape[:2]
        results = defaultdict(list)
        chunk = self.hparams.chunk

        self.anim_nerf.set_body_model(body_model_params, body_model_params_template)
        rays = self.anim_nerf.convert_to_body_model_space(rays)
        self.anim_nerf.clac_ober2cano_transform()

        if latent_code is not None:
            self.anim_nerf.set_latent_code(latent_code)

        for i in range(0, n_rays, chunk):
            rays_chunk = rays[:, i:i+chunk, :]
            rendered_ray_chunks = self.volume_renderer(self.anim_nerf, rays_chunk, perturb=perturb)
            
            for k, v in rendered_ray_chunks.items():
                results[k] += [v]
                
        for k, v in results.items():
            results[k] = torch.cat(v, 1)

        return results

    def configure_optimizers(self):
        parameters = [ {'params': self.anim_nerf.parameters(), 'lr': self.hparams.train.lr}]
        if self.hparams.deformation_dim > 0 or self.hparams.apperance_dim > 0:
            parameters.append({'params': self.latent_codes.parameters(), 'lr': self.hparams.train.lr})
        if self.hparams.optim_body_params:
            parameters.append({'params': self.body_model_params.parameters(), 'lr': self.hparams.train.lr*0.5})
        self.optimizer = get_optimizer(self.hparams.train, parameters)
        self.scheduler = get_scheduler(self.hparams.train, self.optimizer)
        
        return [self.optimizer], [self.scheduler]
    
    def compute_loss(self, rgbs, alphas, results, frame_idx=None, latent_code=None, fg_points=None, bg_points=None):
        loss = 0
        loss_details = {}

        # rgb
        loss_rgb = F.mse_loss(results['rgbs'], rgbs, reduction='mean')
        loss += loss_rgb
        loss_details['loss_rgb'] = loss_rgb
        
        if self.hparams.n_importance > 0 and not self.hparams.share_fine:
            loss_rgb_fine = F.mse_loss(results['rgbs_fine'], rgbs, reduction='mean')
            loss += loss_rgb_fine
            loss_details['loss_rgb_fine'] = loss_rgb_fine

        # alphas
        loss_alphas = F.l1_loss(results['alphas'], alphas)
        loss += self.hparams.train.lambda_alphas * loss_alphas
        loss_details['loss_alphas'] = loss_alphas

        if self.hparams.n_importance > 0 and not self.hparams.share_fine:
            loss_alphas_fine = F.l1_loss(results['alphas_fine'], alphas)
            loss += self.hparams.train.lambda_alphas * loss_alphas_fine
            loss_details['loss_alphas_fine'] = loss_alphas_fine


        # if latent_code is not None:
        #     loss_latent = torch.mean(torch.pow(latent_code, 2))
        #     loss += self.hparams.lambda_latent * loss_latent
        #     loss_details['loss_latent'] = loss_latent
            
        #     frame_idx_ = torch.clamp(frame_idx+1, 0, self.hparams.num_frames)
        #     latent_code_ = self.latent_codes(frame_idx_)
        #     loss_latent_smooth = F.mse_loss(latent_code, latent_code_)
        #     loss += self.hparams.lambda_latent_smooth * loss_latent_smooth
        #     loss_details['loss_latent_smooth'] = loss_latent_smooth
        
        if self.hparams.use_unpose and fg_points is not None:
            fg_points_sigma = self.anim_nerf.query_canonical_space(fg_points, use_fine=False, only_sigma=True)
            loss_foreground = torch.mean(torch.exp(-2.0/self.hparams.n_samples * torch.relu(fg_points_sigma)))
            loss += self.hparams.train.lambda_foreground * loss_foreground
            loss_details['loss_foreground'] = loss_foreground

            if self.hparams.n_importance > 0 and not self.hparams.share_fine:
                fg_points_sigma_fine = self.anim_nerf.query_canonical_space(fg_points, use_fine=True, only_sigma=True)
                loss_foreground_fine = torch.mean(torch.exp(-2.0/self.hparams.n_samples * torch.relu(fg_points_sigma_fine)))
                loss += self.hparams.train.lambda_foreground * loss_foreground_fine
                loss_details['loss_foreground_fine'] = loss_foreground_fine
        
        if self.hparams.use_unpose and bg_points is not None:
            bg_points_sigma = self.anim_nerf.query_canonical_space(bg_points, use_fine=False, only_sigma=True)
            loss_background = torch.mean(1 - torch.exp(-2.0/self.hparams.n_samples * torch.relu(bg_points_sigma)))
            loss += self.hparams.train.lambda_background * loss_background
            loss_details['loss_background'] = loss_background

            if self.hparams.n_importance > 0 and not self.hparams.share_fine:
                bg_points_sigma_fine = self.anim_nerf.query_canonical_space(bg_points, use_fine=True, only_sigma=True)
                loss_background_fine = torch.mean(1 - torch.exp(-2.0/self.hparams.n_samples * torch.relu(bg_points_sigma_fine)))
                loss += self.hparams.train.lambda_background * loss_background_fine
                loss_details['loss_background_fine'] = loss_background_fine

        # normal
        points = self.anim_nerf.verts_template.detach()
        points += torch.randn_like(points) * self.hparams.dis_threshold * 0.5
        points_neighbs = points + torch.randn_like(points) * self.hparams.train.epsilon
        points_normal = self.anim_nerf.query_canonical_space(points, use_fine=False, only_normal=True)
        points_neighbs_normal = self.anim_nerf.query_canonical_space(points_neighbs, use_fine=False, only_normal=True)
        points_normal = points_normal / (torch.norm(points_normal, p=2, dim=-1, keepdim=True) + 1e-5)
        points_neighbs_normal = points_neighbs_normal / (torch.norm(points_neighbs_normal, p=2, dim=-1, keepdim=True) + 1e-5)
        loss_normals = F.mse_loss(points_normal, points_neighbs_normal)
        # loss_normals = torch.mean((torch.norm(points_normal, p=2, dim=-1) - 1)**2)
        loss += self.hparams.train.lambda_normals * loss_normals
        loss_details['loss_normals'] = loss_normals

        if self.hparams.n_importance > 0 and not self.hparams.share_fine:
            points_normal_fine = self.anim_nerf.query_canonical_space(points, use_fine=True, only_normal=True)
            points_neighbs_normal_fine = self.anim_nerf.query_canonical_space(points_neighbs, use_fine=True, only_normal=True)
            points_normal_fine = points_normal_fine / (torch.norm(points_normal_fine, p=2, dim=-1, keepdim=True) + 1e-5)
            points_neighbs_normal_fine = points_neighbs_normal_fine / (torch.norm(points_neighbs_normal_fine, p=2, dim=-1, keepdim=True) + 1e-5)
            loss_normals_fine = F.mse_loss(points_normal_fine, points_neighbs_normal_fine)
            # loss_normals_fine = torch.mean((torch.norm(points_normal_fine, p=2, dim=-1) - 1)**2)
            loss += self.hparams.train.lambda_normals * loss_normals_fine
            loss_details['loss_normals_fine'] = loss_normals_fine

        # if body_model_params is not None:
        #     loss_pose = F.mse_loss(results['joints'].clone(), self.anim_nerf.model(**body_model_params)['joints'].clone())
        #     loss += self.hparams.lambda_pose * loss_pose
        #     loss_details['loss_pose'] = loss_pose

        #     frame_id_ = torch.clamp(frame_id+1, 0, self.body_model_params.num_frame-1)
        #     body_model_params_ref_ = self.body_model_params(frame_id_)
        #     loss_pose_smooth = F.mse_loss(self.anim_nerf.joints, self.anim_nerf.model(**body_model_params_ref_)['joints'])
        #     loss += self.hparams.lambda_pose_smooth * loss_pose_smooth
        #     loss_details['loss_pose_smooth'] = loss_pose_smooth

        return loss, loss_details

    def training_step(self, batch, batch_nb):
        frame_id, cam_id, frame_idx, rays, rgbs, alphas, body_model_params, body_model_params_template, fg_points, bg_points = self.decode_batch(batch)
        if self.hparams.latent_dim > 0:
            latent_code = self.latent_codes(frame_idx)
        else:
            latent_code = None
        if self.hparams.optim_body_params:
            body_model_params = self.body_model_params(frame_idx)
        results = self(rays, body_model_params, body_model_params_template, latent_code=latent_code)
        loss, loss_details = self.compute_loss(rgbs, alphas, results, frame_idx=frame_idx, fg_points=fg_points, bg_points=bg_points)
        self.log('train/loss', loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        for loss_name in loss_details.keys():
            self.log('train/{}'.format(loss_name), loss_details[loss_name], on_step=True, on_epoch=False, prog_bar=True, logger=True)
        
        with torch.no_grad():
            if 'rgbs_fine' in results:
                train_psnr = psnr(results['rgbs_fine'], rgbs)
            else:
                train_psnr = psnr(results['rgbs'], rgbs)
            self.log('train/psnr',  train_psnr, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        
        lr = get_learning_rate(self.optimizer)
        self.log('lr', lr, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        return loss
        

    def validation_step(self, batch, batch_nb):
        frame_id, cam_id, frame_idx, rays, rgbs, alphas, body_model_params, body_model_params_template, fg_points, bg_points = self.decode_batch(batch)
        if self.hparams.latent_dim > 0:
            if frame_idx != -1:
                latent_code = self.latent_codes(frame_idx)
            else:
                latent_code = self.latent_codes(torch.zeros_like(frame_idx))
        else:
            latent_code = None
        if self.hparams.optim_body_params and frame_idx != -1:
            body_model_params = self.body_model_params(frame_idx)
        # else:
        #     body_model_params['betas'] = self.body_model_params.betas(torch.zeros_like(frame_idx))
        results = self(rays, body_model_params, body_model_params_template, latent_code=latent_code)
        loss, _ = self.compute_loss(rgbs, alphas, results)
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        if 'rgbs_fine' in results:
            val_psnr = psnr(results['rgbs_fine'], rgbs)
        else:
            val_psnr = psnr(results['rgbs'], rgbs)
        self.log('val/psnr', val_psnr, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        W, H = self.hparams.img_wh
        
        def visualize(frame_id, cam_id, rgbs_gt, rgbs, depths, W, H):
            img = rgbs.cpu().view(H, W, 3).permute(2, 0, 1) # (3, H, W)
            img_gt = rgbs_gt.cpu().view(H, W, 3).permute(2, 0, 1) # (3, H, W)
            depth = visualize_depth(depths.cpu().view(H, W))
            stack = torch.stack([img_gt, img, depth]) # (3, 3, H, W)
            self.logger.experiment.add_images('val/GT_pred_depth_cam{:0>3d}_{:0>6d}'.format(cam_id, frame_id), stack, self.global_step)

        if batch_nb % self.hparams.val.vis_freq == 0:
            if 'rgbs_fine' in results:
                visualize(frame_id.item(), cam_id.item(), rgbs, results['rgbs_fine'], results['depths_fine'], W, H)
            else:
                visualize(frame_id.item(), cam_id.item(), rgbs, results['rgbs'], results['depths'], W, H)
        
        return loss
    
    def test_step(self, batch, batch_nb):
        frame_id, cam_id, frame_idx, rays, rgbs, alphas, body_model_params, body_model_params_template, fg_points, bg_points = self.decode_batch(batch)
        if self.hparams.latent_dim > 0:
            if frame_idx != -1:
                latent_code = self.latent_codes(frame_idx)
            else:
                latent_code = self.latent_codes(torch.zeros_like(frame_idx))
        else:
            latent_code = None
        if self.hparams.optim_body_params and frame_idx != -1:
            body_model_params = self.body_model_params(frame_idx)
        # else:
        #     body_model_params['betas'] = self.body_model_params.betas(torch.zeros_like(frame_idx))
        results = self(rays, body_model_params, body_model_params_template, latent_code=latent_code, perturb=0.0)
        loss, _ = self.compute_loss(rgbs, alphas, results)
        self.log('test/loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=False)

        if 'rgbs_fine' in results:
            test_psnr = psnr(results['rgbs_fine'], rgbs)
        else:
            test_psnr = psnr(results['rgbs'], rgbs)
        self.log('test/psnr', test_psnr, on_step=False, on_epoch=True, prog_bar=True, logger=False)
        
        W, H = self.hparams.img_wh
        
        def visualize(frame_id, cam_id, rgbs_gt, rgbs, depths, W, H):
            img = rgbs.cpu().view(H, W, 3).permute(2, 0, 1) # (3, H, W)
            img_gt = rgbs_gt.cpu().view(H, W, 3).permute(2, 0, 1) # (3, H, W)
            depth = visualize_depth(depths.cpu().view(H, W))
            stack = torch.stack([img_gt, img, depth]) # (3, 3, H, W)
            os.makedirs(os.path.join(self.hparams.outputs_dir, self.hparams.exp_name, 'cam{:0>3d}'.format(cam_id)), exist_ok=True)
            save_image(stack, '{}/{}/cam{:0>3d}/{:0>6d}.png'.format(self.hparams.outputs_dir, self.hparams.exp_name, cam_id, frame_id))
            #self.logger.experiment.add_images('test/GT_pred_depth_{}'.format(nb), stack, self.global_step)

        if batch_nb % self.hparams.test.vis_freq == 0:
            if 'rgbs_fine' in results:
                visualize(frame_id.item(), cam_id.item(), rgbs, results['rgbs_fine'], results['depths_fine'], W, H)
            else:
                visualize(frame_id.item(), cam_id.item(), rgbs, results['rgbs'], results['depths'], W, H)
        
        return loss

if __name__ == '__main__':
    # torch.autograd.set_detect_anomaly(True)
    cfg = get_cfg()
    data = AnimNeRFData(cfg)
    system = AnimNeRFSystem(cfg)
    print(system)
    if cfg.train.ckpt_path is not None:
        if cfg.train.model_names_to_load is None:
            system.load_from_checkpoint(cfg.train.ckpt_path)
        else:
            for model_name in cfg.train.model_names_to_load:
                load_ckpt(getattr(system, model_name), cfg.train.ckpt_path, model_name)
                for param in getattr(system, model_name).parameters():
                    param.requires_grad = cfg.train.pretrained_model_requires_grad
        
    checkpoint_callback = ModelCheckpoint(dirpath=f'{cfg.checkpoints_dir}/{cfg.exp_name}',
                                          filename='{epoch:d}',
                                          monitor='train/psnr',
                                          mode='max',
                                          save_top_k=cfg.train.save_top_k,
                                          save_last=cfg.train.save_last)

    logger = TestTubeLogger(
        save_dir=cfg.logs_dir,
        name=cfg.exp_name,
        debug=False,
        create_git_tag=False
    )

    trainer = Trainer(max_epochs=cfg.train.max_epochs,
                      callbacks=[checkpoint_callback],
                      logger=logger,
                      progress_bar_refresh_rate=1,
                      gpus=cfg.num_gpus,
                      accelerator='dp',
                      num_sanity_val_steps=1,
                      benchmark=True,
                      profiler="simple")

    trainer.fit(system, data)
    trainer.test(datamodule=data)