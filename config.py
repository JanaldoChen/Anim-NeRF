import os
import argparse
from numpy import False_
import yacs
from yacs.config import CfgNode as CN

def get_default_config():
    cfg = CN()

    cfg.num_gpus = -1
    cfg.exp_name = 'male-3-casual'
    cfg.dataset_name = 'anim_nerf'
    cfg.root_dir = './data/male-3-casual'
    cfg.model_type = 'smpl'
    cfg.gender = 'male'
    cfg.model_path = './smplx/models'
    cfg.checkpoints_dir = './checkpoints'
    cfg.logs_dir = './logs'
    cfg.outputs_dir = './outputs'

    cfg.img_wh = (512, 512)
    cfg.freqs_xyz = 10
    cfg.freqs_dir = 4
    cfg.use_view = False
    cfg.use_knn = True
    cfg.k_neigh = 4
    cfg.use_unpose = True
    cfg.unpose_view = False
    cfg.use_deformation = False
    cfg.deformation_dim = 0
    cfg.apperance_dim = 0
    cfg.latent_dim = 0
    cfg.pose_dim = 69
    cfg.optim_body_params = True

    cfg.dis_threshold = 0.2
    cfg.n_samples = 64
    cfg.n_importance = 16
    cfg.n_depth = 0
    cfg.share_fine = False
    cfg.chunk = 2048
    cfg.query_inside = False

    cfg.white_bkgd = True


    # ============== Train ===============
    cfg.train = CN()
    cfg.train.frame_start_ID = 1
    cfg.train.frame_end_ID = 400
    cfg.train.frame_skip = 4
    cfg.train.cam_IDs = None
    cfg.train.subsampletype = 'foreground_pixel'
    cfg.train.subsamplesize = 32
    cfg.train.fore_rate = 0.9
    cfg.train.fore_erode = 3
    cfg.train.lambda_alphas = 0.1
    cfg.train.lambda_foreground = 0.01
    cfg.train.lambda_background = 0.01
    cfg.train.lambda_normals = 0.01
    cfg.train.lambda_cycle = 0.1
    cfg.train.epsilon = 0.01
    cfg.train.batch_size = 16
    cfg.train.max_epochs = 30
    cfg.train.max_steps = 200000
    cfg.train.lr = 5e-4
    cfg.train.optimizer = CN({'type': 'adam', 'momentum': 0.9, 'weight_decay': 0})
    cfg.train.scheduler = CN({'type': 'poly', 'poly_exp': 0.9})
    # cfg.train.lr_scheduler = CN({'type': 'step', 'decay_step': [20], 'decay_gamma': 0.1})
    cfg.train.num_workers = 8
    cfg.train.save_top_k = 1
    cfg.train.save_last = True
    cfg.train.resume = False
    cfg.train.ckpt_path = None
    cfg.train.model_names_to_load = None
    cfg.train.pretrained_model_requires_grad = False
    cfg.train.strategy = 'dp'



    # ============== Val ===============
    cfg.val = CN()
    cfg.val.frame_start_ID = 400
    cfg.val.frame_end_ID = 500
    cfg.val.frame_skip = 4
    cfg.val.cam_IDs = None
    cfg.val.batch_size = 1
    cfg.val.num_workers = 8
    cfg.val.vis_freq = 20

    # ============== Test ===============
    cfg.test = CN()
    cfg.test.frame_start_ID = 400
    cfg.test.frame_end_ID = 500
    cfg.test.frame_skip = 4
    cfg.test.cam_IDs = None
    cfg.test.batch_size = 1
    cfg.test.num_workers = 8
    cfg.test.vis_freq = 4

    return cfg

def get_cfg():

    cfg = get_default_config()

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_file", default="configs/default.yaml", type=str)
    parser.add_argument("--type", type=str, default="train")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg.merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)
    cfg.frame_IDs = list(range(cfg.train.frame_start_ID, cfg.train.frame_end_ID+1, cfg.train.frame_skip))
    cfg.num_frames = len(cfg.frame_IDs)

    return cfg


