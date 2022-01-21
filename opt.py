import argparse

def get_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument('--root_dir', type=str,
                        default='data/BR_BM_04_BR0',
                        help='root directory of dataset')
    parser.add_argument('--dataset_name', type=str, default='aist_plus_plus',
                        choices=['anim_nerf', 'zju_mocap', 'human36m', 'iper', 'people_snapshot', 'aist_plus_plus', 'mutil_garment_synthetize'],
                        help='which dataset to train/val')
    parser.add_argument('--checkpoints_dir', type=str,
                        default='checkpoints',
                        help='root directory of checkpoints')
    parser.add_argument('--logs_dir', type=str,
                        default='logs',
                        help='root directory of logs')
    parser.add_argument('--outputs_dir', type=str,
                        default='outputs',
                        help='root directory of output')
    parser.add_argument('--template_path', type=str,
                        default=None,
                        help='template path')
    parser.add_argument('--img_wh', nargs="+", type=int, default=[512, 512],
                        help='resolution (img_w, img_h) of the image')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'val', 'test'],
                        help='mode to choice')
    parser.add_argument('--white_bkgd', default=False, action='store_true',
                        help='if use white background')
    parser.add_argument('--frame_start_ID', type=int, default=1, 
                        help='frame start ID')
    parser.add_argument('--frame_end_ID', type=int, default=400, 
                        help='frame end ID')
    parser.add_argument('--frame_skip', type=int, default=4, 
                        help='frame skip')
    parser.add_argument('--cam_IDs', nargs="+", type=int, default=None,
                        help='cam ID list')
    parser.add_argument('--subsampletype', type=str, default='foreground_pixel', 
                        help='subsampletype')
    parser.add_argument('--subsamplesize', type=int, default=32, 
                        help='subsamplesize')
    parser.add_argument('--lambda_alphas', type=float, default=0.1, 
                        help='lambda of alphas')
    parser.add_argument('--latent_dim', type=int, default=0, 
                        help='latent code dim per frame')
    parser.add_argument('--deformation_dim', type=int, default=0, 
                        help='deformation codes dim')
    parser.add_argument('--apperance_dim', type=int, default=0, 
                        help='apperance codes dim')
    parser.add_argument('--lambda_latent', type=float, default=0.001, 
                        help='latent code weight for l2 regularization')
    parser.add_argument('--lambda_latent_smooth', type=float, default=0.001, 
                        help='latent code smooth weight for l2 regularization')
    parser.add_argument('--lambda_foreground', type=float, default=0.001, 
                        help='foreground regularization')
    parser.add_argument('--lambda_background', type=float, default=0.01, 
                        help='background regularization')
    parser.add_argument('--freqs_xyz', type=int, default=10, 
                        help='the frequency of xyz')
    parser.add_argument('--freqs_dir', type=int, default=4, 
                        help='the frequency of xyz')
    parser.add_argument('--use_view', default=False, action='store_true',
                        help='if use viewdir')
    parser.add_argument('--k_neigh', type=int, default=4, 
                        help='the number of neighbors')
    parser.add_argument('--use_knn', default=False, action='store_true',
                        help='if use knn')
    parser.add_argument('--use_deformation', default=False, action='store_true',
                        help='if use deformation')
    parser.add_argument('--use_unpose', default=False, action='store_true',
                        help='if use unpose')
    parser.add_argument('--unpose_view', default=False, action='store_true',
                        help='if unpose viewdir')
    parser.add_argument('--use_rays_unpose', default=False, action='store_true',
                        help='if use rays unpose')
    parser.add_argument('--use_weight_std', default=False, action='store_true',
                        help='if use weight std')
    parser.add_argument('--optim_pose', default=False, action='store_true',
                        help='if optimize smpl pose')
    parser.add_argument('--optim_shape', default=False, action='store_true',
                        help='if optimize smpl shape')
    parser.add_argument('--optim_body_params', default=False, action='store_true',
                        help='if optimize smpl params')
    parser.add_argument('--lambda_pose', type=float, default=0.001, 
                        help='lambda of pose code')
    parser.add_argument('--lambda_pose_smooth', type=float, default=0.01,
                        help='lambda of pose smooth')
    parser.add_argument('--lambda_deformation', type=float, default=10., 
                        help='lambda of deformation field')
    parser.add_argument('--lambda_cycle', type=float, default=0.1, 
                        help='lambda of cycle deformation')
    parser.add_argument('--dis_threshold', type=float, default=0.2, 
                        help='distance threshold')
    parser.add_argument('--pose_dim', type=int, default=69, 
                        help='smpl pose dim')
    parser.add_argument('--clamp_rays', default=False, action='store_true',
                        help='if clamp the ray to surfaces')
    parser.add_argument('--n_samples', type=int, default=64,
                        help='number of coarse samples')
    parser.add_argument('--n_importance', type=int, default=16,
                        help='number of fine samples')
    parser.add_argument('--loss_type', type=str, default='mse',
                        choices=['mse'],
                        help='loss betas to use')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='batch size')
    parser.add_argument('--chunk', type=int, default=2048,
                        help='chunk size to split the input to avoid OOM')
    parser.add_argument('--max_epochs', type=int, default=30,
                        help='number of training epochs')
    parser.add_argument('--max_steps', type=int, default=200000,
                        help='number of training steps')
    parser.add_argument('--num_gpus', type=int, default=-1,
                        help='number of gpus')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='number of cpu workers')
                        
    parser.add_argument('--model_path', type=str, default='smplx/models',
                        help='smpl or smplx models folder')
    parser.add_argument('--gender', type=str, default='neutral',
                        help='the gender of subject',
                        choices=['neutral', 'male', 'female'])
    parser.add_argument('--model_type', type=str, default='smpl',
                        help='smpl or smplx',
                        choices=['smpl', 'smplh', 'smplx'])

    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='pretrained checkpoint path to load')
    parser.add_argument('--prefixes_to_ignore', nargs='+', type=str, default=['loss'],
                        help='the prefixes to ignore in the checkpoint state dict')

    parser.add_argument('--optimizer', type=str, default='adam',
                        help='optimizer type',
                        choices=['sgd', 'adam', 'radam', 'ranger'])
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='learning rate momentum')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay')
    parser.add_argument('--lr_scheduler', type=str, default='poly',
                        help='scheduler type',
                        choices=['steplr', 'cosine', 'poly'])
    ###########################
    #### params for steplr ####
    parser.add_argument('--decay_step', nargs='+', type=int, default=[20],
                        help='scheduler decay step')
    parser.add_argument('--decay_gamma', type=float, default=0.1,
                        help='learning rate decay amount')
    ###########################
    #### params for poly ####
    parser.add_argument('--poly_exp', type=float, default=0.9,
                        help='exponent for polynomial learning rate decay')
    ###########################

    parser.add_argument('--exp_name', type=str, default='exp',
                        help='experiment name')

    return parser.parse_args()
