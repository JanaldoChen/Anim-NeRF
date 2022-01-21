import os
import numpy as np 
import torch
import trimesh
import argparse
import cv2
from tqdm import tqdm
from utils.util import load_pickle_file, write_pickle_file
from smplx import body_models, create

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_root', type=str, default='data/people_snapshot',
                        help='data root')
    parser.add_argument('--people_ID', type=str, default='male-3-casual',
                        help='people id')
    parser.add_argument('--gender', type=str, default='male',
                        help='gender')
    parser.add_argument('--model_path', type=str, default='smplx/models',
                        help='smpl model path')
    parser.add_argument('--model_type', type=str, default='smpl',
                        help='model type')
    parser.add_argument('--template_path', type=str, default='assets/X_pose.pkl',
                        help='template pose path')
    parser.add_argument('--chunk', type=int, default=1024,
                        help='chunk size')
    parser.add_argument('--num_points', type=int, default=64*64*64,
                        help='number of points')
    parser.add_argument('--vis', action='store_true', default=False,
                        help='visualization')

    return parser.parse_args()


if __name__ == "__main__":
    args = get_opts()
    data_root = args.data_root
    model_path = args.model_path
    model_type = args.model_type
    people_ID = args.people_ID
    gender = args.gender
    chunk = args.chunk
    num_points = args.num_points

    body_model = create(model_path=model_path, model_type=model_type, gender=gender).to(device)

    params_dir = os.path.join(data_root, people_ID, '{}s'.format(model_type))
    frame_IDs = [frame_id[:-4] for frame_id in os.listdir(params_dir)]
    frame_IDs = sorted(frame_IDs)

    betas = []
    for frame_id in frame_IDs:
        params = load_pickle_file(os.path.join(params_dir, "{:0>6}.pkl".format(frame_id)))
        betas.append(params['betas'])
    betas = np.stack(betas).mean(0)

    params_template = load_pickle_file(args.template_path)
    pose_dim = 69 if model_type == 'smpl' else 63
    body_params_template = {
        'betas': torch.from_numpy(betas).float().unsqueeze(0).to(device),
        'global_orient': torch.from_numpy(params_template['global_orient']).float().unsqueeze(0).to(device),
        'body_pose': torch.from_numpy(params_template['body_pose'][:pose_dim]).float().unsqueeze(0).to(device),
        'transl': torch.from_numpy(params_template['transl']).float().unsqueeze(0).to(device)
    }

    verts = body_model(**body_params_template)['vertices'].squeeze(0).detach().cpu().numpy()
    body_mesh = trimesh.Trimesh(vertices=verts, faces=body_model.faces, process=False)
    #smpl_mesh.show()

    orig_bbox = np.stack([verts.min(0), verts.max(0)])
    center =orig_bbox.mean(0)
    scale = [2.0, 2.0, 5.0]
    dxyz = orig_bbox[1] - orig_bbox[0]
    bbox = np.stack([center - dxyz*scale/2, center+dxyz*scale/2])

    points = np.random.rand(num_points, 3)
    points = points * (bbox[1, :] - bbox[0, :]) + bbox[0, :]
    nv = points.shape[0]
    distances = []
    for i in tqdm(range(0, nv, chunk)):
        points_chunk = points[i:i+chunk, :]
        distances_chunk = body_mesh.nearest.signed_distance(points_chunk)
        distances.append(distances_chunk)
    distances = np.concatenate(distances, axis=0)
    distances *= -1

    template_params = {
        'betas': betas,
        'body_pose': params_template['body_pose'][:pose_dim],
        'global_orient': params_template['global_orient'],
        'transl': params_template['transl'],
        'model_type': model_type,
        'gender': gender,
        'verts': verts,
        'faces': body_model.faces,
        'center': center,
        'bbox': bbox,
        'points': points,
        'distances': distances
    }
    write_pickle_file(os.path.join(data_root, people_ID, '{}_template.pkl'.format(model_type)), template_params)

    if args.vis:
        points_outside = points[distances>0]
        points_inside = points[distances<0]

        Scene = trimesh.scene.scene.Scene()
        cloud_outside = trimesh.PointCloud(vertices=points_outside, colors=[255, 0, 0])
        cloud_inside = trimesh.PointCloud(vertices=points_inside, colors=[0, 255, 0])
        Scene.add_geometry(body_mesh)
        Scene.add_geometry(cloud_outside)
        Scene.add_geometry(cloud_inside)
        Scene.show()

