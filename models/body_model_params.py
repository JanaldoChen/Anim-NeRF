import torch
import torch.nn as nn
import torch.nn.functional as F

class BodyModelParams(nn.Module):
    def __init__(self, num_frames, model_type='smpl'):
        super(BodyModelParams, self).__init__()
        self.num_frames = num_frames
        self.model_type = model_type
        self.params_dim = {
            'betas': 10,
            'global_orient': 3,
            'transl': 3,
        }
        if model_type == 'smpl':
            self.params_dim.update({
                'body_pose': 69,
            })
        elif model_type == 'smplh':
            self.params_dim.update({
                'body_pose': 63,
                'left_hand_pose': 6,
                'right_hand_pose': 6,
            })
        elif model_type == 'smplx':
            self.params_dim.update({
                'body_pose': 63,
                'left_hand_pose': 6,
                'right_hand_pose': 6,
                'jaw_pose': 3,
                # 'leye_pose': 3,
                # 'reye_pose': 3,
                'expression': 10,
            })
        else:
            assert ValueError(f'Unknown model type {model_type}, exiting!')
        
        self.param_names = self.params_dim.keys()
        
        for param_name in self.param_names:
            if param_name == 'betas':
                param = nn.Embedding(1, self.params_dim[param_name])
                param.weight.data.fill_(0)
                param.weight.requires_grad = False
                setattr(self, param_name, param)
            else:
                param = nn.Embedding(num_frames, self.params_dim[param_name])
                param.weight.data.fill_(0)
                param.weight.requires_grad = False
                setattr(self, param_name, param)
    
    def init_parameters(self, param_name, data, requires_grad=False):
        if param_name == 'betas':
            data = torch.mean(data, dim=0, keepdim=True)
        getattr(self, param_name).weight.data = data[..., :self.params_dim[param_name]]
        getattr(self, param_name).weight.requires_grad = requires_grad

    def set_requires_grad(self, param_name, requires_grad=True):
        getattr(self, param_name).weight.requires_grad = requires_grad

    def forward(self, frame_ids):
        params = {}
        for param_name in self.param_names:
            if param_name == 'betas':
                params[param_name] = getattr(self, param_name)(torch.zeros_like(frame_ids))
            else:
                params[param_name] = getattr(self, param_name)(frame_ids)
        return params
