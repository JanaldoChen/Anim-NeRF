import torch
import torch.nn as nn
import torch.nn.functional as F

from smplx import create

from models.nerf import NeRF, DeRF

def compute_rotation_matrix_from_ortho6d(ortho6d):
    x_raw = ortho6d[..., 0:3]
    y_raw = ortho6d[..., 3:6]

    x = F.normalize(x_raw, dim=-1)
    z = torch.cross(x, y_raw, dim=-1)
    z = F.normalize(z, dim=-1)
    y = torch.cross(z, x, dim=-1)
    
    x = x.unsqueeze(-1)
    y = y.unsqueeze(-1)
    z = z.unsqueeze(-1)
    matrix = torch.cat((x,y,z), dim=-1)
    return matrix

def batch_index_select(data, inds):
    bs, nv = data.shape[:2]
    device = data.device
    inds = inds + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None]
    data = data.reshape(bs*nv, *data.shape[2:])
    return data[inds.long()]

def batch_transform(P, v, pad_ones=True):
    if pad_ones:
        homo = torch.ones((*v.shape[:-1], 1), dtype=v.dtype, device=v.device)
    else:
        homo = torch.zeros((*v.shape[:-1], 1), dtype=v.dtype, device=v.device)
    v_homo = torch.cat((v, homo), dim=-1)
    v_homo = torch.matmul(P, v_homo.unsqueeze(-1))
    v_ = v_homo[..., :3, 0]
    return v_

class AnimNeRF(nn.Module):
    def __init__(self,
            model_path='smplx/models',
            model_type='smpl',
            gender='male',
            freqs_xyz=10,
            freqs_dir=4,
            use_view=False,
            use_unpose=False,
            unpose_view=False,
            k_neigh=4,
            use_knn=False,
            use_deformation=False,
            deformation_dim=0,
            apperance_dim=0,
            use_fine=False,
            share_fine=False,
            dis_threshold=0.2,
            query_inside=False,
            **kwargs
        ):
        super(AnimNeRF, self).__init__()

        self.freqs_xyz = freqs_xyz
        self.freqs_dir = freqs_dir
        self.use_view = use_view
        self.use_unpose = use_unpose
        self.unpose_view = unpose_view
        self.k_neigh = k_neigh
        self.use_knn = use_knn
        self.use_deformation = use_deformation
        self.deformation_dim = deformation_dim
        self.apperance_dim = apperance_dim
        self.use_fine = use_fine
        self.share_fine=share_fine
        self.dis_threshold = dis_threshold
        self.query_inside = query_inside

        self.body_model = create(model_path, model_type, gender=gender)

        if use_knn:
            from knn_cuda import KNN
            self.knn = KNN(k=k_neigh, transpose_mode=True)
        self.weight_std = 0.1

        self.lbs_dim = self.body_model.lbs_weights.shape[1]

        if use_deformation:
            self.derf = DeRF(freqs_xyz=freqs_xyz, out_channels=9)

        self.nerf = NeRF(freqs_xyz=freqs_xyz, freqs_dir=freqs_dir, use_view=use_view, deformation_dim=deformation_dim, apperance_dim=apperance_dim)

        if use_fine:
            if share_fine:
                self.nerf_fine = self.nerf
            else:
                self.nerf_fine = NeRF(freqs_xyz=freqs_xyz, freqs_dir=freqs_dir, use_view=use_view, deformation_dim=deformation_dim, apperance_dim=apperance_dim)

    def set_latent_code(self, latent_code):
        if self.deformation_dim > 0:
            self.deformation_code = latent_code[:, :self.deformation_dim]
            if self.apperance_dim > 0:
                self.apperance_code = latent_code[:, self.deformation_dim:self.deformation_dim+self.apperance_dim]
        else:
            if self.apperance_dim > 0:
                self.apperance_code = latent_code[:, :self.apperance_dim]

    def set_body_model(self, body_model_params, body_model_params_template=None):
        body_model_out = self.body_model(**body_model_params, return_verts=True)
        self.verts = body_model_out['vertices']
        self.joints = body_model_out['joints'][:, :self.lbs_dim]
        self.verts_transform = body_model_out['vertices_transform']
        self.joints_transform = body_model_out['joints_transform']
        self.shape_offsets = body_model_out['shape_offsets']
        self.pose_offsets = body_model_out['pose_offsets']
            
        self.global_transform = body_model_out['joints_transform'][:, 0, :, :].clone()

        if body_model_params_template is not None:
            smpl_out_template = self.body_model(**body_model_params_template, return_verts=True)
            self.verts_template = smpl_out_template['vertices']
            self.joints_template = smpl_out_template['joints'][:, :self.lbs_dim]
            self.verts_transform_template = smpl_out_template['vertices_transform']
            self.joints_transform_template = smpl_out_template['joints_transform']
            self.shape_offsets_template = smpl_out_template['shape_offsets']
            self.pose_offsets_template = smpl_out_template['pose_offsets']

    def convert_to_body_model_space(self, rays):
        bs, n_rays = rays.shape[:2]

        global_transform_inv = torch.inverse(self.global_transform).unsqueeze(1) # (bs, 1, 4, 4)
        rays_o = batch_transform(global_transform_inv.expand(-1, n_rays, -1, -1), rays[:, :, 0:3], pad_ones=True)
        rays_d = batch_transform(global_transform_inv.expand(-1, n_rays, -1, -1), rays[:, :, 3:6], pad_ones=False)
        cam_dist = torch.norm(rays_o, dim=-1, keepdim=True)
        near = torch.max(rays[:, :, 6:7], cam_dist - 1.0)
        far = torch.min(rays[:, :, 7:8], cam_dist + 1.0)
        new_rays = torch.cat((rays_o, rays_d, near, far), dim=-1)
        
        self.verts = batch_transform(global_transform_inv.expand(-1, self.verts.shape[1], -1, -1), self.verts, pad_ones=True)
        self.joints = batch_transform(global_transform_inv.expand(-1, self.joints.shape[1], -1, -1), self.joints, pad_ones=True)
        self.global_transform = torch.matmul(global_transform_inv.squeeze(1), self.global_transform)
        self.verts_transform = torch.matmul(global_transform_inv.expand(-1, self.verts.shape[1], -1, -1), self.verts_transform)
        self.joints_transfrom = torch.matmul(global_transform_inv.expand(-1, self.joints.shape[1], -1, -1), self.joints_transform)

        return new_rays
    
    def clac_ober2cano_transform(self):
        ober2cano_transform = torch.inverse(self.verts_transform).clone()
        ober2cano_transform[..., :3, 3] += self.shape_offsets_template - self.shape_offsets
        ober2cano_transform[..., :3, 3] += self.pose_offsets_template - self.pose_offsets
        self.ober2cano_transform = torch.matmul(self.verts_transform_template, ober2cano_transform)
    
    def get_neighbs(self, xyz, verts, verts_transform_inv):
        bs, nv = verts.shape[:2]
        device = verts.device
 
        if self.use_knn:
            with torch.no_grad():
                neighbs_dist, neighbs = self.knn(verts, xyz)
        else:
            xyz_v = xyz.unsqueeze(2) - verts.unsqueeze(1)
            dist = torch.norm(xyz_v, dim=-1, p=2)
            neighbs_dist, neighbs = dist.topk(self.k_neigh, largest=False, dim=-1)

        weight_std2 = 2. * self.weight_std ** 2
        xyz_neighbs_lbs_weight = self.body_model.lbs_weights[neighbs] # (bs, n_rays*K, k_neigh, 24)
        xyz_neighbs_weight_conf = torch.exp(-torch.sum(torch.abs(xyz_neighbs_lbs_weight - xyz_neighbs_lbs_weight[..., 0:1, :]), dim=-1)/weight_std2) # (bs, n_rays*K, k_neigh)
        xyz_neighbs_weight_conf = torch.gt(xyz_neighbs_weight_conf, 0.9).float()
        xyz_neighbs_weight = torch.exp(-neighbs_dist) # (bs, n_rays*K, k_neigh)
        xyz_neighbs_weight *= xyz_neighbs_weight_conf
        xyz_neighbs_weight = xyz_neighbs_weight / xyz_neighbs_weight.sum(-1, keepdim=True) # (bs, n_rays*K, k_neigh)

        xyz_neighbs_transform_inv = batch_index_select(verts_transform_inv, neighbs) # (bs, n_rays*K, k_neigh, 4, 4)
        xyz_transform_inv = torch.sum(xyz_neighbs_weight.unsqueeze(-1).unsqueeze(-1) * xyz_neighbs_transform_inv, dim=2) # (bs, n_rays*K, 4, 4)

        xyz_dist = torch.sum(xyz_neighbs_weight * neighbs_dist, dim=2, keepdim=True) # (bs, n_rays*K, 1)

        return xyz_dist, xyz_transform_inv

    def unpose(self, xyz, viewdir=None):
        bs, nv = xyz.shape[:2]
        xyz_dist, xyz_transform_inv = self.get_neighbs(xyz, self.verts, self.ober2cano_transform.clone())

        xyz_valid = torch.lt(xyz_dist, self.dis_threshold).float()
        # xyz_transfrom_ident = torch.eye(4, dtype=xyz_transform_inv.dtype, device=xyz_transform_inv.device).expand(bs, nv, -1, -1)
        # xyz_transform_inv = xyz_valid.unsqueeze(-1) * xyz_transform_inv + (1 - xyz_valid.unsqueeze(-1)) * xyz_transfrom_ident

        xyz_unposed = batch_transform(xyz_transform_inv, xyz)
        if self.use_view and self.unpose_view and viewdir is not None:
            viewdir = batch_transform(xyz_transform_inv, viewdir)

        return xyz_unposed, viewdir, xyz_valid

    def deformation(self, xyz, xyz_valid=None):
        bs, nv = xyz.shape[:2]

        if self.deformation_dim > 0:
            deformation_code = self.deformation_code.unsqueeze(1).expand(-1, nv, -1)

        xyz_decoded = self.derf(xyz, deformation_code)
        rot = compute_rotation_matrix_from_ortho6d(xyz_decoded[..., :6])
        trans = xyz_decoded[..., 6:9]

        if xyz_valid is not None:
            rot_ident = torch.eye(3, dtype=rot.dtype, device=rot.device).expand(bs, nv, -1, -1)
            rot = xyz_valid.unsqueeze(-1) * rot + (1 - xyz_valid.unsqueeze(-1)) * rot_ident

        xyz_deformed = torch.matmul(rot, xyz.unsqueeze(-1)).squeeze(-1) + trans
        return xyz_deformed

    def query_canonical_space(self, xyz, viewdir=None, use_fine=False, only_sigma=False, only_normal=False):
        bs, nv = xyz.shape[:2]
        
        if not self.use_deformation and self.deformation_dim > 0:
            deformation_code = self.deformation_code.unsqueeze(1).expand(-1, nv, -1)
        else:
            deformation_code = None

        if only_sigma:
            if not use_fine:
                sigma = self.nerf.get_sigma(xyz, deformation_code=deformation_code, only_sigma=only_sigma)
            else:
                sigma = self.nerf_fine.get_sigma(xyz, deformation_code=deformation_code, only_sigma=only_sigma)
            return sigma

        if only_normal:
            if not use_fine:
                normal = self.nerf.get_normal(xyz, deformation_code=deformation_code)
            else:
                normal = self.nerf_fine.get_normal(xyz, deformation_code=deformation_code)
            return normal
        
        if self.apperance_dim > 0:
            apperance_code = self.apperance_code.unsqueeze(1).expand(-1, nv, -1)
        else:
            apperance_code = None

        if not use_fine:
            rgb, sigma = self.nerf(xyz, viewdir=viewdir, deformation_code=deformation_code, apperance_code=apperance_code)
        else:
            rgb, sigma = self.nerf_fine(xyz, viewdir=viewdir, deformation_code=deformation_code, apperance_code=apperance_code)

        return rgb, sigma

    def query_canonical_space_inside(self, xyz, viewdir=None, valid=None, use_fine=False, only_sigma=False, only_normal=False):
        bs, nv = xyz.shape[:2]
        device = xyz.device
        
        if not self.use_deformation and self.deformation_dim > 0:
            deformation_code = self.deformation_code.unsqueeze(1).expand(-1, nv, -1)
        else:
            deformation_code = None

        if valid is not None:
            inside_inds = (valid > 0).squeeze(-1)
        else:
            inside_inds = torch.bool(bs, nv, device=device)
        
        sigma = torch.zeros(bs, nv, 1, device=device).float() - 1e5

        if only_sigma:
            if not use_fine:
                sigma[inside_inds] = self.nerf.get_sigma(xyz[inside_inds], deformation_code=deformation_code[inside_inds])
            else:
                sigma[inside_inds] = self.nerf_fine.get_sigma(xyz[inside_inds], deformation_code=deformation_code[inside_inds])
            return sigma

        normal = torch.zeros(bs, nv, 3, device=device).float()
        if only_normal:
            if not use_fine:
                normal[inside_inds] = self.nerf.get_normal(xyz[inside_inds], deformation_code=deformation_code[inside_inds])
            else:
                normal[inside_inds] = self.nerf_fine.get_normal(xyz[inside_inds], deformation_code=deformation_code[inside_inds])
            return normal

        if self.apperance_dim > 0:
            apperance_code = self.apperance_code.unsqueeze(1).expand(-1, nv, -1)
        else:
            apperance_code = None

        rgb = torch.zeros(bs, nv, 3, device=device).float()

        if not use_fine:
            rgb[inside_inds], sigma[inside_inds] = self.nerf(xyz[inside_inds], viewdir=viewdir[inside_inds], deformation_code=deformation_code[inside_inds], apperance_code=apperance_code[inside_inds])
        else:
            rgb[inside_inds], sigma[inside_inds] = self.nerf_fine(xyz[inside_inds], viewdir=viewdir[inside_inds], deformation_code=deformation_code[inside_inds], apperance_code=apperance_code[inside_inds])

        return rgb, sigma
    
    def forward(self, xyz, viewdir=None, use_fine=False):
        bs, nv = xyz.shape[:2]

        if self.use_unpose:
            xyz, viewdir, valid = self.unpose(xyz, viewdir)
        else:
            valid = torch.ones((bs, nv, 1), dtype=xyz.dtype, device=xyz.device)
        
        if self.use_deformation:
            xyz = self.deformation(xyz, valid)

        if self.query_inside:
            rgb, sigma = self.query_canonical_space_inside(xyz, viewdir, valid, use_fine)
        else:
            rgb, sigma = self.query_canonical_space(xyz, viewdir, use_fine)
            sigma[valid < 1] = -1e5

        return rgb, sigma
    