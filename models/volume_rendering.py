import torch
import torch.nn as nn
import torch.nn.functional as F
if torch.__version__ < '1.6.0':
    from torchsearchsorted import searchsorted

class VolumeRenderer(nn.Module):
    def __init__(
        self, 
        n_coarse=64,
        n_fine=0,
        n_fine_depth=0,
        share_fine=False,
        noise_std=1.0,
        depth_std=0.02,
        white_bkgd=True,
        lindisp=True
        ):
        super(VolumeRenderer, self).__init__()
        self.n_coarse = n_coarse
        self.n_fine = n_fine
        self.n_fine_depth = n_fine_depth
        self.share_fine = share_fine
        self.noise_std = noise_std
        self.depth_std = depth_std
        self.lindisp = lindisp
        self.white_bkgd = white_bkgd
        
    def sample_coarse(self, rays, perturb=0.):
        """
        Stratified sampling. Note this is different from original NeRF slightly.
        :param rays ray [origins (3), directions (3), near (1), far (1)] (bs, n_rays, 8)
        :return (bs, n_rays, Kc)
        """
        bs, n_rays = rays.shape[:2]
        device = rays.device
        near, far = rays[..., 6:7], rays[..., 7:8]  # (bs, n_rays, 1)

        step = 1.0 / self.n_coarse
        z_steps = torch.linspace(0, 1 - step, self.n_coarse, device=device)  # (Kc)
        z_steps = z_steps.unsqueeze(0).unsqueeze(0).repeat(bs, n_rays, 1)  # (bs, n_rays, Kc)

        if self.lindisp:  # Use linear sampling in depth space
            z_samp =  near * (1 - z_steps) + far * z_steps  # (bs, n_rays, Kc)
        else:  # Use linear sampling in disparity space
            z_samp = 1 / (1 / near * (1 - z_steps) + 1 / far * z_steps)  # (bs, n_rays, Kc)

        if perturb > 0:
            mids = .5 * (z_samp[..., 1:] + z_samp[..., :-1])
            upper = torch.cat([mids, z_samp[..., -1:]], -1)
            lower = torch.cat([z_samp[..., :1], mids], -1)
            # stratified samples in those intervals
            t_rand = perturb * torch.rand(z_samp.shape, device=device)
            z_samp = lower + (upper - lower) * t_rand

        return z_samp
    
    
    def sample_fine(self, bins, weights, det=False, eps=1e-5):
        """
        Weighted stratified (importance) sample
        :param bins (bs, n_rays, Kc-1)
        :param weights (bs, n_rays,, Kc-2)
        :return (bs, n_rays,, Kf)
        """
        bs, n_rays = bins.shape[:2]
        device = bins.device

        weights = weights.detach() + eps  # Prevent division by zero
        pdf = weights / torch.sum(weights, -1, keepdim=True)  # (bs, n_rays, Kc)
        cdf = torch.cumsum(pdf, -1)  # (bs, n_rays, Kc)
        cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)  # (bs, n_rays, Kc+1)

        if det:
            u = torch.linspace(0., 1., steps=self.n_fine, device=device)
            u = u.expand(bs, n_rays, self.n_fine)
        else:
            u = torch.rand(bs, n_rays, self.n_fine, device=device)  # (bs, n_rays, Kf)
        
        u = u.contiguous()

        if torch.__version__ < '1.6.0':
            inds = searchsorted(cdf.reshape(bs*n_rays, -1), u.reshape(bs*n_rays, -1), side='right').reshape(bs, n_rays, -1)
        else:
            inds = torch.searchsorted(cdf, u, right=True)
        below = torch.clamp_min(inds - 1, 0)
        above = torch.clamp_max(inds, self.n_coarse - 2)

        inds_sampled = torch.stack([below, above], -1).view(bs, n_rays, self.n_fine * 2)
        cdf_g = torch.gather(cdf, 2, inds_sampled).view(bs, n_rays, self.n_fine, 2)
        bins_g = torch.gather(bins, 2, inds_sampled).view(bs, n_rays, self.n_fine, 2)

        denom = cdf_g[..., 1] - cdf_g[..., 0]
        denom[denom < eps] = 1
        z_samp = bins_g[..., 0] + (u - cdf_g[..., 0]) / denom * (bins_g[..., 1] - bins_g[..., 0])

        return z_samp

    def sample_fine_depth(self, rays, depth):
        """
        Sample around specified depth
        :param rays ray [origins (3), directions (3), near (1), far (1)] (bs, n_rays, 8)
        :param depth (bs, n_rays, 1)
        :return (bs, n_rays, Kfd)
        """
        z_samp = depth.repeat(1, 1, self.n_fine_depth)
        z_samp += torch.randn_like(z_samp) * self.depth_std
        # Clamp does not support tensor bounds
        near, far = rays[..., 6:7], rays[..., 7:8]  # (B, 1)
        z_samp = torch.min(torch.max(z_samp, near), far)
        return z_samp
    
    def composite(self, model, rays, z_samp, coarse=True, far=True, perturb=0., **kwargs):
        bs, n_rays, K = z_samp.shape

        # (bs, n_rays, K, 3)
        xyz = rays[..., None, :3] + z_samp.unsqueeze(-1) * rays[..., None, 3:6]
        viewdir = rays[..., None, 3:6].expand(-1, -1, K, -1)
        xyz = xyz.reshape(bs, -1, 3)  # (bs, n_rays*K, 3)
        viewdir = viewdir.reshape(bs, -1, 3) # (bs, n_rays*K, 3)
        
        # (bs, n_rays*K, 4)
        rgbs, sigmas = model(xyz, viewdir, use_fine=not coarse, **kwargs)

        rgbs = rgbs.reshape(bs, n_rays, K, 3)
        sigmas = sigmas.reshape(bs, n_rays, K)

        if self.noise_std > 0.0 and perturb > 0:
            sigmas = sigmas + torch.randn_like(sigmas) * self.noise_std

        deltas = z_samp[..., 1:] - z_samp[..., :-1]  # (bs, n_rays, K-1)
        
        if far:
            delta_inf = 1e10 * torch.ones_like(deltas[..., :1])  # infty (bs, n_rays, 1)
        else:
            delta_inf = rays[..., 7:8] - z_samp[..., -1:]
        deltas = torch.cat([deltas, delta_inf], -1)  # (bs, n_rays, K)

        #deltas = deltas * torch.norm(rays[..., None, 3:6], dim=-1)

        # compute the gradients in log space of the alphas, for NV TV occupancy regularizer
        alphas = 1 - torch.exp(-deltas * torch.relu(sigmas))  # (bs, n_rays, K)
        alphas_shifted = torch.cat([torch.ones_like(alphas[..., :1]), 1 - alphas + 1e-10], -1)  # (bs, n_rays, K+1) = [1, a1, a2, ...]
        T = torch.cumprod(alphas_shifted, -1)  # (bs, n_rays, K+1)
        weights = alphas * T[..., :-1]  # (bs, n_rays, K)
        weights_sum = torch.sum(weights, dim=-1, keepdim=True) # (bs, n_rays, 1)

        rgb_final = torch.sum(weights.unsqueeze(-1) * rgbs, -2)  # (bs, n_rays, 3)
        depth_final = torch.sum(weights * z_samp, -1, keepdim=True)  # (bs, n_rays, 1)

        if self.white_bkgd:
            depth_final = depth_final + (1 - weights_sum) * rays[..., 7:8]
            rgb_final = rgb_final + 1 - weights_sum # (bs, n_rays, 3)
            
        return (
            weights,
            rgb_final,
            depth_final,
            weights_sum
        )
        
        
    def forward(self, model, rays, perturb=0., **kwargs):

        z_coarse = self.sample_coarse(rays[..., :8], perturb=perturb)  # (bs, n_rays, Kc)

        if self.n_fine > 0 and self.share_fine:
            with torch.no_grad():
                weights, rgbs, depths, alphas = self.composite(
                    model,
                    rays,
                    z_coarse,
                    coarse=True,
                    far=True,
                    perturb=perturb,
                    **kwargs
                    )
        else:
            weights, rgbs, depths, alphas = self.composite(
                model,
                rays,
                z_coarse,
                coarse=True,
                far=True,
                perturb=perturb,
                **kwargs
                )
            
        output = {
            'rgbs': rgbs,
            'alphas': alphas,
            'depths': depths
            }

        if self.n_fine > 0 or self.n_fine_depth > 0:
            z_combine = z_coarse

            if self.n_fine > 0:
                z_coarse_mid = 0.5 * (z_coarse[..., :-1] + z_coarse[..., 1:]) # (bs, n_rays, Kc - 1)
                z_fine = self.sample_fine(z_coarse_mid, weights[..., 1:-1].detach(), det=(perturb==0)).detach() # (bs, n_rays, Kf)
                z_combine = torch.cat([z_combine, z_fine], dim=-1)

            if self.n_fine_depth > 0:
                z_fine_depth = self.sample_fine_depth(rays, depth=depths)
                z_combine = torch.cat([z_combine, z_fine_depth.detach()], dim=-1)

            z_combine, _ = torch.sort(z_combine, dim=-1)

            weights_fine, rgbs_fine, depths_fine, alphas_fine = self.composite(
                model,
                rays,
                z_combine,
                coarse=False,
                far=True,
                perturb=perturb,
                **kwargs
                )

            if self.share_fine:
                output = {
                    'rgbs': rgbs_fine,
                    'alphas': alphas_fine,
                    'depths': depths_fine
                    }
            else:
                output.update({
                    'rgbs_fine': rgbs_fine,
                    'alphas_fine': alphas_fine,
                    'depths_fine': depths_fine
                })

        return output