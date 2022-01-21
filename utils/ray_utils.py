import cv2
import torch
import numpy as np

def get_pixelcoords(H, W, mask=None, subsampletype='pixel', subsamplesize=32, fore_rate=0.9, mask_erode=1):
    
    def sample(indx, indy, n_pixels):
            select_indexs = np.random.choice(indx.shape[0], n_pixels, replace=False)
            px = indx[select_indexs]
            py = indy[select_indexs]
            return px, py

    if subsampletype == 'pixel':
        indx, indy = np.meshgrid(
            np.arange(0, H),
            np.arange(0, W),
            indexing='ij'
            )
        indx = indx.flatten()
        indy = indy.flatten()
        px, py = sample(indx.flatten(), indy.flatten(), subsamplesize*subsamplesize)
        px = px.reshape(subsamplesize, subsamplesize)
        py = py.reshape(subsamplesize, subsamplesize)

    elif subsampletype == 'random_pixel':
        px = np.random.randint(0, H, size=(subsamplesize, subsamplesize))
        py = np.random.randint(0, W, size=(subsamplesize, subsamplesize))
    
    elif subsampletype == 'foreground_pixel':
        # foreground_pixels
        kernel = np.ones((mask_erode, mask_erode), np.uint8)
        mask_inside = cv2.erode(mask.copy(), kernel)
        mask_dilate1 = cv2.dilate(mask.copy(), kernel)
        kernel = np.ones((64, 64), np.uint8)
        mask_dilate2 = cv2.dilate(mask.copy(), kernel)
        mask_outside = mask_dilate2 - mask_dilate1

        indx, indy = np.where(mask_inside > 0)
        fore_pixels = int(subsamplesize*subsamplesize * fore_rate)
        fore_px, fore_py = sample(indx, indy, fore_pixels)
        indx, indy = np.where(mask_outside > 0)
        back_pixels = subsamplesize*subsamplesize - fore_pixels
        back_px, back_py = sample(indx, indy, back_pixels)
        px = np.concatenate((fore_px, back_px), axis=0).reshape(subsamplesize, subsamplesize)
        py = np.concatenate((fore_py, back_py), axis=0).reshape(subsamplesize, subsamplesize)

    elif subsampletype == "patch":
        indx = np.random.randint(0, H - subsamplesize + 1)
        indy = np.random.randint(0, W - subsamplesize + 1)
        px, py = np.meshgrid(
            np.arange(indx, indx + subsamplesize),
            np.arange(indy, indy + subsamplesize),
            indexing='ij'
            )
        
    elif subsampletype == 'foreground_patch':
        mask = mask[subsamplesize//2:H-subsamplesize//2, subsamplesize//2:W-subsamplesize//2, 0].numpy() > 0
        indx, indy = np.where(mask)
        indx = indx + subsamplesize // 2
        indy = indy + subsamplesize // 2
        select_index = np.random.choice(np.sum(mask))
        indx, indy = indx[select_index], indy[select_index]
        px, py = np.meshgrid(
            np.arange(indx - subsamplesize // 2, indx + subsamplesize // 2),
            np.arange(indy - subsamplesize // 2, indy + subsamplesize // 2),
            indexing='ij'
            )
    else:
        px, py = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')

    pixelcoords = np.stack((px, py), axis=-1)
    return pixelcoords

def get_ray_directions(H, W, focal):
    """
    Get ray directions for all pixels in camera coordinate.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems

    Inputs:
        H, W, focal: image height, width and focal length

    Outputs:
        directions: (H, W, 3), the direction of the rays in camera coordinate
    """
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    # the direction here is without +0.5 pixel centering as calibration is not so accurate
    # see https://github.com/bmild/nerf/issues/24
    directions = \
        torch.stack([(i-W/2)/focal, -(j-H/2)/focal, -torch.ones_like(i)], -1) # (H, W, 3)

    directions = directions / torch.norm(directions, dim=-1, keepdim=True)

    return directions


def get_rays(directions, c2w):
    """
    Get ray origin and normalized directions in world coordinate for all pixels in one image.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems

    Inputs:
        directions: (H, W, 3) precomputed ray directions in camera coordinate
        c2w: (3, 4) transformation matrix from camera coordinate to world coordinate

    Outputs:
        rays_o: (H, W, 3), the origin of the rays in world coordinate
        rays_d: (H, W, 3), the normalized direction of the rays in world coordinate
    """
    # Rotate ray directions from camera coordinate to the world coordinate
    rays_d = directions @ c2w[:, :3].T # (H, W, 3)
    # The origin of all rays is the camera origin in world coordinate
    rays_o = c2w[:, 3].expand(rays_d.shape) # (H, W, 3)

    # rays_d = rays_d.view(-1, 3)
    # rays_o = rays_o.view(-1, 3)

    return rays_o, rays_d


def get_ndc_rays(H, W, focal, near, rays_o, rays_d):
    """
    Transform rays from world coordinate to NDC.
    NDC: Space such that the canvas is a cube with sides [-1, 1] in each axis.
    For detailed derivation, please see:
    http://www.songho.ca/opengl/gl_projectionmatrix.html
    https://github.com/bmild/nerf/files/4451808/ndc_derivation.pdf

    In practice, use NDC "if and only if" the scene is unbounded (has a large depth).
    See https://github.com/bmild/nerf/issues/18

    Inputs:
        H, W, focal: image height, width and focal length
        near: (N_rays) or float, the depths of the near plane
        rays_o: (N_rays, 3), the origin of the rays in world coordinate
        rays_d: (N_rays, 3), the direction of the rays in world coordinate

    Outputs:
        rays_o: (N_rays, 3), the origin of the rays in NDC
        rays_d: (N_rays, 3), the direction of the rays in NDC
    """
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d

    # Store some intermediate homogeneous results
    ox_oz = rays_o[...,0] / rays_o[...,2]
    oy_oz = rays_o[...,1] / rays_o[...,2]
    
    # Projection
    o0 = -1./(W/(2.*focal)) * ox_oz
    o1 = -1./(H/(2.*focal)) * oy_oz
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - ox_oz)
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - oy_oz)
    d2 = 1 - o2
    
    rays_o = torch.stack([o0, o1, o2], -1) # (B, 3)
    rays_d = torch.stack([d0, d1, d2], -1) # (B, 3)
    
    return rays_o, rays_d