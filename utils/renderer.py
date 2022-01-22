import math
import trimesh
import pyrender
import numpy as np
from pyrender.constants import RenderFlags

class WeakPerspectiveCamera(pyrender.Camera):
    PIXEL_CENTER_OFFSET = 0.5
    def __init__(self,
                 scale,
                 translation,
                 znear=pyrender.camera.DEFAULT_Z_NEAR,
                 zfar=pyrender.camera.DEFAULT_Z_FAR,
                 name=None):
        super(WeakPerspectiveCamera, self).__init__(
            znear=znear,
            zfar=zfar,
            name=name,
        )
        self.scale = scale
        self.translation = translation

    def get_projection_matrix(self, width=None, height=None):
        P = np.eye(4)
        P[0, 0] = self.scale[0]
        P[1, 1] = self.scale[1]
        P[0, 3] = self.translation[0] * self.scale[0]
        P[1, 3] = -self.translation[1] * self.scale[1]
        P[2, 2] = -1

        return P

class Renderer:
    def __init__(self, resolution=(224,224), wireframe=False, use_raymond_lighting=True):
        self.resolution = resolution
        self.wireframe = wireframe
        self.renderer = pyrender.OffscreenRenderer(
            viewport_width=self.resolution[0],
            viewport_height=self.resolution[1],
            point_size=1.0
        )

        # set the scene
        if use_raymond_lighting:
            self.scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=(0.0, 0.0, 0.0))
        else:
            self.scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=(0.3, 0.3, 0.3))

        if use_raymond_lighting:
            light_nodes = self._create_raymond_lights()
            for node in light_nodes:
                self.scene.add_node(node)
        else:
            light_node = pyrender.Node(pyrender.DirectionalLight())
            self.scene.add_node(light_node)

    def _create_raymond_lights(self):
        thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
        phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])

        nodes = []

        for phi, theta in zip(phis, thetas):
            xp = np.sin(theta) * np.cos(phi)
            yp = np.sin(theta) * np.sin(phi)
            zp = np.cos(theta)

            z = np.array([xp, yp, zp])
            z = z / np.linalg.norm(z)
            x = np.array([-z[1], z[0], 0.0])
            if np.linalg.norm(x) == 0:
                x = np.array([1.0, 0.0, 0.0])
            x = x / np.linalg.norm(x)
            y = np.cross(z, x)

            matrix = np.eye(4)
            matrix[:3, :3] = np.c_[x, y, z]
            nodes.append(
                pyrender.Node(
                    light=pyrender.DirectionalLight(color=np.ones(3),
                                                    intensity=1.0),
                    matrix=matrix
                ))

        return nodes
    
    def set_camera(self, fx, fy, cx, cy, R, t):
        for node in self.scene.get_nodes():
            if node.name == 'camera':
                self.scene.remove_node(node)
        camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, zfar=1000)
        camera_pose = np.eye(4)
        camera_pose[:3, :3] = R
        camera_pose[:3, 3] = t
        self.scene.add(camera, pose=camera_pose, name='camera')

    def render(self, verts, faces,  verts_color=None, angle=180, axis=[1, 0, 0], mesh_filename=None, color=[1.0, 1.0, 0.9], smooth=True):

        mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_colors=verts_color, process=False)
        mesh.vertex_colors = verts_color

        Rx = trimesh.transformations.rotation_matrix(math.radians(angle), axis)
        mesh.apply_transform(Rx)

        if mesh_filename is not None:
            mesh.export(mesh_filename)

        if verts_color is None:
            material = pyrender.MetallicRoughnessMaterial(
                metallicFactor=0.0,
                alphaMode='OPAQUE',
                baseColorFactor=(color[0], color[1], color[2], 1.0)
            )
            mesh = pyrender.Mesh.from_trimesh(mesh, smooth=smooth, material=material, wireframe=self.wireframe)
        else:
            mesh = pyrender.Mesh.from_trimesh(mesh, smooth=smooth, wireframe=self.wireframe)

        mesh_node = self.scene.add(mesh, 'mesh')

        if self.wireframe:
            render_flags = RenderFlags.RGBA | RenderFlags.ALL_WIREFRAME
        else:
            render_flags = RenderFlags.RGBA

        rgba, _ = self.renderer.render(self.scene, flags=render_flags)

        self.scene.remove_node(mesh_node)

        return rgba