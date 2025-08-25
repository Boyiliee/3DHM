import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import torch
from torchvision.utils import make_grid
import numpy as np
import pyrender
import trimesh
import cv2
import math



def create_raymond_lights():
    import pyrender
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
        matrix[:3,:3] = np.c_[x,y,z]
        nodes.append(pyrender.Node(
          light=pyrender.DirectionalLight(color=np.ones(3), intensity=1.0),
          matrix=matrix
        ))
    return nodes


LIGHT_BLUE=(0.65098039,  0.74117647,  0.85882353)

class Renderer:
    """
    Renderer used for visualizing the SMPL model
    Code adapted from https://github.com/vchoutas/smplify-x
    """
    def __init__(self, focal_length=5000, img_res=224, faces=None):
        self.renderer = pyrender.OffscreenRenderer(viewport_width=img_res, viewport_height=img_res, point_size=1.0)
        self.focal_length = focal_length
        self.camera_center = [img_res // 2, img_res // 2]
        self.faces = faces
        self.img_res = img_res

    def visualize_tb(self, vertices, camera_translation):
        rend_imgs = []
        rend_masks = []
        rend_depths = []
        for i in range(vertices.shape[0]):
            # ############# debug and add rotation
            # rotation_matrix = np.array([[0, -1, 0],
            #                             [1,  0, 0],
            #                             [0,  0, 1]])
            # # Apply the rotation
            # vertices = np.matmul(vertices, rotation_matrix)
            # ############# debug and add rotation
            rend_img, valid_mask, depth = self.visualize_all(vertices[i:i+1], camera_translation[i:i+1], np.zeros((1, 3))+1, np.zeros((self.img_res, self.img_res, 3)), use_image=True, side_view=False)
            rend_imgs.append(torch.from_numpy(rend_img))
            rend_masks.append(torch.from_numpy(valid_mask))
            rend_depths.append(torch.from_numpy(depth))
        rend_imgs = torch.stack(rend_imgs, dim=0)
        rend_masks = torch.stack(rend_masks, dim=0)
        rend_depths = torch.stack(rend_depths, dim=0)
        return rend_imgs, rend_masks, rend_depths
    
    def visualize_all(self, vertices, camera_translation, color, images, use_image=True, side_view=False):

        baseColorFactors = np.hstack([color[:, [2,1,0]], np.ones((color.shape[0], 1))])
        verts = vertices.copy()
        fl   = self.focal_length
        if side_view:
            around = cv2.Rodrigues(np.array([0, math.radians(90), 0]))[0]
            center = verts.mean(axis=1)
            verts = np.dot((verts - center), around) + center

        
        cam_trans = camera_translation.copy()
        verts = verts + cam_trans#[:,None]
        
        if(use_image): color, depth = self.__call__(verts, focal_length=fl, baseColorFactors=baseColorFactors)
        else: color, depth = self.other_call(verts, focal_length=fl, baseColorFactors=baseColorFactors)
            
        
        valid_mask = color.sum(axis=2, keepdims=True)>0
        if(use_image):
            output_img = color[:, :, :3] * valid_mask + (1 - valid_mask) * images
        else:
            output_img = color[:, :, :3] * valid_mask
        return output_img, valid_mask, depth
    
    
    def __call__(self, vertices, focal_length=5000, baseColorFactors=[(1.0, 1.0, 0.9, 1.0)]):
        scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],
                               ambient_light=(0.3, 0.3, 0.3))
        
        for i_, verts in enumerate(vertices):
            # material = pyrender.MetallicRoughnessMaterial(
            #     metallicFactor=0.0,
            #     alphaMode='OPAQUE',
            #     baseColorFactor=baseColorFactors[i_])
            material = pyrender.MetallicRoughnessMaterial(
                metallicFactor=0.0,
                roughnessFactor=0.5,
                alphaMode='OPAQUE',
                baseColorFactor=baseColorFactors[i_])
            
            mesh = trimesh.Trimesh(verts.copy(), self.faces.copy())
            rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
            mesh.apply_transform(rot)
            mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
            scene.add(mesh, 'mesh')


        camera_pose = np.eye(4)
        camera = pyrender.IntrinsicsCamera(fx=self.focal_length, fy=self.focal_length, cx=self.camera_center[0], cy=self.camera_center[1], zfar=1000)
        scene.add(camera, pose=camera_pose)


        light_nodes = create_raymond_lights()
        for node in light_nodes:
            scene.add_node(node)

    
        color, rend_depth = self.renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        color = color.astype(np.float32) / 255.0
        
        output_img = color[:, :, :3]

        
        return output_img, rend_depth



    def other_call(self, vertices, focal_length=5000, baseColorFactors=[(1.0, 1.0, 0.9, 1.0)]):
        scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=(1.0, 1.0, 1.0))
        
        for i_, verts in enumerate(vertices):
            material = pyrender.MetallicRoughnessMaterial(
                metallicFactor=0.0,
                alphaMode='OPAQUE',
                baseColorFactor=baseColorFactors[i_])
            
            mesh = trimesh.Trimesh(verts.copy(), self.faces.copy())
            rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
            mesh.apply_transform(rot)
            mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
            scene.add(mesh, 'mesh')


        camera_pose = np.eye(4)
        camera = pyrender.IntrinsicsCamera(fx=self.focal_length, fy=self.focal_length, cx=self.camera_center[0], cy=self.camera_center[1], zfar=1000)
        scene.add(camera, pose=camera_pose)

        color, rend_depth = self.renderer.render(scene, flags=pyrender.RenderFlags.FLAT)
        color = color.astype(np.float32) / 255.0
        
        output_img = color[:, :, :3]

        
        return output_img, rend_depth
