import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import os
import hydra
import torch
import numpy as np
import pandas as pd
import json
import itertools

from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig
import cv2
import shutil
import joblib
import pickle
from PIL import Image


from phalp.configs.base import FullConfig
from phalp.models.hmar.hmr import HMR2018Predictor
from phalp.trackers.PHALP import PHALP
from phalp.utils import get_pylogger
from phalp.configs.base import CACHE_DIR

from hmr2.datasets.utils import expand_bbox_to_aspect_ratio

import torch
from torch.utils.data import DataLoader, Dataset
import argparse
from pathlib import Path
from tqdm import tqdm

# Imports for Texture Extraction
from smt.utils.utils_dataset import process_image_pre as process_image
from smt.utils.smpl_utils import SMPL as SMPL_smt
from smt.utils.renderer import Renderer
from smt.utils.utils import compute_uvsampler
import neural_renderer as nr


warnings.filterwarnings('ignore')

log = get_pylogger(__name__)

class HMR2Predictor(HMR2018Predictor):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        # Setup our new model
        from hmr2.models import download_models, load_hmr2

        # Download and load checkpoints
        download_models()
        model, _ = load_hmr2()

        self.model = model
        self.model.eval()

    def forward(self, x):
        hmar_out = self.hmar_old(x)
        batch = {
            'img': x[:,:3,:,:],
            'mask': (x[:,3,:,:]).clip(0,1),
        }
        model_out = self.model(batch)
        out = hmar_out | {
            'pose_smpl': model_out['pred_smpl_params'],
            'pred_cam': model_out['pred_cam'],
        }
        return out
    
class HMR2023TextureSampler(HMR2Predictor):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)

        # Model's all set up. Now, load tex_bmap and tex_fmap
        # Texture map atlas
        root_path = Path(__file__).parent
        bmap = np.load(os.path.join(root_path, 'data/bmap_256.npy'))
        fmap = np.load(os.path.join(root_path, 'data/fmap_256.npy'))
        self.register_buffer('tex_bmap', torch.tensor(bmap, dtype=torch.float))
        self.register_buffer('tex_fmap', torch.tensor(fmap, dtype=torch.long))

        self.img_size = 256         #self.cfg.MODEL.IMAGE_SIZE
        self.focal_length = 5000.   #self.cfg.EXTRA.FOCAL_LENGTH

        import neural_renderer as nr
        self.neural_renderer = nr.Renderer(dist_coeffs=None, orig_size=self.img_size,
                                          image_size=self.img_size,
                                          light_intensity_ambient=1,
                                          light_intensity_directional=0,
                                          anti_aliasing=False)

    def forward(self, x):
        batch = {
            'img': x[:,:3,:,:],
            'mask': (x[:,3,:,:]).clip(0,1),
        }
        model_out = self.model(batch)

        # from hmr2.models.prohmr_texture import unproject_uvmap_to_mesh

        def unproject_uvmap_to_mesh(bmap, fmap, verts, faces):
            # bmap:  256,256,3
            # fmap:  256,256
            # verts: B,V,3
            # faces: F,3
            valid_mask = (fmap >= 0)

            fmap_flat = fmap[valid_mask]      # N
            bmap_flat = bmap[valid_mask,:]    # N,3

            face_vids = faces[fmap_flat, :]  # N,3
            face_verts = verts[:, face_vids, :] # B,N,3,3

            bs = face_verts.shape
            map_verts = torch.einsum('bnij,ni->bnj', face_verts, bmap_flat) # B,N,3

            return map_verts, valid_mask

        pred_verts = model_out['pred_vertices'] + model_out['pred_cam_t'].unsqueeze(1)
        device = pred_verts.device
        face_tensor = torch.tensor(self.smpl.faces.astype(np.int64), dtype=torch.long, device=device)
        map_verts, valid_mask = unproject_uvmap_to_mesh(self.tex_bmap, self.tex_fmap, pred_verts, face_tensor) # B,N,3

        # Project map_verts to image using K,R,t
        # map_verts_view = einsum('bij,bnj->bni', R, map_verts) + t # R=I t=0
        focal = self.focal_length / (self.img_size / 2)
        map_verts_proj = focal * map_verts[:, :, :2] / map_verts[:, :, 2:3] # B,N,2
        map_verts_depth = map_verts[:, :, 2] # B,N

        # Render Depth. Annoying but we need to create this
        K = torch.eye(3, device=device)
        K[0, 0] = K[1, 1] = self.focal_length
        K[1, 2] = K[0, 2] = self.img_size / 2  # Because the neural renderer only support squared images
        K = K.unsqueeze(0)
        R = torch.eye(3, device=device).unsqueeze(0)
        t = torch.zeros(3, device=device).unsqueeze(0)
        rend_depth = self.neural_renderer(pred_verts,
                                        face_tensor[None].expand(pred_verts.shape[0], -1, -1).int(),
                                        # textures=texture_atlas_rgb,
                                        mode='depth',
                                        K=K, R=R, t=t)

        rend_depth_at_proj = torch.nn.functional.grid_sample(rend_depth[:,None,:,:], map_verts_proj[:,None,:,:]) # B,1,1,N
        rend_depth_at_proj = rend_depth_at_proj.squeeze(1).squeeze(1) # B,N

        img_rgba = torch.cat([batch['img'], batch['mask'][:,None,:,:]], dim=1) # B,4,H,W
        img_rgba_at_proj = torch.nn.functional.grid_sample(img_rgba, map_verts_proj[:,None,:,:]) # B,4,1,N
        img_rgba_at_proj = img_rgba_at_proj.squeeze(2) # B,4,N

        visibility_mask = map_verts_depth <= (rend_depth_at_proj + 1e-4) # B,N
        img_rgba_at_proj[:,3,:][~visibility_mask] = 0

        # Paste image back onto square uv_image
        uv_image = torch.zeros((batch['img'].shape[0], 4, 256, 256), dtype=torch.float, device=device)
        uv_image[:, :, valid_mask] = img_rgba_at_proj

        out = {
            'uv_image':  uv_image,
            'uv_vector' : self.hmar_old.process_uv_image(uv_image),
            'pose_smpl': model_out['pred_smpl_params'],
            'pred_cam':  model_out['pred_cam'],
        }
        return out

class HMR2_4dhuman(PHALP):
    def __init__(self, cfg):
        super().__init__(cfg)

    def setup_hmr(self):
        self.HMAR = HMR2023TextureSampler(self.cfg)

    def get_detections(self, image, frame_name, t_, additional_data=None, measurments=None):
        outputs     = self.detector(image)   

        height, width = image.shape[:2]
        
        pred_bbox = np.array([[0, 0, width, height]])  
        pred_scores = np.array([0.9])  
        pred_classes = np.array([1])  
        pred_masks = np.zeros((1, height, width), dtype=np.uint8)  
        ground_truth_track_id = [1 for i in list(range(len(pred_scores)))]
        ground_truth_annotations = [[] for i in list(range(len(pred_scores)))]
        # Pad bounding boxes 
        pred_bbox_padded = expand_bbox_to_aspect_ratio(pred_bbox)
        # Hack to predict SMPL on non-human data

        return (
            pred_bbox, pred_bbox_padded, pred_masks, pred_scores, pred_classes,
            ground_truth_track_id, ground_truth_annotations
        )

@dataclass
class Human4DConfig(FullConfig):
    # override defaults if needed
    expand_bbox_shape: Optional[Tuple[int]] = (192,256)
    driving_videos: Optional[str] = None
    reference_imgs: Optional[str] = None

    pass

cs = ConfigStore.instance()
cs.store(name="config", node=Human4DConfig)

class VideoDataset(Dataset):
    def __init__(self, directory, transform=None):  
        self.directory = directory  
        self.transform = transform  
        self.videos = []
        
        # Traverse the directory to find all video files  
        for video_source in os.listdir(directory):
            video_source = os.path.join(directory, video_source)
            if os.path.isdir(video_source):
                for video in os.listdir(video_source):  
                    if video.endswith(('.mp4', '.avi', '.mov')):
                        self.videos.append(os.path.join(video_source, video))
    
    def __len__(self):  
        return len(self.videos)  
    
    def __getitem__(self, idx):  
        video_path = self.videos[idx]
        return video_path
    
class SingleVideoDataset(Dataset):
    def __init__(self, directory, transform=None):  
        self.directory = directory  
        self.transform = transform  
        self.videos = []
        
        # Traverse the directory to find all video files  
        for video in os.listdir(directory):
            if video.endswith(('.mp4', '.avi', '.mov')):
                self.videos.append(os.path.join(directory, video))
            elif video.endswith(('.jpg', '.png')):
                self.videos.append(directory)
    def __len__(self):  
        return len(self.videos)  
    
    def __getitem__(self, idx):  
        video_path = self.videos[idx]
        return video_path

def flatten_row_with_empty_lists(row):
    """
    Flattens a row of lists while preserving empty lists as [[]].

    Parameters:
    row (pd.Series): A row from a pandas DataFrame.

    Returns:
    list: A list containing the flattened elements with empty lists preserved as [[]].
    """
    result = []
    
    for sublist in row:
        if not sublist:  # Check if the sublist is empty
            result.append([])  # Preserve the empty list as [[]]
        else:
            if type(sublist) is list:
                for item in sublist:  # Append each item from the non-empty sublist
                    result.append(item)
            else:
                    result.append(sublist)

    
    return result

def preprocess_reference_images(base_ref_path: Path, num_frames: int = 5, fps: int = 1):
    """
    Converts PNG images in base_ref_path/images/ to 5-frame MP4 videos.
    Output videos are saved in base_ref_path/image_stem/image_stem.mp4.
    """
    log.info(f"Starting preprocessing of reference images in {base_ref_path / 'images'}")
    image_dir = base_ref_path / "images"

    if not image_dir.is_dir():
        log.warning(f"Subdirectory 'images' not found in {base_ref_path}. Skipping reference image preprocessing.")
        return

    processed_image_count = 0
    # Process PNG and JPG images with a single glob pattern
    # Search for all supported image formats
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        image_files.extend(image_dir.glob(ext))
    
    for img_path in image_files:
        folder_name = img_path.stem  # e.g., your_ref_img_A
        output_folder_path = base_ref_path / folder_name
        os.makedirs(output_folder_path, exist_ok=True)

        img = cv2.imread(str(img_path))
        if img is None:
            log.warning(f"Could not read image {img_path}. Skipping.")
            continue

        video_name = f"{folder_name}.mp4"
        output_video_path = output_folder_path / video_name

        height, width, layers = img.shape
        if layers != 3:
            log.warning(f"Image {img_path} does not have 3 channels (shape: {img.shape}). Skipping.")
            continue
            
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        try:
            video_writer = cv2.VideoWriter(str(output_video_path), fourcc, float(fps), (width, height))
            if not video_writer.isOpened():
                log.error(f"Failed to open VideoWriter for {output_video_path}. Check OpenCV installation and codecs.")
                continue
            for _ in range(num_frames):
                video_writer.write(img)
            video_writer.release()
            log.info(f"Created video {output_video_path} from {img_path}")
            processed_image_count += 1
        except Exception as e:
            log.error(f"Error writing video for {img_path}: {e}")
            if 'video_writer' in locals() and video_writer.isOpened():
                video_writer.release() # Ensure writer is released on error

    if processed_image_count == 0 and list(image_dir.glob("*.png")): # Check if there were PNGs but none processed
        log.warning(f"No PNG images were successfully processed into videos from {image_dir}.")
    elif processed_image_count > 0:
        log.info(f"Finished preprocessing {processed_image_count} reference images into videos.")
    else:
        log.info(f"No PNG images found in {image_dir} to preprocess.")


def reformat_pkl_files(base_path: Path):
    """
    Finds all 'demo_img.pkl' files in subdirectories of base_path,
    renames them, and reformats their content.
    """
    log.info(f"Reformatting .pkl files in: {base_path}")

    # Find all 'demo_img.pkl' files under the base_path.
    # The structure is expected to be base_path/video_name/results/demo_img.pkl
    original_pkl_paths = list(base_path.glob("**/results/*.pkl"))

    if not original_pkl_paths:
        log.warning(f"No '.pkl' files found under {base_path}. Nothing to reformat.")
        return

    for ori_pkl_path in tqdm(original_pkl_paths, desc="Reformatting pkl files"):
        results_dir = ori_pkl_path.parent
        video_subfolder_name = results_dir.parent.name
        
        # Clean and reformat the pkl file.
        try:
            with open(ori_pkl_path, 'rb') as f:
                data_null = joblib.load(f)
        except Exception as e:
            log.error(f"Failed to load {ori_pkl_path}: {e}")
            continue

        if not isinstance(data_null, dict) or not data_null:
            log.warning(f"Skipping reformatting for {ori_pkl_path} as it's empty or not a dictionary.")
            continue
            
        df = pd.DataFrame(data_null)
        first_val_dict = list(data_null.values())[0]
        for current_numer, pkl_dict in enumerate(data_null.values()):
            if current_numer == 0:
                for key, val in first_val_dict.items():
                    first_val_dict[key] = df.loc[key].to_numpy()
                break
        first_val_dict["pose_shape"] = flatten_row_with_empty_lists(first_val_dict.pop("pose"))
        
        frame_paths_abs = flatten_row_with_empty_lists(first_val_dict.pop("frame_path"))
        frame_paths_rel = [os.path.relpath(p, results_dir) if (p and os.path.isabs(p)) else p for p in frame_paths_abs]
        first_val_dict["frame_name"] = frame_paths_rel
        
        first_val_dict["has_detection"] = np.array([[[1.]]] * len(first_val_dict["frame_name"]))
        first_val_dict["mask"] = flatten_row_with_empty_lists(first_val_dict["mask"])
        
        first_val_dict["frame_bbox"] = []
        first_val_dict["smpl"] = []
        curr_index = 0
        for key, val in data_null.items():
            if curr_index == 0:
                curr_index += 1
                continue
            curr_index += 1
            first_val_dict["frame_bbox"].append(val["bbox"][0])
            first_val_dict["smpl"].append(val["smpl"][0])
        first_val_dict["smpl"].append(first_val_dict["smpl"][-1]) 
        first_val_dict["frame_bbox"].append(first_val_dict["frame_bbox"][-1])   
        
        new_cb = []
        for i in first_val_dict["camera_bbox"]:
            new_cb.append(np.array(i)[[0]])
            
        first_val_dict["camera_bbox"] = np.array(new_cb)
        for i in range(len(first_val_dict["camera_bbox"])):
            if np.array(first_val_dict["camera_bbox"][i]).shape[0] > 1:
                print("more than 1 camera:", ori_pkl_path)
                first_val_dict["camera_bbox"] = [c_bbox[:1] for c_bbox in first_val_dict["camera_bbox"]]
                break
        skip_keys = ["frame"]
        for key in skip_keys:
            if key in first_val_dict.keys():
                first_val_dict.pop(key)
                
        with open(ori_pkl_path, 'wb') as f:
            joblib.dump(first_val_dict, f)
        num_frames = len(first_val_dict["frame_name"])
        dest_pkl_path = ori_pkl_path.parent / f"eval_{video_subfolder_name}_1_{num_frames}.pkl"
        shutil.move(ori_pkl_path, dest_pkl_path)

def masked_median(x, mask, dim=0):
    """Compute the median of tensor x along dim, ignoring values where mask is False."""
    mask = mask.bool()
    x_nan = x.float().masked_fill(~mask, float("nan"))
    x_median, _ = x_nan.nanmedian(dim=dim)
    return x_median

def unproject_uvmap_to_mesh(bmap, fmap, verts, faces):
    # bmap:  H,W,3
    # fmap:  H,W
    # verts: B,V,3
    # faces: F,3
    valid_mask = (fmap >= 0)
    fmap_flat = fmap[valid_mask]      # N
    bmap_flat = bmap[valid_mask,:]    # N,3
    face_vids = faces[fmap_flat, :]  # N,3
    face_verts = verts[:, face_vids, :] # B,N,3,3
    map_verts = torch.einsum('bnij,ni->bnj', face_verts, bmap_flat) # B,N,3
    return map_verts, valid_mask

class TextureExtractor:
    def __init__(self, device=0, img_size=1024):
        self.device = device
        self.img_size = img_size

        root_path = Path(__file__).parent
        smpl_cfg_dict = {
            "MODEL_PATH" : os.path.join(root_path, "data/models/smpl/"),
            "GENDER": "neutral",
            "MODEL_TYPE": "smpl",
            "NUM_BODY_JOINTS": 23,
            "JOINT_REGRESSOR_EXTRA": os.path.join(root_path, "data/SMPL_to_J19.pkl"),
            "TEXTURE": os.path.join(root_path, "data/texture.npz"),
        }
        
        smpl_cfg = {k.lower(): v for k,v in smpl_cfg_dict.items()}
        self.smpl = SMPL_smt(**smpl_cfg).to(self.device)
        self.face_tensor = torch.tensor(self.smpl.faces.astype(np.int64), dtype=torch.long, device=self.device)

        bmap_path = Path(root_path / f'data/bmap_{self.img_size}.npy')
        fmap_path = Path(root_path / f'data/fmap_{self.img_size}.npy')
        if not bmap_path.exists() or not fmap_path.exists():
             raise FileNotFoundError(f"bmap/fmap for size {self.img_size} not found. Please ensure 'bmap_{self.img_size}.npy' and 'fmap_{self.img_size}.npy' are in the working directory.")
        self.bmap = torch.tensor(np.load(bmap_path), dtype=torch.float, device=self.device)
        self.fmap = torch.tensor(np.load(fmap_path), dtype=torch.long, device=self.device)

        self.focal_length = 5000.
        self.focal = self.focal_length / (self.img_size / 2)
        self.K = torch.eye(3, device=self.device)
        self.K[0, 0] = self.K[1, 1] = self.focal_length
        self.K[1, 2] = self.K[0, 2] = self.img_size / 2.0
        self.K = self.K.unsqueeze(0)
        self.R = torch.eye(3, device=self.device).unsqueeze(0).float()
        self.t = torch.zeros(3, device=self.device).unsqueeze(0).float()

        self.nmr_renderer = nr.Renderer(dist_coeffs=None, orig_size=self.img_size,
                                        image_size=self.img_size,
                                        light_intensity_ambient=0.8,
                                        light_intensity_directional=0.0,
                                        anti_aliasing=False, far=200).to(self.device)
        
        self.mean = torch.tensor([0.5, 0.5, 0.5], device=self.device).view(3, 1, 1) * 255.0
        self.std = torch.tensor([0.5, 0.5, 0.5], device=self.device).view(3, 1, 1) * 255.0
        tmask_standard = cv2.imread(str(root_path / 'data/smpl_uv_mask.png')) / 255.0
        tmask_standard = cv2.resize(tmask_standard, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        threshold_value = 0.01 # Example threshold (depends on your specific mask values)
        _, binary_mask = cv2.threshold(tmask_standard, threshold_value, 1, cv2.THRESH_BINARY)
        tmask_standard = tmask_standard * binary_mask
        self.tmask_standard = torch.tensor(tmask_standard)[:,:,0].unsqueeze(0).unsqueeze(0).to(self.device)
        
    def extract_from_pkl(self, pkl_path: Path, frame_limit=1000, frame_interval=10, read_only_first_frame=False):
        log.info(f"Extracting texture from {pkl_path}")
        pkl_name = pkl_path.stem
        tmap_dir = pkl_path.parent.parent # Assumes structure: base/video_name/results/
        save_file_path = tmap_dir / f"texture.png"
        
        if save_file_path.exists():
            try:
                texture_img = cv2.imread(str(save_file_path))
                if texture_img is not None and texture_img.shape[0] >= self.img_size:
                    log.info(f"High-res texture already exists for {pkl_path}. Skipping.")
                    return
            except Exception as e:
                log.warning(f"Could not read existing texture {save_file_path}: {e}")

        try:
            a1 = joblib.load(pkl_path)
        except Exception as e:
            log.error(f'*** {pkl_path} could not be loaded: {e} ***')
            return
        a1["frame_name"] = [(pkl_path.parent / Path(f)).resolve() for f in a1["frame_name"]]
        show_uvs = []
        show_masks = []
        
        end_frame = min(len(a1["frame_name"]), frame_limit)
        if read_only_first_frame:
             indices = [0]
        else:
             indices = range(0, end_frame, frame_interval)

        for index in tqdm(indices, desc=f"Processing frames for {pkl_name}"):
            if not Path(a1["frame_name"][index]).exists():
                log.warning(f"Image not found: {a1['frame_name'][index]}")
                continue
            # Use absolute path for frame_name
            frame_path = Path(a1["frame_name"][index])
            if not frame_path.is_absolute():
                frame_path = (pkl_path.parent / frame_path).resolve()
            img = cv2.imread(str(frame_path))
            if img is None:
                continue

            if len(a1['frame_bbox']) == 0: continue
            bbox = a1['frame_bbox'][index].astype(int)
            camera = np.array(a1['camera_bbox'][index][:1])
            
            if isinstance(a1['smpl'][index], int):
                a1['smpl'][index] = a1['smpl'][index-1]

            smpl_params = a1['smpl'][index]
            global_orient = torch.from_numpy(smpl_params['global_orient'][None, :, :, :]).to(self.device)
            body_pose = torch.from_numpy(smpl_params['body_pose'][None, :, :, :]).to(self.device)
            betas = torch.from_numpy(smpl_params['betas'][None, :]).to(self.device)

            x1, y1, w, h = bbox
            center_ = np.array([x1 + w/2, y1 + h/2])
            scale_ = max(w, h)
            image_tmp = process_image(img, center_, 1.0*np.max(scale_), output_size=self.img_size)
            img_tf = torch.tensor(image_tmp[[2, 1, 0], :, :], dtype=torch.float, device=self.device).unsqueeze(0)

            camera_tf = torch.from_numpy(camera).float().to(self.device)
            focal_length_tf = torch.ones((camera_tf.shape[0], 1), device=self.device) * self.focal_length
            pred_cam_t = torch.stack([camera_tf[:,1], camera_tf[:,2], 2*focal_length_tf[:, 0]/(self.img_size * camera_tf[:,0] + 1e-9)],dim=-1)
            
            smpl_output = self.smpl(global_orient=global_orient.float(), betas=betas.float(), body_pose=body_pose.float(), pose2rot=False)
            pred_verts = smpl_output.vertices + pred_cam_t.unsqueeze(1)

            map_verts, valid_mask = unproject_uvmap_to_mesh(self.bmap, self.fmap, pred_verts.float(), self.face_tensor)
            map_verts_proj = self.focal * map_verts[:, :, :2] / (map_verts[:, :, 2:3] + 1e-9)
            map_verts_depth = map_verts[:, :, 2]

            rend_depth = self.nmr_renderer(pred_verts.float(), self.face_tensor[None].expand(pred_verts.shape[0], -1, -1).int(), mode='depth', K=self.K.float(), R=self.R.float(), t=self.t.float())
            rend_depth_at_proj = torch.nn.functional.grid_sample(rend_depth[:,None,:,:], map_verts_proj[:,None,:,:], align_corners=False).squeeze(1).squeeze(1)

            img_rgba = torch.cat([img_tf, torch.ones_like(img_tf[:, :1, :, :])], dim=1)
            img_rgba_at_proj = torch.nn.functional.grid_sample(img_rgba, map_verts_proj[:,None,:,:], align_corners=False).squeeze(2)

            visibility_mask = map_verts_depth <= (rend_depth_at_proj + 1e-4)
            img_rgba_at_proj[:,3,:][~visibility_mask] = 0
            
            uv_image = torch.zeros((img_rgba.shape[0], 4, self.img_size, self.img_size), dtype=torch.float, device=self.device)
            uv_image[:, :, valid_mask] = img_rgba_at_proj
            
            show_uvs.append(uv_image[0, :3])
            show_masks.append(uv_image[0, 3:])

        if not show_uvs:
            log.warning(f"No valid frames found to extract texture for {pkl_path}")
            return
        # check if the number is correct
        video_length = len(show_uvs)
        # # generate texturemap mask, in the advanced version, ***** for whole batch, we should load all frames from one batch *****
        tmask_max = 0
        tmask_max_i = 0
        tmask_max_i_stack = []
        stack_num = 10
        tmap_mask = []
        visible_mask = torch.zeros_like(show_masks[0]).to(self.device)
        for tmask_i in range(video_length):
            tmask_tmp = show_masks[tmask_i]
            # need to go though the median filter
            tmask_tmp = cv2.medianBlur(tmask_tmp.repeat(3,1,1).permute(1, 2, 0).cpu().numpy(), ksize=5)
            tmask_tmp[tmask_tmp>0] = 1
            # if tmask_tmp.sum() > tmask_max:
            #     tmask_max = tmask_tmp.sum()
            #     tmask_max_i = tmask_i
            if len(tmask_max_i_stack) < stack_num or tmask_tmp.sum() > tmask_max_i_stack[0][1]:
                tmask_max = tmask_tmp.sum()
                if len(tmask_max_i_stack) < stack_num:
                    tmask_max_i_stack.append([tmask_i, tmask_max])
                else:
                    tmask_max_i_stack.pop(0)
                    tmask_max_i_stack.append([tmask_i, tmask_max])

                tmask_max_i_stack = sorted(tmask_max_i_stack, key=lambda x: x[1])
                tmask_max_i = tmask_max_i_stack[-1][0]

            # N * h * w
            tmask_tmp = torch.tensor(tmask_tmp).to(self.device)[:,:,0].squeeze(-1).unsqueeze(0)
            tmap_mask.append(tmask_tmp)
            visible_mask = visible_mask + tmask_tmp
            
        visible_mask[visible_mask>1] = 1
        
        show_uvs = torch.stack(show_uvs, dim=0)
        tmap_mask = torch.stack(tmap_mask, dim=0)
        
        for [tmask_max_i, _] in tmask_max_i_stack:
            show_uvs = torch.cat([show_uvs, show_uvs[tmask_max_i].unsqueeze(0).to(self.device)], dim=0)
            tmap_mask = torch.cat([tmap_mask, self.tmask_standard], dim=0)
        
        tmap_full = masked_median(show_uvs, tmap_mask)
        if tmap_full.dim() == 4:
            tmap_full = tmap_full.squeeze(0)

        show_uv = tmap_full * self.std + self.mean
        show_uv_pil = Image.fromarray(show_uv.permute(1, 2, 0).cpu().numpy().astype(np.uint8))
        show_uv_pil.save(save_file_path)
        log.info(f'SAVED TEXTURE: {save_file_path}')

        save_mask_path = tmap_dir / f"mask.pkl"
        with open(save_mask_path, 'wb') as f:
            pickle.dump(visible_mask.cpu(), f)
        log.info(f'SAVED MASK: {save_mask_path}')

def extract_textures(base_path: Path):
    """
    Finds all reformatted pkl files and extracts textures for them.
    """
    log.info(f"Starting texture extraction for .pkl files in: {base_path}")
    
    try:
        texture_extractor = TextureExtractor()
    except FileNotFoundError as e:
        log.error(f"Could not initialize TextureExtractor: {e}")
        log.error("Please make sure bmap_1024.npy and fmap_1024.npy are in the current directory.")
        return

    reformatted_pkls = list(base_path.glob("**/results/eval_*.pkl"))
    if not reformatted_pkls:
        log.warning(f"No 'eval_*.pkl' files found under {base_path}. Nothing to extract textures from.")
        return

    for pkl_path in tqdm(reformatted_pkls, desc="Extracting textures"):
        is_ref = False

        texture_extractor.extract_from_pkl(pkl_path, read_only_first_frame=is_ref)

@hydra.main(version_base="1.2", config_name="config")
def main(cfg: DictConfig) -> Optional[float]:
    """Main function for running the PHALP tracker."""
    phalp_tracker = HMR2_4dhuman(cfg)

    # Process driving videos
    if cfg.driving_videos:
        log.info(f"Processing driving videos from: {cfg.driving_videos}")
        driving_videos_path = Path(cfg.driving_videos)
        if driving_videos_path.is_dir():
            driving_dataset = VideoDataset(driving_videos_path)
            if len(driving_dataset) > 0:
                driving_data_loader = DataLoader(driving_dataset, batch_size=1, shuffle=False, num_workers=0)
                with torch.no_grad():
                    for video_path_list in driving_data_loader:
                        video_path = video_path_list[0] # DataLoader returns a list
                        log.info(f"Tracking driving video: {video_path}")
                        phalp_tracker.cfg.video.source = video_path
                        phalp_tracker.cfg.video.output_dir = os.path.dirname(video_path) # e.g. driving_videos/video1
                        phalp_tracker.track()
                reformat_pkl_files(driving_videos_path)
                extract_textures(driving_videos_path)
            else:
                log.info(f"No videos found by VideoDataset in {driving_videos_path}")
        elif driving_videos_path.is_file() and str(driving_videos_path).endswith(('.mp4', '.avi', '.mov')):
            log.info(f"Tracking single driving video: {driving_videos_path}")
            phalp_tracker.cfg.video.source = str(driving_videos_path)
            phalp_tracker.cfg.video.output_dir = str(driving_videos_path.parent)
            phalp_tracker.track()
            reformat_pkl_files(driving_videos_path.parent)
            extract_textures(driving_videos_path.parent)
        else:
            log.error(f"Invalid path or file type for driving_videos: {cfg.driving_videos}")
    else:
        log.info("No driving_videos path provided.")

    # Process reference images/videos
    if cfg.reference_imgs:
        log.info(f"Processing reference items from: {cfg.reference_imgs}")
        reference_path = Path(cfg.reference_imgs)

        if reference_path.is_dir():
            # Preprocess images in 'reference_path/images/*.png' to videos
            # New videos will be in 'reference_path/image_stem/image_stem.mp4'
            preprocess_reference_images(reference_path, num_frames=5) # 5 frames as requested

            # Now, VideoDataset will look for videos in reference_path/subfolder/video.mp4
            ref_video_dataset = VideoDataset(reference_path)
            if len(ref_video_dataset) > 0:
                ref_data_loader = DataLoader(ref_video_dataset, batch_size=1, shuffle=False, num_workers=0)
                with torch.no_grad():
                    for video_path_list in ref_data_loader:
                        video_path = video_path_list[0]
                        log.info(f"Tracking reference video (from preprocessed image or existing video): {video_path}")
                        phalp_tracker.cfg.video.source = video_path
                        phalp_tracker.cfg.video.output_dir = os.path.dirname(video_path) # e.g.     reference_imgs/your_ref_img_A
                        phalp_tracker.track()
                # Optional: Remove the original images directory after processing
                if (reference_path / "images").exists():
                    try:
                        shutil.rmtree(reference_path / "images")
                        log.info(f"Removed original images directory after processing: {reference_path / 'images'}")
                    except Exception as e:
                        log.warning(f"Failed to remove images directory: {e}")
            # Now reformat the .pkl files for efficiency.
                reformat_pkl_files(reference_path)
                extract_textures(reference_path)
            else:
                log.info(f"No videos found by VideoDataset in {reference_path} (after attempting image preprocessing).")

        elif reference_path.is_file():
            if str(reference_path).endswith(('.mp4', '.avi', '.mov')):
                # Handle single reference video file directly
                log.info(f"Tracking single reference video file: {reference_path}")
                phalp_tracker.cfg.video.source = str(reference_path)
                phalp_tracker.cfg.video.output_dir = str(reference_path.parent)
                phalp_tracker.track()
                reformat_pkl_files(reference_path.parent)
                extract_textures(reference_path.parent)
            elif str(reference_path).endswith(('.png', '.jpg')):
                # Handle single reference image file: convert to a 5-frame video then track
                log.info(f"Processing single reference image file: {reference_path}")
                
                img_stem = reference_path.stem
                # Output directory for the video from single image: reference_path.parent / img_stem / img_stem.mp4
                single_img_video_output_dir = reference_path.parent / img_stem
                os.makedirs(single_img_video_output_dir, exist_ok=True)
                
                img = cv2.imread(str(reference_path))
                if img is None:
                    log.error(f"Could not read image: {reference_path}")
                else:
                    video_name = f"{img_stem}.mp4"
                    output_video_path = single_img_video_output_dir / video_name
                    
                    height, width, layers = img.shape
                    if layers != 3:
                        log.error(f"Single image {reference_path} does not have 3 channels. Skipping.")
                    else:
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        try:
                            video_writer = cv2.VideoWriter(str(output_video_path), fourcc, 1.0, (width, height))
                            if not video_writer.isOpened():
                                log.error(f"Failed to open VideoWriter for single image video {output_video_path}.")
                            else:
                                for _ in range(5): # 5 frames
                                    video_writer.write(img)
                                video_writer.release()
                                log.info(f"Converted single image {reference_path} to video {output_video_path}")

                                phalp_tracker.cfg.video.source = str(output_video_path)
                                phalp_tracker.cfg.video.output_dir = str(single_img_video_output_dir)
                                phalp_tracker.track()
                                reformat_pkl_files(single_img_video_output_dir)
                                extract_textures(single_img_video_output_dir)
                        except Exception as e:
                            log.error(f"Error writing video for single image {reference_path}: {e}")
                            if 'video_writer' in locals() and video_writer.isOpened():
                                video_writer.release()
            else:
                log.error(f"Unsupported file type for single reference file: {reference_path}")
        else:
            log.error(f"Invalid path for reference_imgs: {cfg.reference_imgs}")
    else:
        log.info("No reference_imgs path provided.")

    # Generate reference + motion pairs
    if cfg.driving_videos and cfg.reference_imgs:
        log.info("Generating reference and motion pairs.")
        
        driving_base_path = Path(cfg.driving_videos)
        reference_base_path = Path(cfg.reference_imgs)
        
        motion_paths = []
        if driving_base_path.is_dir():
            for p in driving_base_path.iterdir():
                # A processed motion directory should contain texture.png and mask.pkl
                if p.is_dir() and (p / "texture.png").exists() and (p / "mask.pkl").exists():
                    motion_paths.append(str(p.resolve()))
        elif driving_base_path.is_file():
             # For a single file, the output is in the parent directory, which is the base path.
             if (driving_base_path.parent / "texture.png").exists() and (driving_base_path.parent / "mask.pkl").exists():
                 motion_paths.append(str(driving_base_path.parent.resolve()))


        reference_paths = []
        if reference_base_path.is_dir():
            for p in reference_base_path.iterdir():
                # A processed reference directory should contain texture.png and mask.pkl
                if p.is_dir() and (p / "texture.png").exists() and (p / "mask.pkl").exists():
                    reference_paths.append(str(p.resolve()))
        elif reference_base_path.is_file():
            # For a single image, output is in a subfolder named after the image stem
            ref_subfolder = reference_base_path.parent / reference_base_path.stem
            if ref_subfolder.is_dir() and (ref_subfolder / "texture.png").exists() and (ref_subfolder / "mask.pkl").exists():
                reference_paths.append(str(ref_subfolder.resolve()))

        if not motion_paths:
            log.warning("No processed driving videos found to generate pairs.")
        if not reference_paths:
            log.warning("No processed reference images/videos found to generate pairs.")

        if motion_paths and reference_paths:

            all_pairs = []
            # Generate all pausible pairs --> Num_Ref x Num_motion
            for ref_path, motion_path in itertools.product(reference_paths, motion_paths):
                all_pairs.append({
                    "reference": ref_path,
                    "motion": motion_path
                })
            
            output_json_path = driving_base_path.parent / "processed_datas.json"
            with open(output_json_path, 'w') as f:
                json.dump(all_pairs, f, indent=4)
            
            log.info(f"Successfully generated {len(all_pairs)} pairs.")
            log.info(f"Saved pairs to {output_json_path}")


if __name__ == "__main__":
    main()
