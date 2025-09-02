import os

import cv2
import joblib
import numpy as np
import torch
# from phalp.utils.utils_dataset import process_image, process_mask
from smt.utils.utils_dataset import process_image, process_mask
from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path
import json

from smt.utils import get_pylogger
from pycocotools import mask as mask_utils
# import torchvision.transforms as transforms
import random
import shutil
import pickle
import torch.nn.functional as F

log = get_pylogger(__name__)

MAX_CACHE_SIZE = 1000
_MEAN = [0.5, 0.5, 0.5]
_STD = [0.5, 0.5, 0.5]
IMG_SIZE = 256*3
VIDEO_LENGTH = 16
NUM_REPEAT = 1
NUM_CLIPS = 45

def find_indices(tensor, C):
    B, _, _ = tensor.shape
    indices = torch.where(tensor == 1)[0]
    num_ones = len(indices)
    
    # Handle edge cases
    if num_ones == 0:
        return None
    elif num_ones == 1:
        return (indices[0], indices[0])
    elif num_ones < C:
        return (indices[0], indices[-1])
    else:
        # we know there are at least C ones
    
        # 1. Find the indices of the C largest values
        possible_start_indices = indices - (indices[-1] - (C -1))
        start_indices = torch.where(possible_start_indices <= 0)[0] 

        # 2. randomly select one of the start indices
        start_index_ = np.random.choice(start_indices)
        start_index  = indices[start_index_]

        # 3. Find the end index
        possible_end_indices = indices - (start_index + (C - 1)) 
        end_indices = torch.where(possible_end_indices >= 0)[0]
        end_index_  = end_indices[0] # this will make sure our distance is C
        # end_index_  = np.random.choice(end_indices) # this will make sure our distance is at least C
        end_index = indices[end_index_]

        return (start_index, end_index)

def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor".format(
            type(ndarray)))
    return ndarray

def pil_loader(path: str) -> Image.Image:
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")
    
def crop_and_resize(image, crop_coordinates, new_size):
    """
    Crop and resize an image.

    Parameters:
        image (numpy.ndarray): The input image to be cropped and resized.
        crop_coordinates (tuple): A tuple of (x_start, y_start, x_end, y_end) defining the cropping region.
        new_size (tuple): A tuple of (new_width, new_height) defining the size to resize the image to.

    Returns:
        numpy.ndarray: The cropped and resized image.
    """
    x_start, y_start, x_end, y_end = crop_coordinates
    
    new_width, new_height = new_size
    # Crop the image
    
    cropped_image = image[y_start:y_end, x_start:x_end]

    # Resize the image
    resized_image = cv2.resize(cropped_image, (new_width, new_height))

    return resized_image   
     
class PHALP_action_dataset_fast(Dataset):
    def __init__(self, opt, train=True):
        
        self.opt              = opt
        self.data             = []
        self.track2video      = []
        self.frame_length     = self.opt.frame_length
        self.max_tokens       = self.opt.max_people
        self.train            = False

        # temp arguments
        self.opt.img_size     = IMG_SIZE
        self.pixel_mean_      = np.array([_MEAN[0] * 255, _MEAN[1] * 255, _MEAN[2] * 255]).reshape(3,1,1)
        self.pixel_std_       = np.array([_STD[0] * 255, _STD[1] * 255, _STD[2] * 255]).reshape(3,1,1)

        self.count    = 0
        self.count_f  = 0
        self.all_index = []
        self.ref_paths = []

        with open(self.opt.metadata_path) as f:
            self.list_of_datasets = json.load(f)
        for dict in self.list_of_datasets:
            reference_path, motion_path = dict["reference"], dict["motion"]
            self.get_dataset(root_dir=motion_path,
                            min_track_length=1)
            self.ref_paths.append(os.path.join(reference_path, "results", "eval_{}_1_5.pkl".format(os.path.basename(reference_path))))
            
        log.info("Number of tracks: {}".format(len(self.data)))
        
        self.pose_key = "pose_shape"
        self.ego_id = 0


    def __len__(self):
        return self.count_f

    def get_dataset(self, root_dir="", num_sample=1, min_track_length=VIDEO_LENGTH, total_num_tracks=None):
        list_of_files = os.listdir(os.path.join(root_dir, "results"))

        for i_, video_ in enumerate(list_of_files):
            if video_.endswith(".pkl"):
                if(int(video_.split("_")[-1][:-4])>min_track_length): # Example: 2k2k_00654_1_120.pkl --> 2k2k_00654
                    for _ in range(num_sample):
                        self.data.append(os.path.join(root_dir, "results", video_)) # Example: 2k2k_00654_1_120.pkl
                    self.count += 1
                    
                    max_end = int(video_.split("_")[-1][:-4])
                    min_start = int(video_.split("_")[-2])
                    video_length = max_end - min_start + 1 # Example: 120
                    
                    interval = 1
                    self.count_f += video_length // interval
                    start_index = 0
                    for frame_it in range(0, video_length, interval):
                        if frame_it == video_length - 1:
                            self.all_index.append([self.count - 1, frame_it, frame_it - 1, start_index])
                        else:
                            self.all_index.append([self.count - 1, frame_it, frame_it + 1, start_index])

                if(total_num_tracks is not None):
                    if(self.count>=total_num_tracks):
                        break

        log.info("Total number of tracks: {}".format(self.count))
        log.info("Total number of frames: {}".format(self.count_f))
    
    def read_from_phalp_fast_all(self, idx):
        idx = self.all_index[idx][0]
        try:
            detection_data   = joblib.load(self.data[idx])
            reference_data   = joblib.load(self.ref_paths[self.all_index[idx][0]])
            
            video_name = self.data[idx].split("/")[-1][:-4]
            start_frame, end_frame = int(video_name.split("_")[-2]) - 1, self.all_index[idx][1]

            log.info(f"READING from .pkl file: {video_name}")
            has_detection = detection_data["has_detection"]

            if has_detection[start_frame][0, 0] == 0:
                for has_i in range(len(has_detection)):
                    if has_detection[has_i][0, 0] == 1:
                        start_frame = has_i
                        break
            if has_detection[end_frame][0, 0] == 0:
                end_frame = start_frame
        except:
            log.warning("Failed to read the .pkl files.")

        list_of_frames   = list(range(len(detection_data["frame_name"])))
        # Hack the first frame as the reference.
        for key in detection_data.keys():
            detection_data[key][start_frame] = reference_data[key][start_frame]
        return detection_data, start_frame, end_frame, list_of_frames, self.data[idx]

    def read_image(self, img_path, bbox, has_detection):
        if(has_detection>0):
            if(os.path.exists(img_path)):
                img_ = cv2.imread(img_path)
            else:
                log.warning(f"PATH: {img_path} no found!")
                exit(0)

            x1 = int(bbox[0])
            y1 = int(bbox[1])
            x2 = int(bbox[2]) + x1
            y2 = int(bbox[3]) + y1
            
            center_ = np.array([(x1 + x2)/2, (y1 + y2)/2])
            scale_  = np.array([(x2 - x1), (y2 - y1)])
            image_tmp = process_image(img_, center_, 1.0*np.max(scale_), output_size=IMG_SIZE)
        else:
            return torch.zeros(3, IMG_SIZE, IMG_SIZE)
        
        return image_tmp
    def read_image_center_padding_resize(self, img_path, bbox, has_detection):
        if(has_detection>0):
            if(os.path.exists(img_path)):
                img_ = cv2.imread(img_path)
            else:
                log.warning(f"PATH: {img_path} not found!")
                return torch.zeros(3, IMG_SIZE, IMG_SIZE), {}

            original_height, original_width, _ = img_.shape
            
            # Determine padding
            if original_width > original_height:
                pad_top = (original_width - original_height) // 2
                pad_bottom = original_width - original_height - pad_top
                pad_left = 0
                pad_right = 0
                padded_size = original_width
            else:
                pad_left = (original_height - original_width) // 2
                pad_right = original_height - original_width - pad_left
                pad_top = 0
                pad_bottom = 0
                padded_size = original_height
            
            # Apply padding
            padded_img = cv2.copyMakeBorder(img_, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            
            # Resize to IMG_SIZE
            resized_image = cv2.resize(padded_img, (IMG_SIZE, IMG_SIZE))

            # Convert to tensor
            img_n = resized_image.astype(np.float32) / 255.0
            image_tmp = torch.from_numpy(np.transpose(img_n, (2, 0, 1)))
            
            unpad_info = {
                'original_height': original_height,
                'original_width': original_width,
                'pad_top': pad_top,
                'pad_bottom': pad_bottom,
                'pad_left': pad_left,
                'pad_right': pad_right,
                'padded_size': padded_size
            }
        else:
            return torch.zeros(3, IMG_SIZE, IMG_SIZE), {}
        
        return image_tmp, unpad_info

    
    def read_mask(self, mask, bbox, has_detection, mask_size=IMG_SIZE):
        if(has_detection>0):
            x1 = int(bbox[0])
            y1 = int(bbox[1])
            x2 = int(bbox[2]) + x1
            y2 = int(bbox[3]) + y1
            
            center_ = np.array([(x1 + x2)/2, (y1+y2)/2])
            scale_  = np.array([(x2 - x1), (y2 - y1)])
            mask_tmp = process_mask(mask, center_, 1.0*np.max(scale_), output_size=mask_size)
        else:
            return torch.zeros(1, IMG_SIZE, IMG_SIZE)
        
        return mask_tmp

    def unnormalize(self, img):
        img = img * torch.tensor([_STD]) + torch.tensor([_MEAN])
        img = np.torch(img, 0, 1)
        return img

    def __getitem__(self, idx):
        detection_data, start_frame, end_frame, list_of_frames, motion_path = self.read_from_phalp_fast_all(idx)
        video_name = motion_path.split("/")[-1][:-4]
        # /video_name/results/.pkl
        dataset_root = Path(motion_path).parent.parent
        frame_paths = [str((Path(motion_path).parent / Path(p)).resolve()) for p in detection_data["frame_name"]]
        frame_bbox = detection_data["frame_bbox"]
        has_detection = detection_data["has_detection"]
        frame_t = end_frame
        frame_t_f = start_frame
        
        # Center padding to square in default.
        img_t, unpad_info_t = self.read_image_center_padding_resize(frame_paths[frame_t], detection_data["frame_bbox"][frame_t], has_detection[frame_t])

        
        img_tf, unpad_info_tf = self.read_image_center_padding_resize(frame_paths[frame_t_f], detection_data["frame_bbox"][frame_t_f], has_detection[frame_t_f])
        # # If want to use affine transform
        
        pose_shape_t = np.array(detection_data["pose_shape"][frame_t]).reshape(1, -1)
        pose_shape_tf = np.array(detection_data["pose_shape"][frame_t_f]).reshape(1, -1)
        
        try:
            smpl_t_g = detection_data["smpl"][frame_t]['global_orient']
            smpl_t_p = detection_data["smpl"][frame_t]['body_pose']
            smpl_t_b = detection_data["smpl"][frame_t]['betas']
        except:
            smpl_t_g = np.zeros((1,3,3))
            smpl_t_p = np.zeros((23,3,3))
            smpl_t_b = np.zeros((10))
        
        try:  
            smpl_tf_g = detection_data["smpl"][frame_t_f]['global_orient']
            smpl_tf_p = detection_data["smpl"][frame_t_f]['body_pose']
            smpl_tf_b = detection_data["smpl"][frame_t_f]['betas']
        except:
            smpl_tf_g = np.zeros((1,3,3))
            smpl_tf_p = np.zeros((23,3,3))
            smpl_tf_b = np.zeros((10))
        
        camera_t = np.array(detection_data["camera_bbox"][frame_t])
        camera_tf = np.array(detection_data["camera_bbox"][frame_t_f])

        tmap_path = dataset_root / "texture.png"
        if not os.path.exists(tmap_path):
            log.warning("Texture map not found: {}".format(tmap_path))
        tmask = ''

        tmap = cv2.imread(tmap_path)
        tmap = process_image(tmap, center=None, scale=None, usetmap=True, output_size=IMG_SIZE)

        out = {
                "img_t" : img_t,
                "img_tf" : img_tf,
                "ori_img_shape": np.array(cv2.imread(frame_paths[frame_t]).shape),
                "path_t" : frame_paths[frame_t],
                "path_tf" : frame_paths[frame_t_f],
                "has_detection_t" : detection_data["has_detection"][frame_t].astype(np.float32),
                "has_detection_tf" : detection_data["has_detection"][frame_t_f].astype(np.float32),
                "pose_shape_t" : pose_shape_t,
                "pose_shape_tf" : pose_shape_tf,
                "smpl_t_g" : smpl_t_g,
                "smpl_tf_g" : smpl_tf_g,
                "smpl_t_p" : smpl_t_p,
                "smpl_tf_p" : smpl_tf_p,
                "smpl_t_b" : smpl_t_b,
                "smpl_tf_b" : smpl_tf_b,
                "camera_t" : camera_t,
                "camera_tf" : camera_tf,
                "time_delta" : frame_t_f - frame_t,
                "tmap_t": tmap,
                "tmask_t": tmask,
                "mask_t": '',
                "mask_tf": '',
                "frame_t": frame_t,
                "frame_bbox_fix": np.array(detection_data["frame_bbox"][frame_t_f], dtype=np.float32),
                "frame_bbox_t": np.array(frame_bbox[frame_t], dtype=np.float32),
                "unpad_info_t": unpad_info_t,
                "unpad_info_tf": unpad_info_tf
            }
        return out

class PHALP_action_dataset_fast_video(PHALP_action_dataset_fast):
    def get_dataset(self, root_dir="", num_sample=1, min_track_length=VIDEO_LENGTH, total_num_tracks=None):
        list_of_files = os.listdir(os.path.join(root_dir, "results"))
                    
        for i_, video_ in enumerate(list_of_files):
            if video_.endswith(".pkl"):
                if(int(video_.split("_")[-1][:-4])>min_track_length): # Example: 2k2k_00654_1_120.pkl --> 2k2k_00654
                    for _ in range(num_sample):
                        self.data.append(os.path.join(root_dir, "results", video_)) # Example: 2k2k_00654_1_120.pkl
                    self.count += 1
                    
                    max_end = int(video_.split("_")[-1][:-4])
                    min_start = int(video_.split("_")[-2])
                    video_length = max_end - min_start + 1 # Example: 120
                    video_length = min(video_length, VIDEO_LENGTH*NUM_CLIPS*NUM_REPEAT)
                    valid_start = min_start - 1
                    valid_end = valid_start + video_length
                    for start in range(valid_start, valid_end, VIDEO_LENGTH):
                        end = start + VIDEO_LENGTH//NUM_REPEAT
                        if end > valid_end:
                            break
                        self.all_index.append([self.count - 1, start, end, valid_start])
                        self.count_f += NUM_REPEAT

                if(total_num_tracks is not None):
                    if(self.count>=total_num_tracks):
                        break

        log.info("Total number of tracks: {}".format(self.count))
        log.info("Total number of Video Clips: {}".format(self.count_f))
    
    def read_from_phalp_fast_all_video(self, idx):
        idx = self.all_index[idx][0]
        try:            
            log.info(f"Read Motion from .pkl file: {self.data[idx]}")
            log.info(f"Read Reference from .pkl file: {self.ref_paths[idx]}")

            detection_data   = joblib.load(self.data[idx])
            reference_data   = joblib.load(self.ref_paths[idx])
        except:
            log.warning("Failed to read the .pkl files.")

        list_of_frames   = list(range(len(detection_data["frame_name"])))
        ref_frame = self.all_index[idx][3]
        for key in detection_data.keys():
            detection_data[key][ref_frame] = reference_data[key][ref_frame]
        return detection_data, list_of_frames, self.data[idx], self.ref_paths[idx]

    def __len__(self):
        return self.count_f
        
    def __getitem__(self, idx):
        detection_data, list_of_frames, motion_path, ref_path = self.read_from_phalp_fast_all_video(idx)
        
        frame_paths = detection_data["frame_name"]

        dataset_root = Path(motion_path).parent.parent
        frame_paths = [str((Path(motion_path).parent / Path(p)).resolve()) for p in detection_data["frame_name"]]
        frame_paths[0] = str((Path(ref_path).parent / Path(detection_data["frame_name"][0])).resolve())
        frame_bbox = detection_data["frame_bbox"]
        has_detection = detection_data["has_detection"]
        video_name = motion_path.split("/")[-1][:-4]

        start_frame, end_frame, ref_frame = self.all_index[idx][1], self.all_index[idx][2], self.all_index[idx][3]# Example: trainRealV0_00011_1_239

        if np.any(has_detection[start_frame:end_frame] == 0):
            for has_i in range(len(has_detection)):
                if has_detection[has_i][0, 0] == 1:
                    end_frame += has_i - start_frame
                    start_frame = has_i
                    if np.any(has_detection[start_frame:end_frame] == 0):
                        continue
                    else:
                        break
        if len(has_detection) <= max(end_frame,ref_frame):
            return self.__getitem__(random.randint(0, len(self.data) - 1))
        if has_detection[ref_frame][0, 0] == 0:
                ref_frame = start_frame                

 
        frame_t_f = ref_frame
        out_clips = []
        indices_repeated = list(range(start_frame, end_frame))
        indices_repeated = [x for x in indices_repeated for _ in range(NUM_REPEAT)]
        for frame_t in indices_repeated:
            if not os.path.exists(frame_paths[frame_t]):
                frame_paths[frame_t] = os.path.splitext(frame_paths[frame_t])[0] + ".jpg"
            if not os.path.exists(frame_paths[frame_t_f]):
                frame_paths[frame_t_f] = os.path.splitext(frame_paths[frame_t_f])[0] + ".jpg"

            img_t, unpad_info_t = self.read_image_center_padding_resize(frame_paths[frame_t], detection_data["frame_bbox"][frame_t], has_detection[frame_t])

            img_tf, unpad_info_tf = self.read_image_center_padding_resize(frame_paths[frame_t_f], detection_data["frame_bbox"][frame_t_f], has_detection[frame_t_f])
            # # If want to use affine transform
            
            pose_shape_t = np.array(detection_data["pose_shape"][frame_t]).reshape(1, -1)
            pose_shape_tf = np.array(detection_data["pose_shape"][frame_t_f]).reshape(1, -1)
            
            try:
                smpl_t_g = detection_data["smpl"][frame_t]['global_orient']
                smpl_t_p = detection_data["smpl"][frame_t]['body_pose']
                smpl_t_b = detection_data["smpl"][frame_t]['betas']
            except:
                smpl_t_g = np.zeros((1,3,3))
                smpl_t_p = np.zeros((23,3,3))
                smpl_t_b = np.zeros((10))
            
            try:  
                smpl_tf_g = detection_data["smpl"][frame_t_f]['global_orient']
                smpl_tf_p = detection_data["smpl"][frame_t_f]['body_pose']
                smpl_tf_b = detection_data["smpl"][frame_t_f]['betas']
            except:
                smpl_tf_g = np.zeros((1,3,3))
                smpl_tf_p = np.zeros((23,3,3))
                smpl_tf_b = np.zeros((10))
            
            camera_t = np.array(detection_data["camera_bbox"][frame_t])
            camera_tf = np.array(detection_data["camera_bbox"][frame_t_f])
            mask_t = ''
            mask_tf = ''
            tmap_path = Path(ref_path).parent.parent / "texture.png"
            if not os.path.exists(tmap_path):
                log.warning("Texture map not found: {}".format(tmap_path))
            tmask = ''

            tmap = cv2.imread(tmap_path)
            tmap = process_image(tmap, center=None, scale=None, usetmap=True, output_size=IMG_SIZE)

            out = {
                    "img_t" : img_t,
                    "img_tf" : img_tf,
                    "ori_img_shape": np.array(cv2.imread(frame_paths[frame_t]).shape),
                    "path_t" : frame_paths[frame_t],
                    "path_tf" : frame_paths[frame_t_f],
                    "has_detection_t" : detection_data["has_detection"][frame_t].astype(np.float32),
                    "has_detection_tf" : detection_data["has_detection"][frame_t_f].astype(np.float32),
                    "pose_shape_t" : pose_shape_t,
                    "pose_shape_tf" : pose_shape_tf,
                    "smpl_t_g" : smpl_t_g,
                    "smpl_tf_g" : smpl_tf_g,
                    "smpl_t_p" : smpl_t_p,
                    "smpl_tf_p" : smpl_tf_p,
                    "smpl_t_b" : smpl_t_b,
                    "smpl_tf_b" : smpl_tf_b,
                    "camera_t" : camera_t,
                    "camera_tf" : camera_tf,
                    "time_delta" : frame_t_f - frame_t,
                    "tmap_t": tmap,
                    "tmask_t": tmask,
                    "mask_t": '',
                    "mask_tf": '',
                    "frame_t": frame_t,
                    "frame_bbox_fix": np.array(detection_data["frame_bbox"][frame_t_f], dtype=np.float32),
                    "frame_bbox_t": np.array(frame_bbox[frame_t], dtype=np.float32),
                    "unpad_info_t": unpad_info_t,
                    "unpad_info_tf": unpad_info_tf

                }
            out_clips.append(out)
        outs = {key: np.array([out[key] for out in out_clips]) for key in out_clips[0]}
        outs["img_t"] = torch.stack([out["img_t"] for out in out_clips], dim=0)
        outs["img_tf"] = torch.stack([out["img_tf"] for out in out_clips], dim=0)
        outs["tmap_t"] = torch.stack([out["tmap_t"] for out in out_clips], dim=0)
        outs["tmask_t"] = [out["tmask_t"] for out in out_clips]
        outs["mask_t"] = [out["mask_t"] for out in out_clips]
        outs["mask_tf"] = [out["mask_tf"] for out in out_clips]
        outs["path_t"] = [out["path_t"] for out in out_clips]
        outs["path_tf"] = [out["path_tf"] for out in out_clips]
        outs["unpad_info_t"] = [out["unpad_info_t"] for out in out_clips]
        outs["unpad_info_tf"] = [out["unpad_info_tf"] for out in out_clips]
        outs["video_name"] = [video_name for out in out_clips]
        return outs
