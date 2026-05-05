"""
Temporal Video Dataset for Phase 1 Training
Samples frame pairs (f_t, f_{t+s}) from video sequences
TODO: move this file to dataset folder
"""

import os
import random
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

from src.core import register
from ...data._misc import convert_to_tv_tensor
from pycocotools.coco import COCO

@register()
class ViratTemporalDataset(Dataset):
    """
    VIRAT dataset for temporal RT-DETR Phase 1 training
    Loads frame pairs for key/non-key frame training
    Returns pairs as: (image_key, target_key, image_non_key, target_non_key)
    """
    __inject__ = ['transforms', ]

    def __init__(
        self,
        root_dir: str,
        ann_file: str,
        transforms=None,
        max_frame_gap: int = 10,
        pair_sampling_strategy: str = "random",
        frame_stride: int = 1,
    ):
        """
        Args:
            root_dir: Root directory of VIRAT dataset
            ann_file: Path to COCO-format annotation file
            transforms: Image transformations (can be dict or callable)
            max_frame_gap: Maximum frame gap 's' for sampling (1 to max_frame_gap)
            pair_sampling_strategy: Strategy for sampling frame pairs:
                - "all": Sample all possible gaps (1 to max_frame_gap)
                - "random": Sample ONE random gap per frame
                - "fixed_gap": Use only max_frame_gap as the gap
            frame_stride: Sample key frames every N frames
        """
        self.root_dir = Path(root_dir)
        self.transforms = transforms
        self.max_frame_gap = max_frame_gap
        self.pair_sampling_strategy = pair_sampling_strategy
        self.frame_stride = frame_stride

        self.coco = COCO(ann_file)
        
        # Use COCO's internal indexing to save memory
        self.img_id_to_info = self.coco.imgs
        self.img_id_to_anns = self.coco.imgToAnns
        
        # Build video-frame mapping
        self.video_frames = self._build_video_frame_mapping()
        
        self.samples = self._build_sample_pairs()
        
        print(f"Loaded {len(self.samples)} 'frame pairs from VIRAT dataset")
        print(f"  Max frame gap: {self.max_frame_gap}")
        print(f"  Sampling strategy: {self.pair_sampling_strategy}")
        print(f"  Frame stride: {self.frame_stride}")
        print(f"  Transforms: {type(self.transforms).__name__ if self.transforms else 'Default (resize to 640x640)'}")
    
    def _build_video_frame_mapping(self) -> Dict[str, List[Dict]]:
        """Build mapping from video_id to sorted list of frames"""
        video_frames = {}
        
        for img_info in self.coco.dataset['images']:
            video_id = self._extract_video_id(img_info['file_name'])
            
            if video_id not in video_frames:
                video_frames[video_id] = []
            
            video_frames[video_id].append({
                'id': img_info['id'],
                'file_name': img_info['file_name'],
                'frame_idx': self._extract_frame_idx(img_info['file_name']),
                'width': img_info.get('width', 0),
                'height': img_info.get('height', 0),
            })
        
        # Sort frames by frame index
        for video_id in video_frames:
            video_frames[video_id].sort(key=lambda x: x['frame_idx'])
        
        return video_frames
    
    def _extract_video_id(self, file_name: str) -> str:
        """Extract video ID from filename"""
        parts = Path(file_name).parts
        if len(parts) > 1:
            return parts[0]
        return "default_video"
    
    def _extract_frame_idx(self, file_name: str) -> int:
        """Extract frame index from filename"""
        stem = Path(file_name).stem
        numbers = ''.join(filter(str.isdigit, stem))
        return int(numbers) if numbers else 0
    
    def _build_sample_pairs(self) -> List[Tuple[Dict, Dict]]:
        """Build list of valid frame pairs (f_t, f_{t+s})"""
        strategy = self.pair_sampling_strategy.lower()
        
        if strategy == "all":
            return self._build_pairs_all()
        elif strategy == "random":
            return self._build_pairs_random()
        elif strategy == "fixed_gap":
            return self._build_pairs_fixed_gap()
        else:
            print(f"Warning: Unknown sampling strategy '{strategy}', using 'fixed_gap'")
            return self._build_pairs_fixed_gap()
    
    def _build_pairs_all(self) -> List[Tuple[Dict, Dict]]:
        """
        Strategy: 'all'
        Sample ALL possible gaps from 1 to max_frame_gap, respecting frame_stride
        """
        samples = []
        for _, frames in self.video_frames.items():
            # Use range to respect frame_stride!
            for i in range(0, len(frames), self.frame_stride):
                frame_t = frames[i]
                max_offset = min(self.max_frame_gap + 1, len(frames) - i)
                for s in range(1, max_offset):
                    frame_t_s = frames[i + s]
                    samples.append((frame_t, frame_t_s))
        
        return samples
    
    def _build_pairs_random(self) -> List[Tuple[Dict, Dict]]:
        """
        Strategy: 'random'
        Sample ONE random gap per frame, respecting frame_stride
        """
        samples = []
        for _, frames in self.video_frames.items():
            for i in range(0, len(frames), self.frame_stride):
                frame_t = frames[i]
                max_offset = min(self.max_frame_gap + 1, len(frames) - i)
                if max_offset > 1:
                    s = random.randint(1, max_offset - 1)
                    frame_t_s = frames[i + s]
                    samples.append((frame_t, frame_t_s))
        
        return samples
    
    def _build_pairs_fixed_gap(self) -> List[Tuple[Dict, Dict]]:
        """
        Strategy: 'fixed_gap'
        Sample key frames every 'frame_stride' frames, use fixed gap (max_frame_gap)
        """
        samples = []
        for _, frames in self.video_frames.items():
            # Only use every N-th frame as key frame
            for i in range(0, len(frames), self.frame_stride):
                frame_t = frames[i]
                if i + self.max_frame_gap < len(frames):
                    frame_t_s = frames[i + self.max_frame_gap]
                    samples.append((frame_t, frame_t_s))
        
        return samples
    
    def _load_image(self, img_info: Dict) -> Image.Image:
        """Load image from disk"""
        img_path = self.root_dir / img_info['file_name']
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")
        return Image.open(img_path).convert('RGB')
    
    def _get_annotations(self, filename: str) -> List[Dict]:
        """Get annotations for an image"""
        # anns = []
        # for ann in self.virat_data['annotations']:
        #     if ann['image_id'] == img_id:
        #         anns.append(ann)
        # return anns
        return self.filename_to_anns.get(filename, [])
    
    def _prepare_target(self, anns: List[Dict], img_info: Dict) -> Dict:
        """Prepare target dictionary from annotations"""
        boxes = []
        labels = []
        areas = []
        iscrowd = []
        
        for ann in anns:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id'])
            areas.append(ann.get('area', w * h))
            iscrowd.append(ann.get('iscrowd', 0))
        
        h = img_info.get('height', 480)
        w = img_info.get('width', 640)
        
        target = {
            'boxes': torch.as_tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4)),
            'labels': torch.as_tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64),
            'image_id': torch.tensor([img_info['id']]),
            'area': torch.as_tensor(areas, dtype=torch.float32) if areas else torch.zeros((0,)),
            'iscrowd': torch.as_tensor(iscrowd, dtype=torch.int64) if iscrowd else torch.zeros((0,), dtype=torch.int64),
            'orig_size': torch.as_tensor([w, h]),
            'size': torch.as_tensor([w, h]),
        }

        target['boxes'] = convert_to_tv_tensor(target['boxes'], key='boxes', spatial_size=(h, w))
        
        return target
    
    def _apply_default_transform(self, image: Image.Image, target: Dict) -> Tuple[torch.Tensor, Dict]:
        """Apply default transform: resize to 640x640 and convert to tensor"""
        import torchvision.transforms.functional as F
        
        # Resize image
        image = F.resize(image, [640, 640])
        
        # Update target size
        target['size'] = torch.as_tensor([640, 640])
        
        # Scale boxes
        orig_h, orig_w = target['orig_size']
        scale_x = 640 / orig_w
        scale_y = 640 / orig_h
        
        if len(target['boxes']) > 0:
            boxes = target['boxes'].clone()
            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y
            target['boxes'] = boxes
        
        # Convert to tensor
        image = F.to_tensor(image)
        
        return image, target
    
    def __len__(self) -> int:
        return len(self.samples)

    def _apply_shared_pair_transforms(
        self,
        img_key,
        target_key,
        img_non_key,
        target_non_key,
    ):
        # Reuse identical RNG states so key/non-key sample identical transform params.
        torch_state = torch.get_rng_state()
        py_state = random.getstate()
        np_state = np.random.get_state()
        global_samples = getattr(self.transforms, 'global_samples', None)

        transformed_key = self.transforms(img_key, target_key, self)
        img_key, target_key = transformed_key[:2]

        torch.set_rng_state(torch_state)
        random.setstate(py_state)
        np.random.set_state(np_state)
        if global_samples is not None and hasattr(self.transforms, 'global_samples'):
            self.transforms.global_samples = global_samples

        transformed_non_key = self.transforms(img_non_key, target_non_key, self)
        img_non_key, target_non_key = transformed_non_key[:2]
        if global_samples is not None and hasattr(self.transforms, 'global_samples'):
            self.transforms.global_samples = global_samples + 1

        return img_key, target_key, img_non_key, target_non_key
    
    def __getitem__(self, idx: int):
        """Returns: tuple of (image_key, target_key, image_non_key, target_non_key)"""
        frame_t, frame_t_s = self.samples[idx]
        
        # Load key frame and non-key frame
        img_key, img_non_key = self._load_image(frame_t), self._load_image(frame_t_s)
        anns_key, anns_non_key = self.img_id_to_anns.get(frame_t['id'], []), self.img_id_to_anns.get(frame_t_s['id'], [])
        target_key, target_non_key = self._prepare_target(anns_key, frame_t), self._prepare_target(anns_non_key, frame_t_s)
        
        # Apply transforms
        if self.transforms is not None:
            img_key, target_key, img_non_key, target_non_key = self._apply_shared_pair_transforms(
                img_key,
                target_key,
                img_non_key,
                target_non_key,
            )
        else:
            # Apply default transform
            img_key, target_key = self._apply_default_transform(img_key, target_key)
            img_non_key, target_non_key = self._apply_default_transform(img_non_key, target_non_key)
        
        return (img_key, target_key, img_non_key, target_non_key)
     
    def set_epoch(self, epoch):
        """Set epoch for reproducibility"""
        self._epoch = epoch
