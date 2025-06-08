# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import cv2
from torchvision.ops.boxes import batched_nms, box_area
from typing import Any, Dict, List, Optional, Tuple
from ultralytics import YOLO
from functools import partial
import torch.nn.functional as F

from .modeling import ImageEncoderViT, MaskDecoder, PromptEncoder, Samus, TwoWayTransformer
from .utils.amg import (
    MaskData,
    area_from_rle,
    batch_iterator,
    batched_mask_to_box,
    box_xyxy_to_xywh,
    build_all_layer_point_grids,
    calculate_stability_score,
    coco_encode_rle,
    generate_crop_boxes,
    is_box_near_crop_edge,
    mask_to_rle_pytorch,
    remove_small_regions,
    rle_to_mask,
    uncrop_boxes_xyxy,
    uncrop_masks,
    uncrop_points,
)


def build_samus_vit_h(args, checkpoint=None):
    return _build_samus(
        args,
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        checkpoint=checkpoint,
    )


build_samus = build_samus_vit_h


def build_samus_vit_l(args, checkpoint=None):
    return _build_samus(
        args,
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        checkpoint=checkpoint,
    )


def build_samus_vit_b(args, checkpoint=None):
    return _build_samus(
        args,
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
    )


samus_model_registry = {
    "default": build_samus_vit_h,
    "vit_h": build_samus_vit_h,
    "vit_l": build_samus_vit_l,
    "vit_b": build_samus_vit_b,
}


def _build_samus(
    args,
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    checkpoint=None,
):
    prompt_embed_dim = 256
    image_size = args.encoder_input_size
    patch_size = image_size//32
    image_embedding_size = image_size // patch_size
    samus = Samus(
        image_encoder=ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size= patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
    )
    samus.eval()
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
        try:
            samus.load_state_dict(state_dict)
        except:
            new_state_dict = load_from2(samus, state_dict, image_size, patch_size)
            samus.load_state_dict(new_state_dict)
    return samus


def load_from(samus, sam_dict, image_size, patch_size):
    samus_dict = samus.state_dict()
    dict_trained = {k: v for k, v in sam_dict.items() if k in samus_dict}
    rel_pos_keys = [k for k in dict_trained.keys() if 'rel_pos' in k]
    global_rel_pos_keys = [k for k in rel_pos_keys if '2' in k or '5' in  k or '8' in k or '11' in k]
    token_size = int(image_size//patch_size)
    for k in global_rel_pos_keys:
        rel_pos_params = dict_trained[k]
        h, w = rel_pos_params.shape
        rel_pos_params = rel_pos_params.unsqueeze(0).unsqueeze(0)
        rel_pos_params = F.interpolate(rel_pos_params, (token_size * 2 - 1, w), mode='bilinear', align_corners=False)
        dict_trained[k] = rel_pos_params[0, 0, ...]
    samus_dict.update(dict_trained)
    return samus_dict


def load_from2(samus, sam_dict, image_size, patch_size): # load the positional embedding
    samus_dict = samus.state_dict()
    dict_trained = {k: v for k, v in sam_dict.items() if k in samus_dict}
    token_size = int(image_size//patch_size)
    rel_pos_keys = [k for k in dict_trained.keys() if 'rel_pos' in k]
    global_rel_pos_keys = [k for k in rel_pos_keys if '2' in k or '5' in  k or '8' in k or '11' in k]
    for k in global_rel_pos_keys:
        rel_pos_params = dict_trained[k]
        h, w = rel_pos_params.shape
        rel_pos_params = rel_pos_params.unsqueeze(0).unsqueeze(0)
        rel_pos_params = F.interpolate(rel_pos_params, (token_size * 2 - 1, w), mode='bilinear', align_corners=False)
        dict_trained[k] = rel_pos_params[0, 0, ...]
    samus_dict.update(dict_trained)
    return samus_dict


class ModelArgs:
    """Configuration arguments for SAMUS model"""
    def __init__(self, encoder_input_size=1024):
        self.encoder_input_size = encoder_input_size


class YOLOSamAutomaticMaskGenerator:
    def __init__(
        self,
        samus_model_type: str = "vit_h",
        samus_checkpoint: Optional[str] = None,
        encoder_input_size: int = 1024,
        yolo_model_path: str = "yolov8n.pt",
        points_per_detection: int = 5,  # center + 4 surrounding points
        points_per_batch: int = 64,
        pred_iou_thresh: float = 0.88,
        stability_score_thresh: float = 0.95,
        stability_score_offset: float = 1.0,
        box_nms_thresh: float = 0.7,
        yolo_conf_thresh: float = 0.3,
        yolo_iou_thresh: float = 0.5,
        white_region_thresh: int = 200,  # Threshold for white regions (0-255)
        surrounding_point_offset: float = 0.3,  # How far surrounding points are from center (0-1)
        min_mask_region_area: int = 0,
        output_mode: str = "binary_mask",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        """
        Using YOLO detections + SAMUS model, generates masks for detected objects.
        
        Arguments:
          samus_model_type (str): Type of SAMUS model ('vit_h', 'vit_l', 'vit_b').
          samus_checkpoint (str): Path to SAMUS model checkpoint.
          encoder_input_size (int): Input size for the image encoder.
          yolo_model_path (str): Path to YOLO model weights.
          points_per_detection (int): Number of points to generate per YOLO detection.
          yolo_conf_thresh (float): Confidence threshold for YOLO detections.
          yolo_iou_thresh (float): IOU threshold for YOLO NMS.
          white_region_thresh (int): Threshold to identify white regions (0-255).
          surrounding_point_offset (float): Distance of surrounding points from center.
          device (str): Device to run models on.
          Other parameters same as original SAM.
        """
        
        self.device = device
        
        # Initialize YOLO model
        self.yolo_model = YOLO(yolo_model_path)
        print(f"Loaded YOLO model from {yolo_model_path}")
        
        # Initialize SAMUS model
        args = ModelArgs(encoder_input_size=encoder_input_size)
        if samus_model_type not in samus_model_registry:
            raise ValueError(f"Unknown SAMUS model type: {samus_model_type}")
        
        build_fn = samus_model_registry[samus_model_type]
        self.samus_model = build_fn(args, checkpoint=samus_checkpoint)
        self.samus_model.to(device)
        print(f"Loaded SAMUS model ({samus_model_type}) on {device}")
        
        # Create SAM predictor wrapper
        from .predictor import SamPredictor  # Assuming you have this
        self.predictor = SamPredictor(self.samus_model)
        
        # YOLO parameters
        self.yolo_conf_thresh = yolo_conf_thresh
        self.yolo_iou_thresh = yolo_iou_thresh
        
        # Point generation parameters
        self.points_per_detection = points_per_detection
        self.points_per_batch = points_per_batch
        self.white_region_thresh = white_region_thresh
        self.surrounding_point_offset = surrounding_point_offset
        
        # SAM parameters
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self.stability_score_offset = stability_score_offset
        self.box_nms_thresh = box_nms_thresh
        self.min_mask_region_area = min_mask_region_area
        self.output_mode = output_mode
        
        # Validate output mode
        assert output_mode in [
            "binary_mask",
            "uncompressed_rle", 
            "coco_rle",
        ], f"Unknown output_mode {output_mode}."
        
        if output_mode == "coco_rle":
            from pycocotools import mask as mask_utils
            
        if min_mask_region_area > 0:
            import cv2

    def _detect_objects_yolo(self, image: np.ndarray) -> List[Dict]:
        """
        Run YOLO detection on the image.
        
        Returns:
            List of detection dictionaries with 'bbox', 'confidence', 'class'
        """
        results = self.yolo_model(
            image, 
            conf=self.yolo_conf_thresh,
            iou=self.yolo_iou_thresh,
            verbose=False
        )
        
        detections = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                bbox = boxes.xyxy[i].cpu().numpy()  # [x1, y1, x2, y2]
                conf = boxes.conf[i].cpu().numpy()
                cls = int(boxes.cls[i].cpu().numpy())
                
                detections.append({
                    'bbox': bbox,
                    'confidence': float(conf),
                    'class': cls,
                    'class_name': self.yolo_model.names[cls] if hasattr(self.yolo_model, 'names') else str(cls)
                })
        
        return detections

    def _is_point_in_white_region(self, point: Tuple[int, int], image: np.ndarray) -> bool:
        """
        Check if a point is in a white region of the image.
        
        Args:
            point: (x, y) coordinates
            image: Input image in RGB format
            
        Returns:
            True if point is in white region
        """
        x, y = int(point[0]), int(point[1])
        h, w = image.shape[:2]
        
        if x < 0 or x >= w or y < 0 or y >= h:
            return False
            
        # Convert to grayscale for white region detection
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
            
        return gray[y, x] >= self.white_region_thresh

    def _generate_points_from_bbox(self, bbox: np.ndarray, image: np.ndarray) -> List[Tuple[float, float]]:
        """
        Generate point prompts from YOLO bounding box.
        Creates center point + 4 surrounding points in white regions only.
        
        Args:
            bbox: [x1, y1, x2, y2] bounding box coordinates
            image: Input image
            
        Returns:
            List of (x, y) point coordinates
        """
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Always include center point
        points = [(center_x, center_y)]
        
        # Calculate box dimensions for surrounding points
        box_width = x2 - x1
        box_height = y2 - y1
        offset_x = box_width * self.surrounding_point_offset
        offset_y = box_height * self.surrounding_point_offset
        
        # Generate 4 surrounding points (top, bottom, left, right)
        candidate_points = [
            (center_x, center_y - offset_y),  # top
            (center_x, center_y + offset_y),  # bottom
            (center_x - offset_x, center_y),  # left
            (center_x + offset_x, center_y),  # right
        ]
        
        # Only add points that are in white regions and within image bounds
        h, w = image.shape[:2]
        for point in candidate_points:
            x, y = point
            if (0 <= x < w and 0 <= y < h and 
                self._is_point_in_white_region(point, image)):
                points.append(point)
        
        # If we don't have enough points, add more within the bounding box
        attempts = 0
        while len(points) < self.points_per_detection and attempts < 20:
            # Try random points within the bounding box
            rand_x = np.random.uniform(x1, x2)
            rand_y = np.random.uniform(y1, y2)
            candidate = (rand_x, rand_y)
            
            if (self._is_point_in_white_region(candidate, image) and 
                candidate not in points):
                points.append(candidate)
            
            attempts += 1
                
        return points[:self.points_per_detection]

    @torch.no_grad()
    def generate(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Generates masks for YOLO-detected objects in the given image.
        
        Arguments:
          image (np.ndarray): The image to generate masks for, in HWC uint8 format.
          
        Returns:
           list(dict(str, any)): A list over records for masks with YOLO detection info.
        """
        
        # Step 1: Run YOLO detection
        detections = self._detect_objects_yolo(image)
        
        if not detections:
            print("No objects detected by YOLO")
            return []
        
        print(f"Found {len(detections)} YOLO detections")
        
        # Step 2: Generate point prompts for each detection
        all_points = []
        detection_info = []
        
        for det in detections:
            points = self._generate_points_from_bbox(det['bbox'], image)
            if points:
                all_points.extend(points)
                # Store detection info for each point
                for _ in points:
                    detection_info.append(det)
        
        if not all_points:
            print("No valid points generated from detections")
            return []
            
        print(f"Generated {len(all_points)} points from detections")
        
        # Step 3: Generate masks using SAMUS
        mask_data = self._generate_masks_from_points(image, all_points)
        
        # Step 4: Post-process small regions if needed
        if self.min_mask_region_area > 0:
            mask_data = self.postprocess_small_regions(
                mask_data,
                self.min_mask_region_area,
                self.box_nms_thresh,
            )
        
        # Step 5: Encode masks
        if self.output_mode == "coco_rle":
            mask_data["segmentations"] = [coco_encode_rle(rle) for rle in mask_data["rles"]]
        elif self.output_mode == "binary_mask":
            mask_data["segmentations"] = [rle_to_mask(rle) for rle in mask_data["rles"]]
        else:
            mask_data["segmentations"] = mask_data["rles"]
        
        # Step 6: Create output records with YOLO detection info
        curr_anns = []
        for idx in range(len(mask_data["segmentations"])):
            # Find corresponding detection info
            point_idx = min(idx, len(detection_info) - 1)
            det_info = detection_info[point_idx] if detection_info else {}
            
            ann = {
                "segmentation": mask_data["segmentations"][idx],
                "area": area_from_rle(mask_data["rles"][idx]),
                "bbox": box_xyxy_to_xywh(mask_data["boxes"][idx]).tolist(),
                "predicted_iou": mask_data["iou_preds"][idx].item(),
                "point_coords": [mask_data["points"][idx].tolist()],
                "stability_score": mask_data["stability_score"][idx].item(),
                # Add YOLO detection information
                "yolo_bbox": det_info.get('bbox', []).tolist() if hasattr(det_info.get('bbox', []), 'tolist') else det_info.get('bbox', []),
                "yolo_confidence": det_info.get('confidence', 0.0),
                "yolo_class": det_info.get('class', -1),
                "yolo_class_name": det_info.get('class_name', 'unknown'),
            }
            curr_anns.append(ann)
        
        return curr_anns

    def _generate_masks_from_points(self, image: np.ndarray, points: List[Tuple[float, float]]) -> MaskData:
        """
        Generate masks from point prompts using SAMUS.
        """
        # Set image for SAM predictor
        self.predictor.set_image(image)
        
        # Convert points to numpy array
        points_array = np.array(points)
        
        # Generate masks in batches
        data = MaskData()
        
        for (batch_points,) in batch_iterator(self.points_per_batch, points_array):
            batch_data = self._process_point_batch(batch_points, image.shape[:2])
            data.cat(batch_data)
            del batch_data
            
        self.predictor.reset_image()
        
        # Remove duplicates
        if len(data["boxes"]) > 1:
            keep_by_nms = batched_nms(
                data["boxes"].float(),
                data["iou_preds"],
                torch.zeros_like(data["boxes"][:, 0]),  # categories
                iou_threshold=self.box_nms_thresh,
            )
            data.filter(keep_by_nms)
        
        data.to_numpy()
        return data

    def _process_point_batch(self, points: np.ndarray, im_size: Tuple[int, ...]) -> MaskData:
        """
        Process a batch of points to generate masks.
        """
        # Transform points for SAM
        transformed_points = self.predictor.transform.apply_coords(points, im_size)
        in_points = torch.as_tensor(transformed_points, device=self.predictor.device)
        in_labels = torch.ones(in_points.shape[0], dtype=torch.int, device=in_points.device)
        
        # Generate masks
        masks, iou_preds, _ = self.predictor.predict_torch(
            in_points[:, None, :],
            in_labels[:, None], 
            multimask_output=True,
            return_logits=True,
        )
        
        # Store in MaskData
        data = MaskData(
            masks=masks.flatten(0, 1),
            iou_preds=iou_preds.flatten(0, 1),
            points=torch.as_tensor(points.repeat(masks.shape[1], axis=0)),
        )
        del masks
        
        # Filter by predicted IoU
        if self.pred_iou_thresh > 0.0:
            keep_mask = data["iou_preds"] > self.pred_iou_thresh
            data.filter(keep_mask)
        
        # Calculate stability score
        data["stability_score"] = calculate_stability_score(
            data["masks"], self.predictor.model.mask_threshold, self.stability_score_offset
        )
        if self.stability_score_thresh > 0.0:
            keep_mask = data["stability_score"] >= self.stability_score_thresh
            data.filter(keep_mask)
        
        # Threshold masks and calculate boxes
        data["masks"] = data["masks"] > self.predictor.model.mask_threshold
        data["boxes"] = batched_mask_to_box(data["masks"])
        
        # Convert to RLE
        orig_h, orig_w = im_size
        data["rles"] = mask_to_rle_pytorch(data["masks"])
        del data["masks"]
        
        return data

    @staticmethod
    def postprocess_small_regions(mask_data: MaskData, min_area: int, nms_thresh: float) -> MaskData:
        """
        Removes small disconnected regions and holes in masks, then reruns box NMS.
        """
        if len(mask_data["rles"]) == 0:
            return mask_data

        new_masks = []
        scores = []
        for rle in mask_data["rles"]:
            mask = rle_to_mask(rle)
            mask, changed = remove_small_regions(mask, min_area, mode="holes")
            unchanged = not changed
            mask, changed = remove_small_regions(mask, min_area, mode="islands")
            unchanged = unchanged and not changed

            new_masks.append(torch.as_tensor(mask).unsqueeze(0))
            scores.append(float(unchanged))

        # Recalculate boxes and remove new duplicates
        masks = torch.cat(new_masks, dim=0)
        boxes = batched_mask_to_box(masks)
        keep_by_nms = batched_nms(
            boxes.float(),
            torch.as_tensor(scores),
            torch.zeros_like(boxes[:, 0]),
            iou_threshold=nms_thresh,
        )

        # Update RLEs for changed masks
        for i_mask in keep_by_nms:
            if scores[i_mask] == 0.0:
                mask_torch = masks[i_mask].unsqueeze(0)
                mask_data["rles"][i_mask] = mask_to_rle_pytorch(mask_torch)[0]
                mask_data["boxes"][i_mask] = boxes[i_mask]
                
        mask_data.filter(keep_by_nms)
        return mask_data


# Usage example with different SAMUS models
if __name__ == "__main__":
    # Initialize with SAMUS ViT-H (largest model)
    yolo_sam_generator = YOLOSamAutomaticMaskGenerator(
        samus_model_type="vit_h",  # or "vit_l", "vit_b"
        samus_checkpoint="path/to/samus_checkpoint.pth",
        encoder_input_size=1024,  # Can be adjusted based on your needs
        yolo_model_path="yolov8n.pt",
        points_per_detection=5,
        yolo_conf_thresh=0.3,
        white_region_thresh=200,
        surrounding_point_offset=0.3,
        device="cuda"  # or "cpu"
    )
    
    # Load and process image
    image = cv2.imread("your_image.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Generate masks
    masks = yolo_sam_generator.generate(image)
    
    print(f"Generated {len(masks)} masks")
    for i, mask in enumerate(masks):
        print(f"Mask {i}: {mask['yolo_class_name']} (class {mask['yolo_class']}) "
              f"confidence {mask['yolo_confidence']:.3f}")