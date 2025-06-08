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

from .modeling import Samus
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


class YOLOSamAutomaticMaskGenerator:
    def __init__(
        self,
        sam_model: Samus,
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
    ) -> None:
        """
        Using YOLO detections + SAM model, generates masks for detected objects.
        
        Arguments:
          sam_model (Samus): The SAM model to use for mask prediction.
          yolo_model_path (str): Path to YOLO model weights.
          points_per_detection (int): Number of points to generate per YOLO detection.
          yolo_conf_thresh (float): Confidence threshold for YOLO detections.
          yolo_iou_thresh (float): IOU threshold for YOLO NMS.
          white_region_thresh (int): Threshold to identify white regions (0-255).
          surrounding_point_offset (float): Distance of surrounding points from center.
          Other parameters same as original SAM.
        """
        
        # Initialize YOLO model
        self.yolo_model = YOLO(yolo_model_path)
        
        # Initialize SAM predictor
        self.predictor = sam_model
        
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
                    'class': cls
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
        while len(points) < self.points_per_detection:
            # Try random points within the bounding box
            rand_x = np.random.uniform(x1, x2)
            rand_y = np.random.uniform(y1, y2)
            candidate = (rand_x, rand_y)
            
            if (self._is_point_in_white_region(candidate, image) and 
                candidate not in points):
                points.append(candidate)
            
            # Prevent infinite loop
            if len(points) >= 10:  # Safety limit
                break
                
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
        
        # Step 3: Generate masks using SAM
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
            }
            curr_anns.append(ann)
        
        return curr_anns

    def _generate_masks_from_points(self, image: np.ndarray, points: List[Tuple[float, float]]) -> MaskData:
        """
        Generate masks from point prompts using SAM.
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


# Usage example
if __name__ == "__main__":
    # Initialize the generator
    yolo_sam_generator = YOLOSamAutomaticMaskGenerator(
        sam_model=your_sam_model,  # Your initialized SAM model
        yolo_model_path="yolov8n.pt",  # Path to YOLO weights
        points_per_detection=5,
        yolo_conf_thresh=0.3,
        white_region_thresh=200,
        surrounding_point_offset=0.3
    )
    
    # Load and process image
    image = cv2.imread("your_image.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Generate masks
    masks = yolo_sam_generator.generate(image)
    
    print(f"Generated {len(masks)} masks")
    for i, mask in enumerate(masks):
        print(f"Mask {i}: YOLO class {mask['yolo_class']}, confidence {mask['yolo_confidence']:.3f}")