import torch
import torch.nn.functional as F
import cv2
import numpy as np
import os
from typing import List, Tuple, Optional


class SAMUSGradCAM:
    """Compact GradCAM implementation for SAMUS model with proper 3D handling."""
    
    def __init__(self, model: torch.nn.Module, target_layers: List[str]):
        self.model = model
        self.target_layers = target_layers
        self.gradients = {}
        self.activations = {}
        self.hooks = []
        self._register_hooks()
        
    def _register_hooks(self):
        """Register hooks with proper tuple handling."""
        def get_activation(name):
            def hook(module, input, output):
                self.activations[name] = (output[0] if isinstance(output, tuple) and len(output) > 0 else output).detach() if output is not None else None
            return hook
            
        def get_gradient(name):
            def hook(module, grad_input, grad_output):
                grad = grad_output[0] if isinstance(grad_output, tuple) and len(grad_output) > 0 else grad_output
                self.gradients[name] = grad.detach() if grad is not None else None
            return hook
        
        for name, module in self.model.named_modules():
            if name in self.target_layers:
                self.hooks.extend([
                    module.register_forward_hook(get_activation(name)),
                    module.register_full_backward_hook(get_gradient(name))
                ])
    
    def generate_cam(self, image: torch.Tensor, points: Tuple[torch.Tensor, torch.Tensor]) -> np.ndarray:
        """Generate GradCAM heatmap for 3D SAMUS model."""
        self.model.eval()
        
        try:
            with torch.enable_grad():
                image = image.requires_grad_(True)
                model_output = self.model(image, points)
                
                # Extract prediction masks
                pred_masks = self._extract_pred_masks(model_output)
                if pred_masks is None:
                    raise ValueError("No valid output found in model_output")
                
                if not pred_masks.requires_grad:
                    pred_masks = pred_masks.detach().requires_grad_(True) + 0.0
                
                # Calculate target for backprop
                target_logits = pred_masks.sum(dim=-1).flatten().mean() if pred_masks.dim() == 5 else pred_masks.flatten().mean()
                
                # Backward pass
                self.model.zero_grad()
                if hasattr(image, 'grad') and image.grad is not None:
                    image.grad.zero_()
                target_logits.backward(retain_graph=True)
            
            # Generate and combine CAMs
            cams = self._generate_layer_cams()
            return np.mean(cams, axis=0) if cams else self._get_zero_cam(image)
            
        except Exception as e:
            print(f"GradCAM generation failed: {e}")
            return self._get_zero_cam(image)
    
    # def extract_high_activation_points(self, cam: np.ndarray, num_points: int = 1, 
    #                                intensity_threshold: float = 0.8) -> np.ndarray:
    #     """
    #     Extract points from the darkest red regions (highest activation) of GradCAM.
        
    #     Args:
    #         cam: GradCAM heatmap (2D numpy array)
    #         num_points: Number of points to extract
    #         intensity_threshold: Minimum intensity threshold (0-1) for considering a region
        
    #     Returns:
    #         points: Array of shape (num_points, 2) containing [x, y] coordinates
    #     """
    #     # Normalize CAM to 0-1 range
    #     cam_min, cam_max = cam.min(), cam.max()
    #     if cam_max > cam_min:
    #         cam_norm = (cam - cam_min) / (cam_max - cam_min)
    #     else:
    #         cam_norm = np.zeros_like(cam)
        
    #     # Find regions above threshold
    #     high_activation_mask = cam_norm >= intensity_threshold
        
    #     if not np.any(high_activation_mask):
    #         # If no regions above threshold, use the maximum activation point
    #         max_idx = np.unravel_index(np.argmax(cam_norm), cam_norm.shape)
    #         return np.array([[max_idx[1], max_idx[0]]])  # Return as [x, y]
        
    #     # Get coordinates of high activation regions
    #     y_coords, x_coords = np.where(high_activation_mask)
    #     activation_values = cam_norm[y_coords, x_coords]
        
    #     # Sort by activation strength (descending)
    #     sorted_indices = np.argsort(activation_values)[::-1]
        
    #     points = []
    #     for i in range(min(num_points, len(sorted_indices))):
    #         idx = sorted_indices[i]
    #         # Return as [x, y] format (width, height)
    #         points.append([x_coords[idx], y_coords[idx]])
        
    #     # If we need more points, add some random high-activation points
    #     while len(points) < num_points:
    #         if len(sorted_indices) > 0:
    #             # Add a random point from high activation regions
    #             random_idx = np.random.choice(len(sorted_indices))
    #             idx = sorted_indices[random_idx]
    #             points.append([x_coords[idx], y_coords[idx]])
    #         else:
    #             # Fallback: add center point
    #             h, w = cam.shape
    #             points.append([w//2, h//2])
        
    #     return np.array(points[:num_points])

    # def generate_gradcam_and_extract_points(self, image: torch.Tensor, 
    #                                   initial_points: Tuple[torch.Tensor, torch.Tensor],
    #                                   num_points: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    #     """
    #     Generate GradCAM and extract high-activation points in one step.
        
    #     Args:
    #         image: Input image tensor
    #         initial_points: Initial points for GradCAM generation
    #         num_points: Number of points to extract
        
    #     Returns:
    #         cam: Generated GradCAM heatmap
    #         points: Extracted high-activation points
    #     """
    #     # Generate GradCAM
    #     cam = self.generate_cam(image, initial_points)
        
    #     # Extract points from high activation regions
    #     points = self.extract_high_activation_points(cam, num_points)
        
    #     return cam, points
        
    def _extract_pred_masks(self, model_output):
        """Extract prediction masks from various output formats."""
        if isinstance(model_output, dict):
            for key in ['masks', 'pred_masks', 'low_res_logits']:
                if key in model_output and model_output[key] is not None:
                    return model_output[key]
            return next((v for v in model_output.values() if v is not None), None)
        return model_output[0] if isinstance(model_output, tuple) else model_output
    
    def _generate_layer_cams(self):
        """Generate CAMs for all target layers."""
        cams = []
        for layer_name in self.target_layers:
            if not (layer_name in self.gradients and layer_name in self.activations and
                   self.gradients[layer_name] is not None and self.activations[layer_name] is not None):
                continue
                
            grads, acts = self.gradients[layer_name], self.activations[layer_name]
            if grads.shape != acts.shape:
                continue
            
            # Handle 3D transformer outputs
            if grads.dim() == 3:
                grads, acts = self._reshape_3d_features(grads, acts, layer_name)
                if grads is None:
                    continue
            
            if grads.dim() == 4 and acts.dim() == 4:
                cam = self._compute_cam(grads, acts)
                if cam is not None:
                    cams.append(cam)
        
        return cams
    
    def _reshape_3d_features(self, grads, acts, layer_name):
        """Reshape 3D transformer features to 4D."""
        spatial_size = int(np.sqrt(grads.shape[1]))
        if spatial_size * spatial_size != grads.shape[1]:
            return None, None
        
        grads = grads.reshape(grads.shape[0], spatial_size, spatial_size, grads.shape[2]).permute(0, 3, 1, 2)
        acts = acts.reshape(acts.shape[0], spatial_size, spatial_size, acts.shape[2]).permute(0, 3, 1, 2)
        return grads, acts
    
    def _compute_cam(self, grads, acts):
        """Compute CAM from gradients and activations."""
        weights = torch.mean(grads, dim=(2, 3), keepdim=True)
        cam = F.relu(torch.sum(weights * acts, dim=1, keepdim=True)).squeeze()
        
        if cam.dim() != 2:
            return None
            
        cam_cpu = cam.cpu()
        cam_min, cam_max = float(cam_cpu.min()), float(cam_cpu.max())
        return (cam_cpu - cam_min) / (cam_max - cam_min) if cam_max > cam_min else torch.zeros_like(cam_cpu)
    
    def _get_zero_cam(self, image):
        """Return zero CAM with proper shape."""
        return np.zeros((image.shape[2], image.shape[3]))
    
        
    
    def visualize_cam(self, original_image: np.ndarray, cam: np.ndarray, 
                     points: Optional[np.ndarray] = None, save_path: Optional[str] = None, 
                     alpha: float = 0.4) -> np.ndarray:
        """Create and save visualization."""
        # Handle 3D image - take middle slice
        if original_image.ndim == 4:
            slice_idx = original_image.shape[-1] // 2
            original_image = original_image[:, :, slice_idx] if original_image.shape[-1] > 3 else original_image[:, :, :, slice_idx]
        
        # Prepare image
        if original_image.ndim == 2:
            original_image = np.stack([original_image] * 3, axis=-1)
        
        if original_image.dtype != np.uint8:
            original_image = (original_image * 255).astype(np.uint8) if original_image.max() <= 1.0 else np.clip(original_image, 0, 255).astype(np.uint8)
        
        # Resize CAM
        if cam.shape != original_image.shape[:2]:
            cam = cv2.resize(cam, (original_image.shape[1], original_image.shape[0]))
        
        # Create heatmap
        cam_min, cam_max = float(cam.min()), float(cam.max())
        cam_normalized = ((cam - cam_min) / (cam_max - cam_min) * 255).astype(np.uint8) if cam_max > cam_min and cam_max > 1e-6 else np.zeros_like(cam, dtype=np.uint8)
        heatmap = cv2.applyColorMap(cam_normalized, cv2.COLORMAP_JET)
        
        # Create overlay
        overlay = cv2.addWeighted(original_image, 1-alpha, heatmap, alpha, 0)
        
        # Add points
        if points is not None:
            points = points.reshape(1, -1) if points.ndim == 1 else points
            for i in range(points.shape[0]):
                if points.shape[1] >= 2:
                    x, y = int(points[i, 0]), int(points[i, 1])
                    cv2.circle(overlay, (x, y), 6, (0, 255, 0), -1)
                    cv2.circle(overlay, (x, y), 8, (255, 255, 255), 2)
        
        # Save
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, overlay)
        
        return overlay
    
    
    
    def __del__(self):
        """Clean up hooks."""
        for hook in self.hooks:
            try:
                hook.remove()
            except:
                pass