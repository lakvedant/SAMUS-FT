import torch
import random
import cv2
import numpy as np
from scipy.ndimage import maximum_filter

def generate_click_prompt(img, msk, pt_label=1):
    # return: img, prompt, prompt mask
    pt_list = []
    msk_list = []
    b, c, h, w, d = msk.size()
    msk = msk[:,0,:,:,:]
    for i in range(d):
        pt_list_s = []
        msk_list_s = []
        for j in range(b):
            msk_s = msk[j,:,:,i]
            indices = torch.nonzero(msk_s)
            if indices.size(0) == 0:
                random_index = torch.randint(0, h, (2,)).to(device=msk.device)
                new_s = msk_s
            else:
                random_index = random.choice(indices)
                label = msk_s[random_index[0], random_index[1]]
                new_s = torch.zeros_like(msk_s)
                new_s = (msk_s == label).to(dtype=torch.float)
            pt_list_s.append(random_index)
            msk_list_s.append(new_s)
        pts = torch.stack(pt_list_s, dim=0)
        msks = torch.stack(msk_list_s, dim=0)
        pt_list.append(pts)
        msk_list.append(msks)
    pt = torch.stack(pt_list, dim=-1)
    msk = torch.stack(msk_list, dim=-1)
    msk = msk.unsqueeze(1)
    return img, pt, msk

def get_click_prompt(datapack, opt):
    if 'pt' not in datapack:
        imgs, pt, masks = generate_click_prompt(imgs, masks)
    else:
        pt = datapack['pt']
        point_labels = datapack['p_label']

    point_coords = pt
    coords_torch = torch.as_tensor(point_coords, dtype=torch.float32, device=opt.device)
    labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=opt.device)
    if len(pt.shape) == 2:
        coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
    pt = (coords_torch, labels_torch)
    return pt

def gradcam_to_binary_mask(cam, img_height, img_width, opt):
    """Convert GradCAM to binary mask with morphological postprocessing."""
    # Hyperparameters (configurable via opt)
    threshold = getattr(opt, 'gradcam_threshold', 0.5)
    morph_kernel_size = getattr(opt, 'morph_kernel_size', 3)
    apply_opening = getattr(opt, 'apply_opening', True)
    apply_closing = getattr(opt, 'apply_closing', True)
    
    # Convert CAM to grayscale and normalize
    if len(cam.shape) > 2:
        cam_gray = cv2.cvtColor(cam, cv2.COLOR_RGB2GRAY) if cam.shape[-1] == 3 else cam.squeeze()
    else:
        cam_gray = cam
    cam_normalized = (cam_gray - cam_gray.min()) / (cam_gray.max() - cam_gray.min())
    
    # RESIZE CAM to match original image dimensions
    cam_resized = cv2.resize(cam_normalized, (img_width, img_height), interpolation=cv2.INTER_LINEAR)
    
    # Apply threshold to create binary mask
    binary_mask = (cam_resized > threshold).astype(np.uint8) * 255
    
    # Apply morphological operations (postprocessing)
    if apply_opening or apply_closing:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))
        if apply_opening:
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        if apply_closing:
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    
    return binary_mask

def generate_autoprompt_from_gradcam(imgs, model, gradcam_obj, opt, num_points=1):
    """Generate click prompts from GradCAM using binary mask approach."""
    b, c, h, w = imgs.shape
    all_coords, all_labels = [], []
    
    for batch_idx in range(b):
        try:
            # Create initial seed point (center of image)
            center_x, center_y = w // 2, h // 2
            seed_coords = torch.tensor([[center_x, center_y]], dtype=torch.float32, device=opt.device)
            seed_labels = torch.ones(1, dtype=torch.int, device=opt.device)
            pt_for_gradcam = (seed_coords.unsqueeze(0), seed_labels.unsqueeze(0))
            
            # Generate CAM using gradcam_obj
            cam = gradcam_obj.generate_cam(imgs[batch_idx:batch_idx+1], pt_for_gradcam)
            
            if isinstance(cam, torch.Tensor):
                cam_np = cam.detach().cpu().numpy().squeeze()
            else:
                cam_np = np.array(cam).squeeze()
            
            # Convert GradCAM to binary mask
            binary_mask = gradcam_to_binary_mask(cam_np, h, w, opt)
            
            # Convert binary mask to torch tensor and add batch/channel dimensions for generate_click_prompt
            # binary_mask is (H, W), we need (B, C, H, W, D) format
            binary_mask_torch = torch.from_numpy(binary_mask / 255.0).float().to(opt.device)  # Normalize to 0-1
            binary_mask_torch = binary_mask_torch.unsqueeze(0).unsqueeze(0).unsqueeze(-1)  # (1, 1, H, W, 1)
            
            # Use generate_click_prompt to get points from the binary mask
            img_for_prompt = imgs[batch_idx:batch_idx+1].unsqueeze(-1)  # Add depth dimension
            _, points, _ = generate_click_prompt(img_for_prompt, binary_mask_torch, pt_label=1)
            
            # Extract coordinates from the returned points (points shape: [B, 2, D])
            coords = points[0, :, 0]  # Get first batch, both coordinates, first depth
            coords_list = [[coords[1].item(), coords[0].item()]]  # Convert [y, x] to [x, y] format
            
            # If we need more points, sample more from the mask
            if num_points > 1:
                # Find all non-zero points in the binary mask
                nonzero_y, nonzero_x = np.where(binary_mask > 0)
                if len(nonzero_y) > 1:
                    # Randomly sample additional points
                    additional_points = min(num_points - 1, len(nonzero_y) - 1)
                    random_indices = np.random.choice(len(nonzero_y), additional_points, replace=False)
                    for idx in random_indices:
                        coords_list.append([nonzero_x[idx], nonzero_y[idx]])
                else:
                    # If not enough points, duplicate the first point
                    for _ in range(num_points - 1):
                        coords_list.append(coords_list[0])
            
            coords = coords_list[:num_points]
            
        except Exception as e:
            print(f"GradCAM failed for batch {batch_idx}: {e}")
            coords = [[w // 2, h // 2]] * num_points
        
        coords_tensor = torch.tensor(coords, dtype=torch.float32, device=opt.device)
        labels_tensor = torch.ones(len(coords), dtype=torch.int, device=opt.device)
        
        all_coords.append(coords_tensor)
        all_labels.append(labels_tensor)
    
    return torch.stack(all_coords), torch.stack(all_labels)

def get_autoprompt_click_prompt(datapack, model, gradcam_obj, opt, num_points=1):
    """Main function to get autoprompts from GradCAM using binary mask approach."""
    imgs = datapack['image']
    
    coords_torch, labels_torch = generate_autoprompt_from_gradcam(
        imgs, model, gradcam_obj, opt, num_points
    )
    
    # Format for SAM: ensure proper dimensions
    if len(coords_torch.shape) == 2:
        coords_torch = coords_torch.unsqueeze(1)
        labels_torch = labels_torch.unsqueeze(1)
    
    return (coords_torch, labels_torch)