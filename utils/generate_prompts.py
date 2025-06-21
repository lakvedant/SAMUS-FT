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
    # Hyperparameters 
    threshold = 0.6
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

def generate_autoprompt_from_gradcam(imgs, model, gradcam_obj, opt, num_points=1, save_first_cam=True, image_filename=None, batch_idx=0):
    """Generate click prompts from GradCAM using binary mask approach."""
    b, c, h, w = imgs.shape
    all_coords, all_labels = [], []
    saved_cams = []  # Store CAMs for potential saving
    
    for batch_idx_inner in range(b):
        try:
            # grid of points 
            grid_size = getattr(opt, 'grid_seed_count', 3)  # You can set this in opt
            xs = torch.linspace(0, w - 1, grid_size)
            ys = torch.linspace(0, h - 1, grid_size)
            grid_coords = torch.cartesian_prod(xs, ys).flip(-1)  # flip to (x, y) order
            grid_coords = grid_coords.to(dtype=torch.float32, device=opt.device)
            seed_coords = grid_coords.unsqueeze(0)  # Shape: (1, N, 2)
            seed_labels = torch.ones(grid_coords.size(0), dtype=torch.int, device=opt.device).unsqueeze(0)  # Shape: (1, N)
            pt_for_gradcam = (seed_coords, seed_labels)
            
            # Generate CAM using gradcam_obj - THIS IS THE FIRST CAM
            cam = gradcam_obj.generate_cam(imgs[batch_idx_inner:batch_idx_inner+1], pt_for_gradcam)
            
            # Store the CAM for potential saving
            if isinstance(cam, torch.Tensor):
                cam_np = cam.detach().cpu().numpy().squeeze()
            else:
                cam_np = np.array(cam).squeeze()
            
            saved_cams.append(cam_np)
            
            # SAVE THE FIRST CAM HERE IF REQUESTED
            # if save_first_cam and image_filename is not None:
            #     save_first_gradcam(cam_np, imgs[batch_idx_inner], image_filename, batch_idx_inner, opt, seed_coords[0].cpu().numpy())
            
            # Convert GradCAM to binary mask
            binary_mask = gradcam_to_binary_mask(cam_np, h, w, opt)
            
            # Convert binary mask to torch tensor and add batch/channel dimensions for generate_click_prompt
            # binary_mask is (H, W), we need (B, C, H, W, D) format
            binary_mask_torch = torch.from_numpy(binary_mask / 255.0).float().to(opt.device)  # Normalize to 0-1
            binary_mask_torch = binary_mask_torch.unsqueeze(0).unsqueeze(0).unsqueeze(-1)  # (1, 1, H, W, 1)
            
            # Use generate_click_prompt to get points from the binary mask
            img_for_prompt = imgs[batch_idx_inner:batch_idx_inner+1].unsqueeze(-1)  # Add depth dimension
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
            print(f"GradCAM failed for batch {batch_idx_inner}: {e}")
            coords = [[w // 2, h // 2]] * num_points
            saved_cams.append(None)
        
        coords_tensor = torch.tensor(coords, dtype=torch.float32, device=opt.device)
        labels_tensor = torch.ones(len(coords), dtype=torch.int, device=opt.device)
        
        all_coords.append(coords_tensor)
        all_labels.append(labels_tensor)
    
    return torch.stack(all_coords), torch.stack(all_labels), saved_cams

def save_first_gradcam(cam_np, img_tensor, image_filename, batch_idx, opt, seed_point):
    """Save the first GradCAM (used for prompt generation) to disk."""
    import os
    import cv2
    
    try:
        # Convert image for visualization
        if img_tensor.dim() == 4:  # [C, H, W, D]
            # Take middle slice
            img_np = img_tensor[:, :, :, img_tensor.shape[-1]//2].permute(1, 2, 0).cpu().numpy()
        else:  # [C, H, W]
            img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
        
        # Normalize image
        if img_np.max() <= 1.0:
            img_np = (img_np * 255).astype(np.uint8)
        
        # Create save directory
        gradcam_dir = os.path.join(opt.result_path, f"gradcam-first-{opt.modelname}")
        os.makedirs(gradcam_dir, exist_ok=True)
        
        # Handle filename
        if isinstance(image_filename, (list, tuple)):
            filename = image_filename[batch_idx] if batch_idx < len(image_filename) else image_filename[0]
        else:
            filename = str(image_filename)
        
        base_name = os.path.splitext(filename)[0]
        save_path = os.path.join(gradcam_dir, f"first_gradcam_{base_name}.png")
        
        # Get original image dimensions
        img_height, img_width = img_np.shape[:2]
        
        # Create a simple visualization of the first CAM
        # Convert CAM to grayscale and normalize
        if len(cam_np.shape) > 2:
            cam_gray = cv2.cvtColor(cam_np, cv2.COLOR_RGB2GRAY) if cam_np.shape[-1] == 3 else cam_np.squeeze()
        else:
            cam_gray = cam_np
        cam_normalized = (cam_gray - cam_gray.min()) / (cam_gray.max() - cam_gray.min())
        
        # Resize CAM to match original image dimensions
        cam_resized = cv2.resize(cam_normalized, (img_width, img_height), interpolation=cv2.INTER_LINEAR)
        
        # Create heatmap
        heatmap = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        # Ensure both images have the same shape and data type
        print(f"Debug - img_np shape: {img_np.shape}, dtype: {img_np.dtype}")
        print(f"Debug - heatmap shape: {heatmap.shape}, dtype: {heatmap.dtype}")
        
        # Handle different image formats
        if len(img_np.shape) == 2:  # Grayscale image
            img_rgb = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
        elif len(img_np.shape) == 3 and img_np.shape[2] == 1:  # Single channel
            img_rgb = cv2.cvtColor(img_np.squeeze(-1), cv2.COLOR_GRAY2RGB)
        elif len(img_np.shape) == 3 and img_np.shape[2] == 3:  # RGB
            img_rgb = img_np
        else:
            print(f"Warning: Unexpected image shape {img_np.shape}, converting to RGB")
            if len(img_np.shape) == 3:
                img_rgb = img_np[:, :, :3] if img_np.shape[2] > 3 else img_np
            else:
                img_rgb = np.stack([img_np] * 3, axis=-1)
        
        # Ensure both are uint8
        img_rgb = img_rgb.astype(np.uint8)
        heatmap = heatmap.astype(np.uint8)
        
        # Ensure same dimensions
        if img_rgb.shape[:2] != heatmap.shape[:2]:
            heatmap = cv2.resize(heatmap, (img_rgb.shape[1], img_rgb.shape[0]))
        
        # Ensure both have 3 channels
        if len(heatmap.shape) == 2:
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_GRAY2RGB)
        if len(img_rgb.shape) == 2:
            img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2RGB)
        
        print(f"Debug - Final img_rgb shape: {img_rgb.shape}, dtype: {img_rgb.dtype}")
        print(f"Debug - Final heatmap shape: {heatmap.shape}, dtype: {heatmap.dtype}")
        
        # Create overlay
        overlay = cv2.addWeighted(img_rgb, 0.6, heatmap, 0.4, 0)
        
        # Draw seed point
        # Iterate through all seed points if seed_point is a list/array of points
        for pt in seed_point:
            x, y = int(pt[0]), int(pt[1])
            cv2.circle(overlay, (x, y), 5, (0, 255, 0), -1)
            cv2.circle(overlay, (x, y), 7, (0, 0, 0), 2)

        
        # Save the visualization
        cv2.imwrite(save_path, overlay)
        
        # Also save the binary mask generated from this CAM
        binary_mask = gradcam_to_binary_mask(cam_np, img_height, img_width, opt)
        mask_path = os.path.join(gradcam_dir, f"first_mask_{base_name}.png")
        cv2.imwrite(mask_path, binary_mask)
        
        # Save the raw CAM for debugging
        cam_raw_path = os.path.join(gradcam_dir, f"first_cam_raw_{base_name}.png")
        cv2.imwrite(cam_raw_path, (cam_resized * 255).astype(np.uint8))
        
        print(f"First GradCAM saved to: {save_path}")
        print(f"First binary mask saved to: {mask_path}")
        print(f"Raw CAM saved to: {cam_raw_path}")
        print(f"Seed point used: {seed_point}")
        
    except Exception as e:
        print(f"Failed to save first GradCAM: {e}")
        import traceback
        traceback.print_exc()

def get_autoprompt_click_prompt(datapack, model, gradcam_obj, opt, num_points=1):
    """Main function to get autoprompts from GradCAM using binary mask approach."""
    imgs = datapack['image']
    image_filename = datapack.get('image_name', None)
    
    # Enable saving of first CAM if visualization is enabled
    save_first_cam = hasattr(opt, 'gradcam_visualization') and opt.gradcam_visualization
    
    coords_torch, labels_torch, saved_cams = generate_autoprompt_from_gradcam(
        imgs, model, gradcam_obj, opt, num_points, 
        save_first_cam=save_first_cam, 
        image_filename=image_filename
    )
    
    # Format for SAM: ensure proper dimensions
    if len(coords_torch.shape) == 2:
        coords_torch = coords_torch.unsqueeze(1)
        labels_torch = labels_torch.unsqueeze(1)
    
    return (coords_torch, labels_torch)