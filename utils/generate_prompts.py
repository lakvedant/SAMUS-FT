import torch
import random
import cv2
import numpy as np
from scipy.ndimage import maximum_filter

def generate_click_prompt(img, msk, pt_label = 1):
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
                # generate a random array between [0-h, 0-h]:
                random_index = torch.randint(0, h, (2,)).to(device = msk.device)
                new_s = msk_s
            else:
                random_index = random.choice(indices)
                label = msk_s[random_index[0], random_index[1]]
                new_s = torch.zeros_like(msk_s)
                # convert bool tensor to int
                new_s = (msk_s == label).to(dtype = torch.float)
                # new_s[msk_s == label] = 1
            pt_list_s.append(random_index)
            msk_list_s.append(new_s)
        pts = torch.stack(pt_list_s, dim=0) # b 2
        msks = torch.stack(msk_list_s, dim=0)
        pt_list.append(pts)  # c b 2
        msk_list.append(msks)
    pt = torch.stack(pt_list, dim=-1) # b 2 d
    msk = torch.stack(msk_list, dim=-1) # b h w d
    msk = msk.unsqueeze(1) # b c h w d
    return img, pt, msk #[b, 2, d], [b, c, h, w, d]

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


def generate_autoprompt_from_gradcam(imgs, model, gradcam_obj, opt, num_points=1):
    """Generate click prompts from GradCAM's highest activation areas."""
    b, c, h, w = imgs.shape
    all_coords, all_labels = [], []
    
    for batch_idx in range(b):
        img_single = imgs[batch_idx:batch_idx+1]
        
        try:
            # Create initial prompt for GradCAM generation
            with torch.no_grad():
                # Use multiple seed points for better coverage
                seed_points = generate_seed_points(h, w, opt.device)
                
                # Generate GradCAM with seed points
                cam = gradcam_obj.generate_cam(img_single, seed_points)
                
                if isinstance(cam, torch.Tensor):
                    cam_np = cam.detach().cpu().numpy().squeeze()
                else:
                    cam_np = np.array(cam).squeeze()
                
                # Extract high-quality points
                coords = extract_peak_points(cam_np, num_points, h, w)
                
        except Exception as e:
            print(f"GradCAM failed for batch {batch_idx}: {e}")
            # Smart fallback: use image-based saliency
            coords = extract_saliency_points(img_single.cpu().numpy(), num_points, h, w)
        
        coords_tensor = torch.tensor(coords, dtype=torch.float32, device=opt.device)
        labels_tensor = torch.ones(len(coords), dtype=torch.int, device=opt.device)
        
        all_coords.append(coords_tensor)
        all_labels.append(labels_tensor)
    
    return torch.stack(all_coords), torch.stack(all_labels)

def generate_seed_points(height, width, device):
    """Generate diverse seed points for initial GradCAM generation."""
    # Use grid-based seed points for better coverage
    grid_size = 3
    coords = []
    for i in range(grid_size):
        for j in range(grid_size):
            x = int(width * (j + 1) / (grid_size + 1))
            y = int(height * (i + 1) / (grid_size + 1))
            coords.append([x, y])
    
    coords_tensor = torch.tensor([coords], dtype=torch.float32, device=device)  # [1, 9, 2]
    labels_tensor = torch.ones(1, len(coords), dtype=torch.int, device=device)  # [1, 9]
    
    return (coords_tensor, labels_tensor)

def extract_peak_points(cam_np, num_points, height, width):
    """Extract points from true activation peaks using advanced peak detection."""
    # Resize and normalize
    if cam_np.shape != (height, width):
        cam_np = cv2.resize(cam_np, (width, height))
    
    cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min() + 1e-8)
    
    # Apply strong smoothing for better peaks
    cam_smooth = cv2.GaussianBlur(cam_np, (9, 9), 2.0)
    
    # Find local maxima using maximum filter
    local_maxima = maximum_filter(cam_smooth, size=max(height, width) // 15)
    peaks = (cam_smooth == local_maxima) & (cam_smooth > np.percentile(cam_smooth, 80))
    
    # Get peak coordinates sorted by intensity
    peak_coords = np.column_stack(np.where(peaks))
    if len(peak_coords) > 0:
        intensities = cam_smooth[peak_coords[:, 0], peak_coords[:, 1]]
        sorted_idx = np.argsort(intensities)[::-1]
        peak_coords = peak_coords[sorted_idx]
        
        coords = []
        for i in range(min(num_points, len(peak_coords))):
            y, x = peak_coords[i]
            coords.append([int(x), int(y)])
        
        # Fill remaining points if needed
        while len(coords) < num_points:
            # Add slightly offset points from best peak
            best_y, best_x = peak_coords[0]
            offset = np.random.randint(-height//20, height//20, 2)
            new_x = np.clip(best_x + offset[0], 0, width-1)
            new_y = np.clip(best_y + offset[1], 0, height-1)
            coords.append([int(new_x), int(new_y)])
            
        return coords[:num_points]
    
    # Fallback to global maximum
    max_idx = np.unravel_index(np.argmax(cam_smooth), cam_smooth.shape)
    return [[int(max_idx[1]), int(max_idx[0])]] * num_points

def extract_saliency_points(img_np, num_points, height, width):
    """Fallback method using image saliency when GradCAM fails."""
    img = img_np.squeeze().transpose(1, 2, 0)  # CHW to HWC
    if img.shape[2] == 3:
        img = (img * 255).astype(np.uint8)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = (img.squeeze() * 255).astype(np.uint8)
    
    # Simple saliency using Laplacian
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    saliency = np.abs(laplacian)
    
    return extract_peak_points(saliency, num_points, height, width)

def get_autoprompt_click_prompt(datapack, model, gradcam_obj, opt, num_points=1):
    """Main function to get autoprompts from GradCAM."""
    imgs = datapack['image']
    coords_torch, labels_torch = generate_autoprompt_from_gradcam(
        imgs, model, gradcam_obj, opt, num_points
    )
    
    # Format for SAM: [b, num_points, 2] and [b, num_points]
    if len(coords_torch.shape) == 2:  # [b, 2] -> [b, 1, 2]
        coords_torch = coords_torch.unsqueeze(1)
        labels_torch = labels_torch.unsqueeze(1)
    
    return (coords_torch, labels_torch)