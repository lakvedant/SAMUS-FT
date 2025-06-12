from ast import arg
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import argparse
from pickle import FALSE, TRUE
from statistics import mode
from tkinter import image_names
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import time
import random
from utils.config import get_config
from utils.evaluation import get_eval
from importlib import import_module

from torch.nn.modules.loss import CrossEntropyLoss
from monai.losses import DiceCELoss
from einops import rearrange
from models.model_dict import get_model
from utils.data_us import JointTransform2D, ImageToImage2D
from utils.loss_functions.sam_loss import get_criterion
from utils.generate_prompts import get_click_prompt
from thop import profile

# Added imports for visualization
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import cv2
from PIL import Image


def visualize_prompts_and_masks(images, masks, predicted_masks, prompts, sample_names, 
                               save_dir="visualization_results", n_samples=5):
    """
    Visualize input images, ground truth masks, predicted masks, and prompts
    
    Args:
        images: Input images tensor [B, C, H, W]
        masks: Ground truth masks tensor [B, H, W] or [B, 1, H, W]
        predicted_masks: Predicted masks tensor [B, H, W] or [B, 1, H, W]
        prompts: Tuple of (points, labels) where points are [B, N, 2] and labels are [B, N]
        sample_names: List of sample names/filenames
        save_dir: Directory to save visualization results
        n_samples: Number of samples to visualize
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Convert tensors to numpy
    if isinstance(images, torch.Tensor):
        images = images.cpu().numpy()
    if isinstance(masks, torch.Tensor):
        masks = masks.cpu().numpy()
    if isinstance(predicted_masks, torch.Tensor):
        predicted_masks = predicted_masks.cpu().numpy()
        
    # Handle prompt points and labels
    if prompts is not None and len(prompts) == 2:
        prompt_points, prompt_labels = prompts
        if isinstance(prompt_points, torch.Tensor):
            prompt_points = prompt_points.cpu().numpy()
        if isinstance(prompt_labels, torch.Tensor):
            prompt_labels = prompt_labels.cpu().numpy()
    else:
        prompt_points, prompt_labels = None, None
    
    # Normalize images to [0, 1] for visualization
    if images.max() > 1.0:
        images = images / images.max()
    
    # Handle different mask dimensions
    if len(masks.shape) == 4 and masks.shape[1] == 1:
        masks = masks.squeeze(1)
    if len(predicted_masks.shape) == 4 and predicted_masks.shape[1] == 1:
        predicted_masks = predicted_masks.squeeze(1)
    
    # Limit to n_samples
    n_samples = min(n_samples, images.shape[0])
    
    for i in range(n_samples):
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original image
        if images.shape[1] == 1:  # Grayscale
            img_display = images[i, 0]
            axes[0, 0].imshow(img_display, cmap='gray')
        else:  # RGB
            img_display = np.transpose(images[i], (1, 2, 0))
            axes[0, 0].imshow(img_display)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Ground truth mask
        axes[0, 1].imshow(masks[i], cmap='jet', alpha=0.7)
        axes[0, 1].set_title('Ground Truth Mask')
        axes[0, 1].axis('off')
        
        # Predicted mask
        axes[0, 2].imshow(predicted_masks[i], cmap='jet', alpha=0.7)
        axes[0, 2].set_title('Predicted Mask')
        axes[0, 2].axis('off')
        
        # Image with prompts overlay
        if images.shape[1] == 1:  # Grayscale
            axes[1, 0].imshow(img_display, cmap='gray')
        else:  # RGB
            axes[1, 0].imshow(img_display)
            
        # Add prompt points if available
        if prompt_points is not None and prompt_labels is not None:
            points = prompt_points[i]  # [N, 2]
            labels = prompt_labels[i]  # [N]
            
            for j in range(len(points)):
                if labels[j] == 1:  # Positive prompt
                    axes[1, 0].scatter(points[j, 0], points[j, 1], 
                                     c='green', s=100, marker='*', 
                                     edgecolor='white', linewidth=2, label='Positive' if j == 0 else "")
                else:  # Negative prompt
                    axes[1, 0].scatter(points[j, 0], points[j, 1], 
                                     c='red', s=100, marker='x', 
                                     linewidth=3, label='Negative' if j == 0 else "")
        
        axes[1, 0].set_title('Image with Prompts')
        axes[1, 0].axis('off')
        if prompt_points is not None:
            axes[1, 0].legend()
        
        # Overlay comparison
        if images.shape[1] == 1:  # Grayscale
            axes[1, 1].imshow(img_display, cmap='gray', alpha=0.7)
        else:  # RGB
            axes[1, 1].imshow(img_display, alpha=0.7)
        axes[1, 1].imshow(masks[i], cmap='Reds', alpha=0.5, label='Ground Truth')
        axes[1, 1].imshow(predicted_masks[i], cmap='Blues', alpha=0.5, label='Prediction')
        axes[1, 1].set_title('GT (Red) vs Pred (Blue) Overlay')
        axes[1, 1].axis('off')
        
        # Difference map
        diff_map = np.abs(masks[i] - predicted_masks[i])
        axes[1, 2].imshow(diff_map, cmap='hot')
        axes[1, 2].set_title('Difference Map')
        axes[1, 2].axis('off')
        
        # Add sample information
        sample_name = sample_names[i] if i < len(sample_names) else f"Sample_{i}"
        fig.suptitle(f'Sample: {sample_name}', fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        # Save the visualization
        save_path = os.path.join(save_dir, f'{sample_name}_visualization.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved visualization for sample {i+1}/{n_samples}: {save_path}")


def enhanced_evaluation_with_visualization(model, valloader, criterion, opt, args, n_vis_samples=5):
    """
    Enhanced evaluation function that includes visualization
    """
    model.eval()
    
    all_images = []
    all_gt_masks = []
    all_pred_masks = []
    all_prompts = []
    all_sample_names = []
    
    total_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for batch_idx, (images, masks, sample_names) in enumerate(valloader):
            if isinstance(sample_names, tuple):
                sample_names = sample_names[0] if len(sample_names) > 0 else [f"batch_{batch_idx}_sample_{i}" for i in range(len(images))]
            
            images = images.cuda()
            masks = masks.cuda()
            
            # Generate prompts (assuming you have this function)
            try:
                prompts = get_click_prompt(masks, class_id=1)
            except:
                # Fallback: create dummy prompts if function not available
                batch_size = images.shape[0]
                # Random points for demonstration
                points = torch.rand(batch_size, 1, 2) * images.shape[-1]
                labels = torch.ones(batch_size, 1)
                prompts = (points.cuda(), labels.cuda())
            
            # Forward pass
            if args.modelname in ['SAM', 'SAMFull', 'SAMHead', 'MSA', 'SAMed', 'SAMUS']:
                outputs = model(images, prompts)
            else:
                outputs = model(images)
            
            # Calculate loss
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            loss = criterion(outputs, masks)
            total_loss += loss.item()
            total_samples += images.shape[0]
            
            # Convert outputs to predictions
            if len(outputs.shape) == 4 and outputs.shape[1] > 1:
                pred_masks = torch.softmax(outputs, dim=1)[:, 1]  # Get foreground class
            else:
                pred_masks = torch.sigmoid(outputs)
                if len(pred_masks.shape) == 4:
                    pred_masks = pred_masks.squeeze(1)
            
            # Store data for visualization
            if len(all_images) < n_vis_samples:
                remaining_samples = n_vis_samples - len(all_images)
                n_current = min(remaining_samples, images.shape[0])
                
                all_images.extend([img.cpu() for img in images[:n_current]])
                all_gt_masks.extend([mask.cpu() for mask in masks[:n_current]])
                all_pred_masks.extend([pred.cpu() for pred in pred_masks[:n_current]])
                all_prompts.extend([prompts] * n_current)
                
                if isinstance(sample_names, list):
                    all_sample_names.extend(sample_names[:n_current])
                else:
                    all_sample_names.extend([f"batch_{batch_idx}_sample_{i}" for i in range(n_current)])
            
            if len(all_images) >= n_vis_samples:
                break
    
    # Create visualization
    if all_images:
        print(f"\nCreating visualizations for {len(all_images)} samples...")
        
        # Stack tensors for visualization
        vis_images = torch.stack(all_images)
        vis_gt_masks = torch.stack(all_gt_masks)
        vis_pred_masks = torch.stack(all_pred_masks)
        
        # Handle prompts (take first valid prompt set)
        vis_prompts = all_prompts[0] if all_prompts and all_prompts[0] is not None else None
        
        visualize_prompts_and_masks(
            vis_images, vis_gt_masks, vis_pred_masks, 
            vis_prompts, all_sample_names, 
            save_dir=f"visualization_results_{args.task}_{args.modelname}",
            n_samples=len(all_images)
        )
    
    avg_loss = total_loss / len(valloader)
    return avg_loss


def main():
    # =========================================== parameters setting ==================================================
    parser = argparse.ArgumentParser(description='Networks')
    parser.add_argument('--modelname', default='SAMUS', type=str, help='type of model, e.g., SAM, SAMFull, SAMHead, MSA, SAMed, SAMUS...')
    parser.add_argument('-encoder_input_size', type=int, default=256, help='the image size of the encoder input, 1024 in SAM and MSA, 512 in SAMed, 256 in SAMUS') 
    parser.add_argument('-low_image_size', type=int, default=128, help='the image embedding size, 256 in SAM and MSA, 128 in SAMed and SAMUS') 
    parser.add_argument('--task', default='BUSI', help='task or dataset name')
    parser.add_argument('--vit_name', type=str, default='vit_b', help='select the vit model for the image encoder of sam')
    parser.add_argument('--sam_ckpt', type=str, default='checkpoints/sam_vit_b_01ec64.pth', help='Pretrained checkpoint of SAM')
    parser.add_argument('--batch_size', type=int, default=8, help='batch_size per gpu')
    parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
    parser.add_argument('--base_lr', type=float, default=0.0001, help='segmentation network learning rate, 0.005 for SAMed, 0.0001 for MSA')
    parser.add_argument('--warmup', type=bool, default=False, help='If activated, warp up the learning from a lower lr to the base_lr')
    parser.add_argument('--warmup_period', type=int, default=250, help='Warp up iterations, only valid whrn warmup is activated')
    parser.add_argument('-keep_log', type=bool, default=False, help='keep the loss&lr&dice during training or not')
    
    # New visualization parameters
    parser.add_argument('--visualize', action='store_true', help='Enable visualization of prompts and masks')
    parser.add_argument('--n_vis_samples', type=int, default=5, help='Number of samples to visualize')
    parser.add_argument('--vis_save_dir', type=str, default=None, help='Directory to save visualizations (default: auto-generated)')

    args = parser.parse_args()
    opt = get_config(args.task)
    print("task", args.task, "checkpoints:", opt.load_path)
    opt.mode = "val"
    opt.visual = True
    opt.modelname = args.modelname
    device = torch.device(opt.device)

    # =============================================================== add the seed to make sure the results are reproducible ==============================================================
    seed_value = 300
    np.random.seed(seed_value)
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True

    # =========================================================================== model and data preparation ============================================================================
    opt.batch_size = args.batch_size * args.n_gpu

    tf_val = JointTransform2D(img_size=args.encoder_input_size, low_img_size=args.low_image_size, ori_size=opt.img_size, crop=opt.crop, p_flip=0, color_jitter_params=None, long_mask=True)
    val_dataset = ImageToImage2D(opt.data_path, opt.test_split, tf_val, img_size=args.encoder_input_size, class_id=1)
    valloader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    if args.modelname=="SAMed":
        opt.classes=2
    model = get_model(args.modelname, args=args, opt=opt)
    model.to(device)
    model.train()

    checkpoint = torch.load(opt.load_path)
    new_state_dict = {}
    for k,v in checkpoint.items():
        if k[:7] == 'module.':
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)

    optimizer = optim.Adam(model.parameters(), lr=args.base_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    criterion = get_criterion(modelname=args.modelname, opt=opt)

    # ========================================================================= begin to evaluate the model ============================================================================
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total_params: {}".format(pytorch_total_params))
    input = torch.randn(1, 1, args.encoder_input_size, args.encoder_input_size).cuda()
    points = (torch.tensor([[[1, 2]]]).float().cuda(), torch.tensor([[1]]).float().cuda())
    flops, params = profile(model, inputs=(input, points), )
    print('Gflops:', flops/1000000000, 'params:', params)

    model.eval()

    if opt.mode == "train":
        dices, mean_dice, _, val_losses = get_eval(valloader, model, criterion=criterion, opt=opt, args=args)
        print("mean dice:", mean_dice)
    else:
        # Run standard evaluation
        mean_dice, mean_hdis, mean_iou, mean_acc, mean_se, mean_sp, std_dice, std_hdis, std_iou, std_acc, std_se, std_sp = get_eval(valloader, model, criterion=criterion, opt=opt, args=args)
        
        print("dataset:" + args.task + " -----------model name: "+ args.modelname)
        print("task", args.task, "checkpoints:", opt.load_path)
        
        # Print formatted evaluation results
        print("\n========== Evaluation Metrics (Foreground Class) ==========\n")
        print(f"{'Metric':<15}{'Mean':>10}{'Std Dev':>15}")
        print("-" * 40)
        print(f"{'Dice Score':<15}{mean_dice[1]:>10.2f}{std_dice[1]:>15.2f}")
        print(f"{'Hausdorff Dist':<15}{mean_hdis[1]:>10.2f}{std_hdis[1]:>15.2f}")
        print(f"{'IoU':<15}{mean_iou[1]:>10.2f}{std_iou[1]:>15.2f}")
        print(f"{'Accuracy':<15}{mean_acc[1]:>10.2f}{std_acc[1]:>15.2f}")
        print(f"{'Sensitivity':<15}{mean_se[1]:>10.2f}{std_se[1]:>15.2f}")
        print(f"{'Specificity':<15}{mean_sp[1]:>10.2f}{std_sp[1]:>15.2f}")
        print("-" * 40)
        
        # Run visualization if requested
        if args.visualize:
            print(f"\n========== Running Visualization for {args.n_vis_samples} samples ==========")
            enhanced_evaluation_with_visualization(
                model, valloader, criterion, opt, args, 
                n_vis_samples=args.n_vis_samples
            )

if __name__ == '__main__':
    main()