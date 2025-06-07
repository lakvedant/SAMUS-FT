from ast import arg
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = ''
import argparse
from pickle import FALSE, TRUE
from statistics import mode
from tkinter import image_names
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import time
import random
import wandb
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


class EarlyStopping:
    """Early stopping utility class"""
    def __init__(self, patience=15, min_delta=0.0001, mode='max', verbose=True):
        """
        Args:
            patience (int): How many epochs to wait after last improvement
            min_delta (float): Minimum change to qualify as an improvement
            mode (str): 'max' for maximizing metric (dice), 'min' for minimizing (loss)
            verbose (bool): Print early stopping messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, metric):
        score = metric
        
        if self.best_score is None:
            self.best_score = score
            return False
            
        if self.mode == 'max':
            # For metrics like dice score (higher is better)
            if score > self.best_score + self.min_delta:
                self.best_score = score
                self.counter = 0
                return False
            else:
                self.counter += 1
        else:
            # For metrics like loss (lower is better)
            if score < self.best_score - self.min_delta:
                self.best_score = score
                self.counter = 0
                return False
            else:
                self.counter += 1
                
        if self.counter >= self.patience:
            if self.verbose:
                print(f"\nEarly stopping triggered! No improvement for {self.patience} epochs.")
                print(f"Best score was: {self.best_score:.4f}")
            self.early_stop = True
            return True
            
        if self.verbose and self.counter > 0:
            print(f"Early stopping counter: {self.counter}/{self.patience}")
            
        return False


def main():
    #  ============================================================================= parameters setting ====================================================================================

    parser = argparse.ArgumentParser(description='Networks')
    parser.add_argument('--modelname', default='SAMUS', type=str, help='type of model, e.g., SAM, SAMFull, MedSAM, MSA, SAMed, SAMUS...')
    parser.add_argument('-encoder_input_size', type=int, default=256, help='the image size of the encoder input, 1024 in SAM and MSA, 512 in SAMed, 256 in SAMUS')
    parser.add_argument('-low_image_size', type=int, default=128, help='the image embedding size, 256 in SAM and MSA, 128 in SAMed and SAMUS')
    parser.add_argument('--task', default='US30K', help='task or dataset name')
    parser.add_argument('--vit_name', type=str, default='vit_b', help='select the vit model for the image encoder of sam')
    parser.add_argument('--sam_ckpt', type=str, default='checkpoints/sam_vit_b_01ec64.pth', help='Pretrained checkpoint of SAM')
    parser.add_argument('--resume_checkpoint', type=str, default='checkpoints/SAMUS.pth', help='Path to trained SAMUS checkpoint to resume from')
    parser.add_argument('--batch_size', type=int, default=8, help='batch_size per gpu')
    parser.add_argument('--data_subset_ratio', type=float, default=0.02, help='Use only a fraction of training data (0.25 = 25% of data)')
    parser.add_argument('--val_subset_ratio', type=float, default=0.02, help='Use only a fraction of validation data (0.2 = 20% of data)')
    parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
    parser.add_argument('--base_lr', type=float, default=0.0005, help='segmentation network learning rate, 0.005 for SAMed, 0.0001 for MSA')
    parser.add_argument('--warmup', type=bool, default=False, help='If activated, warp up the learning from a lower lr to the base_lr') 
    parser.add_argument('--warmup_period', type=int, default=250, help='Warp up iterations, only valid whrn warmup is activated')
    parser.add_argument('-keep_log', type=bool, default=False, help='keep the loss&lr&dice during training or not')
    parser.add_argument('--epochs', type=int, default=5, help='number of training epochs')
    parser.add_argument('--start_epoch', type=int, default=0, help='Starting epoch number (useful for resuming training)')
    
    # NEW: Early stopping parameters
    parser.add_argument('--early_stopping_patience', type=int, default=15, help='Number of epochs to wait for improvement before stopping')
    parser.add_argument('--early_stopping_min_delta', type=float, default=0.0001, help='Minimum change to qualify as improvement')
    
    # NEW: WandB parameters
    parser.add_argument('--use_wandb', type=bool, default=True, help='Use Weights & Biases for logging')
    parser.add_argument('--wandb_project', type=str, default='medical-segmentation', help='WandB project name')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='WandB run name (auto-generated if None)')

    args = parser.parse_args()
    opt = get_config(args.task) 
    
    # Override config epochs with command line argument
    if hasattr(opt, 'epochs'):
        opt.epochs = args.epochs
    else:
        setattr(opt, 'epochs', args.epochs)

    # NEW: Initialize WandB
    if args.use_wandb:
        # Set your API key
        os.environ['WANDB_API_KEY'] = '4ac28743425731f3f01c3d7a2013e64ff47949cf'
        
        # Generate run name if not provided
        if args.wandb_run_name is None:
            timestr = time.strftime('%m%d_%H%M')
            args.wandb_run_name = f"{args.modelname}_{args.task}_{timestr}"
        
        # Initialize wandb
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                'model': args.modelname,
                'task': args.task,
                'epochs': opt.epochs,
                'batch_size': args.batch_size,
                'learning_rate': args.base_lr,
                'encoder_input_size': args.encoder_input_size,
                'low_image_size': args.low_image_size,
                'data_subset_ratio': args.data_subset_ratio,
                'val_subset_ratio': args.val_subset_ratio,
                'warmup': args.warmup,
                'warmup_period': args.warmup_period,
                'early_stopping_patience': args.early_stopping_patience,
                'early_stopping_min_delta': args.early_stopping_min_delta,
                'resume_checkpoint': args.resume_checkpoint,
                'start_epoch': args.start_epoch,
                'fine_tuning': True,
            },
            tags=[args.modelname, args.task, 'fine-tuning']
        )
        print(f"WandB initialized. Project: {args.wandb_project}, Run: {args.wandb_run_name}")

    device = torch.device(opt.device)
    if args.keep_log:
        logtimestr = time.strftime('%m%d%H%M')
        boardpath = opt.tensorboard_path + args.modelname + opt.save_path_code + logtimestr
        if not os.path.isdir(boardpath):
            os.makedirs(boardpath)
        TensorWriter = SummaryWriter(boardpath)

    #  =============================================================== add the seed to make sure the results are reproducible ==============================================================

    seed_value = 1234
    np.random.seed(seed_value)
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True

    #  =========================================================================== model and data preparation ============================================================================
    
    model = get_model(args.modelname, args=args, opt=opt)
    opt.batch_size = args.batch_size * args.n_gpu

    tf_train = JointTransform2D(img_size=args.encoder_input_size, low_img_size=args.low_image_size, ori_size=opt.img_size, crop=opt.crop, p_flip=0.0, p_rota=0.5, p_scale=0.5, p_gaussn=0.0,
                                p_contr=0.5, p_gama=0.5, p_distor=0.0, color_jitter_params=None, long_mask=True)
    tf_val = JointTransform2D(img_size=args.encoder_input_size, low_img_size=args.low_image_size, ori_size=opt.img_size, crop=opt.crop, p_flip=0, color_jitter_params=None, long_mask=True)
    
    # Load full datasets first
    full_train_dataset = ImageToImage2D(opt.data_path, opt.train_split, tf_train, img_size=args.encoder_input_size)
    full_val_dataset = ImageToImage2D(opt.data_path, opt.val_split, tf_val, img_size=args.encoder_input_size)
    
    print(f"Full dataset sizes - Train: {len(full_train_dataset)}, Val: {len(full_val_dataset)}")
    
    # NEW: Improved random subset selection with fixed seed for reproducibility
    # Set a different seed for subset selection to ensure randomness while maintaining reproducibility
    torch.manual_seed(seed_value + 42)  # Different seed for subset selection
    
    # Create random subsets
    if args.data_subset_ratio < 1.0:
        subset_size = int(len(full_train_dataset) * args.data_subset_ratio)
        train_indices = torch.randperm(len(full_train_dataset))[:subset_size]
        train_dataset = torch.utils.data.Subset(full_train_dataset, train_indices)
        print(f"Using {args.data_subset_ratio*100:.1f}% of training data: {len(train_dataset)} samples")
        
        # Log subset indices for reproducibility
        if args.use_wandb:
            wandb.log({"train_subset_size": len(train_dataset), "train_subset_ratio": args.data_subset_ratio})
    else:
        train_dataset = full_train_dataset
        print(f"Using full training dataset: {len(train_dataset)} samples")
    
    if args.val_subset_ratio < 1.0:
        val_subset_size = int(len(full_val_dataset) * args.val_subset_ratio)
        val_indices = torch.randperm(len(full_val_dataset))[:val_subset_size]
        val_dataset = torch.utils.data.Subset(full_val_dataset, val_indices)
        print(f"Using {args.val_subset_ratio*100:.1f}% of validation data: {len(val_dataset)} samples")
        
        # Log subset indices for reproducibility
        if args.use_wandb:
            wandb.log({"val_subset_size": len(val_dataset), "val_subset_ratio": args.val_subset_ratio})
    else:
        val_dataset = full_val_dataset
        print(f"Using full validation dataset: {len(val_dataset)} samples")
    
    # Reset seed to original value for training
    torch.manual_seed(seed_value)
    
    trainloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    valloader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    model.to(device)
    
    # NEW: Load your trained SAMUS checkpoint for fine-tuning
    checkpoint_loaded = False
    initial_dice = 0.0
    
    if args.resume_checkpoint and os.path.exists(args.resume_checkpoint):
        print(f"\n{'='*80}")
        print(f"LOADING CHECKPOINT FOR FINE-TUNING")
        print(f"Checkpoint Path: {args.resume_checkpoint}")
        print(f"{'='*80}")
        
        try:
            checkpoint = torch.load(args.resume_checkpoint, map_location=device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                # If checkpoint contains state_dict, optimizer, epoch info, etc.
                if 'model_state_dict' in checkpoint:
                    model_state_dict = checkpoint['model_state_dict']
                    if 'epoch' in checkpoint:
                        args.start_epoch = checkpoint['epoch'] + 1
                        print(f"Resuming from epoch: {args.start_epoch}")
                    if 'best_dice' in checkpoint:
                        initial_dice = checkpoint['best_dice']
                        print(f"Previous best dice: {initial_dice:.4f}")
                elif 'state_dict' in checkpoint:
                    model_state_dict = checkpoint['state_dict']
                else:
                    # Assume the entire dict is the state_dict
                    model_state_dict = checkpoint
            else:
                # Direct state_dict
                model_state_dict = checkpoint
            
            # Clean state dict keys (remove 'module.' prefix if present)
            new_state_dict = {}
            for k, v in model_state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v  # Remove 'module.' prefix
                else:
                    new_state_dict[k] = v
            
            # Load the state dict
            missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
            
            if missing_keys:
                print(f"Warning: Missing keys in checkpoint: {missing_keys}")
            if unexpected_keys:
                print(f"Warning: Unexpected keys in checkpoint: {unexpected_keys}")
            
            checkpoint_loaded = True
            print("✅ Checkpoint loaded successfully!")
            print(f"Fine-tuning will continue from epoch {args.start_epoch}")
            
        except Exception as e:
            print(f"❌ Error loading checkpoint: {str(e)}")
            print("Training will start from scratch with pretrained weights...")
            checkpoint_loaded = False
    
    elif args.resume_checkpoint:
        print(f"⚠️  Checkpoint file not found: {args.resume_checkpoint}")
        print("Training will start from scratch with pretrained weights...")
    
    # Load original pretrained weights if checkpoint loading failed
    if not checkpoint_loaded and opt.pre_trained:
        print(f"Loading original pretrained weights from: {opt.load_path}")
        checkpoint = torch.load(opt.load_path)
        new_state_dict = {}
        for k,v in checkpoint.items():
            if k[:7] == 'module.':
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict)
      
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    
    if args.warmup:
        b_lr = args.base_lr / args.warmup_period
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=b_lr, betas=(0.9, 0.999), weight_decay=0.1)
    else:
        b_lr = args.base_lr
        optimizer = optim.Adam(model.parameters(), lr=args.base_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
   
    criterion = get_criterion(modelname=args.modelname, opt=opt)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total_params: {}".format(pytorch_total_params))
    
    # NEW: Log model parameters to WandB
    if args.use_wandb:
        wandb.log({"total_parameters": pytorch_total_params})
    
    # NEW: Initialize Early Stopping
    early_stopping = EarlyStopping(
        patience=args.early_stopping_patience,
        min_delta=args.early_stopping_min_delta,
        mode='max',  # We want to maximize dice score
        verbose=True
    )
    
    print(f"\n{'='*80}")
    print(f"TRAINING SETUP:")
    print(f"Mode: {'Fine-tuning from checkpoint' if checkpoint_loaded else 'Training from pretrained weights'}")
    print(f"Starting Epoch: {args.start_epoch + 1}")
    print(f"Total Epochs: {opt.epochs}")
    print(f"Remaining Epochs: {opt.epochs - args.start_epoch}")
    print(f"Batches per Epoch: {len(trainloader)}")
    print(f"Total Iterations: {opt.epochs * len(trainloader)}")
    print(f"Batch Size: {opt.batch_size}")
    print(f"Model: {args.modelname}")
    print(f"Checkpoint Loaded: {'Yes' if checkpoint_loaded else 'No'}")
    if checkpoint_loaded:
        print(f"Previous Best Dice: {initial_dice:.4f}")
    print(f"Early Stopping Patience: {args.early_stopping_patience}")
    print(f"WandB Logging: {'Enabled' if args.use_wandb else 'Disabled'}")
    print(f"{'='*80}\n")

    #  ========================================================================= begin to train the model ============================================================================
    iter_num = args.start_epoch * len(trainloader)  # Adjust iteration counter for resumed training
    max_iterations = opt.epochs * len(trainloader)
    best_dice = max(initial_dice, 0.0)  # Use loaded dice score as starting point
    loss_log, dice_log = [], []
    best_model_path = None
    training_start_time = time.time()
    
    # NEW: Log initial state to WandB
    if args.use_wandb:
        wandb.log({
            "checkpoint_loaded": checkpoint_loaded,
            "starting_epoch": args.start_epoch,
            "initial_best_dice": best_dice,
            "remaining_epochs": opt.epochs - args.start_epoch
        })
    
    for epoch in range(args.start_epoch, opt.epochs):  # Start from the loaded epoch
        model.train()
        train_losses = 0
        epoch_start_time = time.time()
        
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch+1}/{opt.epochs} STARTED")
        print(f"Fine-tuning Mode: {'Yes' if checkpoint_loaded else 'No'}")
        print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")
        
        for batch_idx, (datapack) in enumerate(trainloader):
            imgs = datapack['image'].to(dtype = torch.float32, device=opt.device)
            masks = datapack['low_mask'].to(dtype = torch.float32, device=opt.device)
            bbox = torch.as_tensor(datapack['bbox'], dtype=torch.float32, device=opt.device)
            pt = get_click_prompt(datapack, opt)
            
            pred = model(imgs, pt, bbox)
            train_loss = criterion(pred, masks)
            
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            train_losses += train_loss.item()
            
            # Enhanced batch progress tracking
            if (batch_idx + 1) % 50 == 0 or batch_idx == 0:
                elapsed_time = time.time() - epoch_start_time
                progress_percent = ((batch_idx + 1) / len(trainloader)) * 100
                avg_loss = train_losses / (batch_idx + 1)
                
                print(f"  Batch [{batch_idx+1:4d}/{len(trainloader):4d}] "
                      f"({progress_percent:5.1f}%) | "
                      f"Loss: {train_loss.item():.4f} | "
                      f"Avg Loss: {avg_loss:.4f} | "
                      f"Time: {elapsed_time:.1f}s")
                
                # NEW: Log batch metrics to WandB
                if args.use_wandb:
                    wandb.log({
                        "batch_loss": train_loss.item(),
                        "batch_avg_loss": avg_loss,
                        "batch_progress": progress_percent,
                        "batch_step": epoch * len(trainloader) + batch_idx
                    })
            
            # Learning rate adjustment (account for resumed training)
            if args.warmup and iter_num < args.warmup_period:
                lr_ = args.base_lr * ((iter_num + 1) / args.warmup_period)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            else:
                if args.warmup:
                    shift_iter = iter_num - args.warmup_period
                    assert shift_iter >= 0, f'Shift iter is {shift_iter}, smaller than zero'
                    lr_ = args.base_lr * (1.0 - shift_iter / max_iterations) ** 0.9
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_
            iter_num = iter_num + 1

        # Epoch completion logging
        epoch_time = time.time() - epoch_start_time
        avg_train_loss = train_losses / len(trainloader)
        current_lr = optimizer.state_dict()['param_groups'][0]['lr']
        
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch+1}/{opt.epochs} COMPLETED")
        print(f"Training Loss: {avg_train_loss:.4f}")
        print(f"Epoch Time: {epoch_time:.2f}s")
        print(f"Learning Rate: {current_lr:.8f}")
        
        # Store logs
        loss_log.append(avg_train_loss)
        
        # Traditional logging
        if args.keep_log:
            TensorWriter.add_scalar('train_loss', avg_train_loss, epoch)
            TensorWriter.add_scalar('learning rate', current_lr, epoch)

        # Validation
        if epoch % opt.eval_freq == 0:
            print(f"Running validation...")
            model.eval()
            val_start_time = time.time()
            dices, mean_dice, _, val_losses = get_eval(valloader, model, criterion=criterion, opt=opt, args=args)
            val_time = time.time() - val_start_time
            
            print(f"Validation Loss: {val_losses:.4f}")
            print(f"Validation Dice: {mean_dice:.4f}")
            print(f"Validation Time: {val_time:.2f}s")
            
            # Store validation metrics
            dice_log.append(mean_dice)
            
            # Traditional logging
            if args.keep_log:
                TensorWriter.add_scalar('val_loss', val_losses, epoch)
                TensorWriter.add_scalar('dices', mean_dice, epoch)
                
            # NEW: WandB logging
            if args.use_wandb:
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": avg_train_loss,
                    "val_loss": val_losses,
                    "val_dice": mean_dice,
                    "learning_rate": current_lr,
                    "epoch_time": epoch_time,
                    "val_time": val_time
                })
                
            # Model saving logic
            if mean_dice > best_dice:
                if best_model_path and os.path.exists(best_model_path):
                    try:
                        os.remove(best_model_path)
                        print(f"Removed previous best model: {os.path.basename(best_model_path)}")
                    except OSError as e:
                        print(f"Warning: Could not remove previous model file: {e}")
                
                best_dice = mean_dice
                
                timestr = time.strftime('%m%d%H%M')
                if not os.path.isdir(opt.save_path):
                    os.makedirs(opt.save_path)
                
                best_model_path = opt.save_path + args.modelname + opt.save_path_code + '_best_' + timestr + '_epoch' + str(epoch) + '_dice' + f'{best_dice:.4f}' + '_finetune.pth'
                
                # NEW: Save checkpoint with additional info for future resuming
                checkpoint_data = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'best_dice': best_dice,
                    'loss_log': loss_log,
                    'dice_log': dice_log,
                    'args': vars(args)
                }
                torch.save(checkpoint_data, best_model_path, _use_new_zipfile_serialization=False)
                print(f"NEW BEST MODEL SAVED! Dice: {best_dice:.4f}")
                print(f"Saved as: {os.path.basename(best_model_path)}")
                
                # NEW: Log best model to WandB
                if args.use_wandb:
                    wandb.log({"best_dice": best_dice, "best_model_epoch": epoch + 1})
                    # Optionally save model to WandB
                    wandb.save(best_model_path)
            
            # NEW: Early stopping check
            if early_stopping(mean_dice):
                print(f"\nTraining stopped early at epoch {epoch + 1}")
                if args.use_wandb:
                    wandb.log({"early_stopped": True, "early_stop_epoch": epoch + 1})
                break
        else:
            # If no validation this epoch, append previous dice or 0
            dice_log.append(dice_log[-1] if dice_log else 0.0)
        
        # Overall progress tracking
        total_elapsed_time = time.time() - training_start_time
        remaining_epochs = opt.epochs - (epoch + 1)
        completed_epochs = (epoch + 1) - args.start_epoch
        avg_epoch_time = total_elapsed_time / max(completed_epochs, 1)
        estimated_remaining_time = remaining_epochs * avg_epoch_time
        
        total_progress = ((epoch + 1) / opt.epochs) * 100
        fine_tune_progress = (completed_epochs / max(opt.epochs - args.start_epoch, 1)) * 100
        
        print(f"Overall Progress: {total_progress:.1f}% Complete")
        print(f"Fine-tuning Progress: {fine_tune_progress:.1f}% Complete")
        print(f"Total Elapsed: {total_elapsed_time/60:.1f} minutes")
        print(f"Estimated Remaining: {estimated_remaining_time/60:.1f} minutes")
        print(f"Best Dice So Far: {best_dice:.4f}")
        print(f"{'='*60}")
        
        # Save logs
        if args.keep_log:
            with open(opt.tensorboard_path + args.modelname + opt.save_path_code + logtimestr + '/trainloss.txt', 'w') as f:
                for loss in loss_log:
                    f.write(str(loss)+'\n')
            with open(opt.tensorboard_path + args.modelname + opt.save_path_code + logtimestr + '/dice.txt', 'w') as f:
                for dice in dice_log:
                    f.write(str(dice)+'\n')
    
    # Final training summary
    total_training_time = time.time() - training_start_time
    final_epoch = epoch + 1
    completed_epochs = final_epoch - args.start_epoch
    
    print(f"\n{'='*80}")
    print(f"FINE-TUNING COMPLETED!")
    print(f"Starting Epoch: {args.start_epoch + 1}")
    print(f"Final Epoch: {final_epoch}")
    print(f"Epochs Completed in This Session: {completed_epochs}")
    print(f"Total Training Time (This Session): {total_training_time/60:.2f} minutes ({total_training_time/3600:.2f} hours)")
    print(f"Final Best Dice Score: {best_dice:.4f}")
    print(f"Best Model Saved As: {os.path.basename(best_model_path) if best_model_path else 'No model saved'}")
    print(f"Average Time per Epoch: {total_training_time/max(completed_epochs, 1):.2f} seconds")
    print(f"Early Stopping: {'Yes' if early_stopping.early_stop else 'No'}")
    print(f"Checkpoint Loaded at Start: {'Yes' if checkpoint_loaded else 'No'}")
    print(f"{'='*80}")
    
    # NEW: Final WandB logging
    if args.use_wandb:
        wandb.log({
            "final_best_dice": best_dice,
            "total_training_time_minutes": total_training_time/60,
            "epochs_completed_this_session": completed_epochs,
            "final_epoch": final_epoch,
            "avg_time_per_epoch": total_training_time/max(completed_epochs, 1),
            "early_stopped": early_stopping.early_stop,
            "fine_tuning_completed": True
        })
        
        # Finish WandB run
        wandb.finish()
        print("WandB run completed and synced!")

if __name__ == '__main__':
    main()