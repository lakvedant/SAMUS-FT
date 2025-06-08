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
    def __init__(self, patience=10, min_delta=0.001, mode='max', verbose=True):
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

wandb.login(key='4ac28743425731f3f01c3d7a2013e64ff47949cf')

def main():
    #  ============================================================================= parameters setting ====================================================================================

    parser = argparse.ArgumentParser(description='Networks')
    parser.add_argument('--modelname', default='SAMUS', type=str, help='type of model, e.g., SAM, SAMFull, MedSAM, MSA, SAMed, SAMUS...')
    parser.add_argument('-encoder_input_size', type=int, default=256, help='the image size of the encoder input, 1024 in SAM and MSA, 512 in SAMed, 256 in SAMUS')
    parser.add_argument('-low_image_size', type=int, default=128, help='the image embedding size, 256 in SAM and MSA, 128 in SAMed and SAMUS')
    parser.add_argument('--task', default='US30K', help='task or dataset name')
    parser.add_argument('--vit_name', type=str, default='vit_b', help='select the vit model for the image encoder of sam')
    parser.add_argument('--sam_ckpt', type=str, default='checkpoints/sam_vit_b_01ec64.pth', help='Pretrained checkpoint of SAM')
    parser.add_argument('--resume_checkpoint', type=str, default='/content/drive/MyDrive/US30K/SAMUS.pth', help='Path to trained SAMUS checkpoint to resume from')
    parser.add_argument('--batch_size', type=int, default=8, help='batch_size per gpu')
    parser.add_argument('--data_subset_ratio', type=float, default=0.02, help='Use only a fraction of training data (0.25 = 25% of data)')
    parser.add_argument('--val_subset_ratio', type=float, default=0.02, help='Use only a fraction of validation data (0.2 = 20% of data)')
    parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
    parser.add_argument('--base_lr', type=float, default=0.0001, help='segmentation network learning rate')
    parser.add_argument('--warmup', type=bool, default=False, help='If activated, warp up the learning from a lower lr to the base_lr') 
    parser.add_argument('--warmup_period', type=int, default=100, help='Warp up iterations, only valid when warmup is activated')
    parser.add_argument('-keep_log', type=bool, default=False, help='keep the loss&lr&dice during training or not')
    parser.add_argument('--epochs', type=int, default=5, help='number of training epochs')
    parser.add_argument('--start_epoch', type=int, default=0, help='Starting epoch number (useful for resuming training)')
    
    # Optimized early stopping parameters for faster training
    parser.add_argument('--early_stopping_patience', type=int, default=10, help='Number of epochs to wait for improvement before stopping')
    parser.add_argument('--early_stopping_min_delta', type=float, default=0.001, help='Minimum change to qualify as improvement')
    
    # WandB parameters
    parser.add_argument('--use_wandb', type=bool, default=True, help='Use Weights & Biases for logging')
    parser.add_argument('--wandb_project', type=str, default='medical-segmentation', help='WandB project name')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='WandB run name (auto-generated if None)')

    args = parser.parse_args()
    opt = get_config(args.task) 
    
    # Override config epochs with command line argument
    opt.epochs = args.epochs

    # Initialize WandB FIRST with proper configuration
    if args.use_wandb:
        # Generate run name if not provided
        if args.wandb_run_name is None:
            timestr = time.strftime('%m%d_%H%M')
            args.wandb_run_name = f"{args.modelname}_{args.task}_{timestr}_subset{args.data_subset_ratio}"
        
        # Initialize wandb with explicit configuration
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
                'image_size': f"{args.encoder_input_size}x{args.encoder_input_size}",
                'architecture': 'SAMUS'
            },
            tags=[args.modelname, args.task, 'fine-tuning', '256x256'],
            notes=f"Fine-tuning {args.modelname} on {args.task} with {args.data_subset_ratio*100}% data subset"
        )
        print(f"✅ WandB initialized successfully!")
        print(f"   Project: {args.wandb_project}")
        print(f"   Run: {args.wandb_run_name}")
        print(f"   Dashboard: https://wandb.ai/{wandb.run.entity}/{wandb.run.project}/runs/{wandb.run.id}")

    device = torch.device(opt.device)
    
    # Optimized tensorboard logging
    if args.keep_log:
        logtimestr = time.strftime('%m%d%H%M')
        boardpath = opt.tensorboard_path + args.modelname + opt.save_path_code + logtimestr
        if not os.path.isdir(boardpath):
            os.makedirs(boardpath)
        TensorWriter = SummaryWriter(boardpath)

    # Set seed for reproducibility
    seed_value = 1234
    np.random.seed(seed_value)
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes

    # Model and data preparation
    model = get_model(args.modelname, args=args, opt=opt)
    opt.batch_size = args.batch_size * args.n_gpu

    # Optimized transforms for faster training (reduced augmentation)
    tf_train = JointTransform2D(
        img_size=args.encoder_input_size, 
        low_img_size=args.low_image_size, 
        ori_size=opt.img_size, 
        crop=opt.crop, 
        p_flip=0.0,    # Reduced augmentation for speed
        p_rota=0.2,    # Reduced from 0.5
        p_scale=0.2,   # Reduced from 0.5
        p_gaussn=0.0, 
        p_contr=0.2,   # Reduced from 0.5
        p_gama=0.2,    # Reduced from 0.5
        p_distor=0.0, 
        color_jitter_params=None, 
        long_mask=True
    )
    
    tf_val = JointTransform2D(
        img_size=args.encoder_input_size, 
        low_img_size=args.low_image_size, 
        ori_size=opt.img_size, 
        crop=opt.crop, 
        p_flip=0, 
        color_jitter_params=None, 
        long_mask=True
    )
    
    # Load datasets
    full_train_dataset = ImageToImage2D(opt.data_path, opt.train_split, tf_train, img_size=args.encoder_input_size)
    full_val_dataset = ImageToImage2D(opt.data_path, opt.val_split, tf_val, img_size=args.encoder_input_size)
    
    print(f"Full dataset sizes - Train: {len(full_train_dataset)}, Val: {len(full_val_dataset)}")
    
    # Create subsets with reproducible randomness
    torch.manual_seed(seed_value + 42)
    
    if args.data_subset_ratio < 1.0:
        subset_size = int(len(full_train_dataset) * args.data_subset_ratio)
        train_indices = torch.randperm(len(full_train_dataset))[:subset_size]
        train_dataset = torch.utils.data.Subset(full_train_dataset, train_indices)
        print(f"Using {args.data_subset_ratio*100:.1f}% of training data: {len(train_dataset)} samples")
    else:
        train_dataset = full_train_dataset
        print(f"Using full training dataset: {len(train_dataset)} samples")
    
    if args.val_subset_ratio < 1.0:
        val_subset_size = int(len(full_val_dataset) * args.val_subset_ratio)
        val_indices = torch.randperm(len(full_val_dataset))[:val_subset_size]
        val_dataset = torch.utils.data.Subset(full_val_dataset, val_indices)
        print(f"Using {args.val_subset_ratio*100:.1f}% of validation data: {len(val_dataset)} samples")
    else:
        val_dataset = full_val_dataset
        print(f"Using full validation dataset: {len(val_dataset)} samples")
    
    torch.manual_seed(seed_value)
    
    # Optimized data loaders for speed
    trainloader = DataLoader(
        train_dataset, 
        batch_size=opt.batch_size, 
        shuffle=True, 
        num_workers=4,  # Reduced from 8 for Colab
        pin_memory=True,
        persistent_workers=True,  # Keep workers alive
        prefetch_factor=2         # Prefetch batches
    )
    
    valloader = DataLoader(
        val_dataset, 
        batch_size=opt.batch_size, 
        shuffle=False, 
        num_workers=4,  # Reduced from 8 for Colab
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    model.to(device)
    
    # FIXED: Proper checkpoint loading
    checkpoint_loaded = False
    initial_dice = 0.0
    
    print(f"\n{'='*80}")
    print(f"CHECKPOINT LOADING")
    print(f"{'='*80}")
    
    if args.resume_checkpoint and os.path.exists(args.resume_checkpoint):
        print(f"📁 Found checkpoint: {args.resume_checkpoint}")
        
        try:
            checkpoint = torch.load(args.resume_checkpoint, map_location=device)
            print(f"📋 Checkpoint type: {type(checkpoint)}")
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                print("📝 Checkpoint keys:", list(checkpoint.keys()))
                
                # Try different possible keys for model state
                model_state_dict = None
                if 'model_state_dict' in checkpoint:
                    model_state_dict = checkpoint['model_state_dict']
                    print("✅ Found 'model_state_dict'")
                elif 'state_dict' in checkpoint:
                    model_state_dict = checkpoint['state_dict']
                    print("✅ Found 'state_dict'")
                elif 'model' in checkpoint:
                    model_state_dict = checkpoint['model']
                    print("✅ Found 'model'")
                else:
                    # Assume the entire dict is the state_dict
                    model_state_dict = checkpoint
                    print("✅ Using entire checkpoint as state_dict")
                
                # Get additional info if available
                if 'epoch' in checkpoint:
                    args.start_epoch = checkpoint['epoch'] + 1
                    print(f"📈 Resuming from epoch: {args.start_epoch}")
                if 'best_dice' in checkpoint:
                    initial_dice = checkpoint['best_dice']
                    print(f"🎯 Previous best dice: {initial_dice:.4f}")
                    
            else:
                # Direct state_dict
                model_state_dict = checkpoint
                print("✅ Direct state_dict format")
            
            # Clean state dict keys (remove 'module.' prefix if present)
            new_state_dict = {}
            for k, v in model_state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v  # Remove 'module.' prefix
                else:
                    new_state_dict[k] = v
            
            # Load the state dict with less strict matching for fine-tuning
            missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
            
            if missing_keys:
                print(f"⚠️  Missing keys: {len(missing_keys)} keys")
                if len(missing_keys) < 10:
                    for key in missing_keys[:5]:
                        print(f"   - {key}")
                    if len(missing_keys) > 5:
                        print(f"   ... and {len(missing_keys)-5} more")
                        
            if unexpected_keys:
                print(f"⚠️  Unexpected keys: {len(unexpected_keys)} keys")
                if len(unexpected_keys) < 10:
                    for key in unexpected_keys[:5]:
                        print(f"   - {key}")
                    if len(unexpected_keys) > 5:
                        print(f"   ... and {len(unexpected_keys)-5} more")
            
            checkpoint_loaded = True
            print("✅ Checkpoint loaded successfully for fine-tuning!")
            
        except Exception as e:
            print(f"❌ Error loading checkpoint: {str(e)}")
            print("   Starting from pretrained weights instead...")
            checkpoint_loaded = False
    
    else:
        print(f"⚠️  Checkpoint not found: {args.resume_checkpoint}")
        print("   Will use pretrained weights instead...")
    
    # Load pretrained weights if checkpoint loading failed
    if not checkpoint_loaded and hasattr(opt, 'pre_trained') and opt.pre_trained and hasattr(opt, 'load_path'):
        print(f"🔄 Loading pretrained weights from: {opt.load_path}")
        try:
            pretrained = torch.load(opt.load_path, map_location=device)
            new_state_dict = {}
            for k, v in pretrained.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
            model.load_state_dict(new_state_dict, strict=False)
            print("✅ Pretrained weights loaded!")
        except Exception as e:
            print(f"⚠️  Could not load pretrained weights: {e}")
      
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    
    # Optimized optimizer setup
    if args.warmup:
        b_lr = args.base_lr / args.warmup_period
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()), 
            lr=b_lr, 
            betas=(0.9, 0.999), 
            weight_decay=0.01,  # Reduced weight decay
            eps=1e-8
        )
    else:
        optimizer = optim.AdamW(  # Changed to AdamW for better performance
            model.parameters(), 
            lr=args.base_lr, 
            betas=(0.9, 0.999), 
            eps=1e-08, 
            weight_decay=0.01,  # Small weight decay
            amsgrad=False
        )
   
    criterion = get_criterion(modelname=args.modelname, opt=opt)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🔢 Total trainable parameters: {pytorch_total_params:,}")
    
    # Log to WandB
    if args.use_wandb:
        wandb.log({
            "total_parameters": pytorch_total_params,
            "checkpoint_loaded": checkpoint_loaded,
            "starting_epoch": args.start_epoch,
            "initial_best_dice": initial_dice,
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset)
        })
    
    # Initialize Early Stopping with more aggressive settings
    early_stopping = EarlyStopping(
        patience=args.early_stopping_patience,
        min_delta=args.early_stopping_min_delta,
        mode='max',
        verbose=True
    )
    
    print(f"\n{'='*80}")
    print(f"TRAINING CONFIGURATION")
    print(f"{'='*80}")
    print(f"🎯 Mode: {'Fine-tuning from checkpoint' if checkpoint_loaded else 'Training from pretrained'}")
    print(f"📅 Epochs: {args.start_epoch + 1} → {opt.epochs} (Total: {opt.epochs - args.start_epoch})")
    print(f"🖼️  Image Size: {args.encoder_input_size}×{args.encoder_input_size}")
    print(f"📦 Batch Size: {opt.batch_size}")
    print(f"📊 Train Batches: {len(trainloader)}")
    print(f"📈 Val Batches: {len(valloader)}")
    print(f"🎓 Model: {args.modelname}")
    print(f"⏰ Early Stopping: {args.early_stopping_patience} epochs patience")
    print(f"📊 WandB: {'Enabled' if args.use_wandb else 'Disabled'}")
    if checkpoint_loaded:
        print(f"🎯 Previous Best Dice: {initial_dice:.4f}")
    print(f"{'='*80}\n")

    # Training loop
    iter_num = args.start_epoch * len(trainloader)
    max_iterations = opt.epochs * len(trainloader)
    best_dice = max(initial_dice, 0.0)
    loss_log, dice_log = [], []
    best_model_path = None
    training_start_time = time.time()
    
    for epoch in range(args.start_epoch, opt.epochs):
        model.train()
        train_losses = 0
        epoch_start_time = time.time()
        
        print(f"\n{'='*50}")
        print(f"EPOCH {epoch+1}/{opt.epochs}")
        print(f"{'='*50}")
        
        for batch_idx, datapack in enumerate(trainloader):
            imgs = datapack['image'].to(dtype=torch.float32, device=device, non_blocking=True)
            masks = datapack['low_mask'].to(dtype=torch.float32, device=device, non_blocking=True)
            bbox = torch.as_tensor(datapack['bbox'], dtype=torch.float32, device=device)
            pt = get_click_prompt(datapack, opt)
            
            # Forward pass
            pred = model(imgs, pt, bbox)
            train_loss = criterion(pred, masks)
            
            # Backward pass
            optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
            train_loss.backward()
            optimizer.step()
            
            train_losses += train_loss.item()
            
            # Progress logging (less frequent for speed)
            if (batch_idx + 1) % 25 == 0 or batch_idx == 0:
                progress = ((batch_idx + 1) / len(trainloader)) * 100
                avg_loss = train_losses / (batch_idx + 1)
                elapsed = time.time() - epoch_start_time
                
                print(f"  [{batch_idx+1:3d}/{len(trainloader):3d}] "
                      f"({progress:5.1f}%) | "
                      f"Loss: {train_loss.item():.4f} | "
                      f"Avg: {avg_loss:.4f} | "
                      f"Time: {elapsed:.1f}s")
                
                # WandB logging (less frequent)
                if args.use_wandb and (batch_idx + 1) % 50 == 0:
                    wandb.log({
                        "batch_loss": train_loss.item(),
                        "batch_avg_loss": avg_loss,
                        "epoch_progress": progress,
                        "step": epoch * len(trainloader) + batch_idx
                    })
            
            # Learning rate scheduling
            if args.warmup and iter_num < args.warmup_period:
                lr_ = args.base_lr * ((iter_num + 1) / args.warmup_period)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            elif args.warmup:
                shift_iter = iter_num - args.warmup_period
                lr_ = args.base_lr * (1.0 - shift_iter / max_iterations) ** 0.9
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
                    
            iter_num += 1

        # Epoch summary
        epoch_time = time.time() - epoch_start_time
        avg_train_loss = train_losses / len(trainloader)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"\n Epoch {epoch+1} Summary:")
        print(f"   Loss: {avg_train_loss:.4f}")
        print(f"   Time: {epoch_time:.1f}s")
        print(f"   LR: {current_lr:.2e}")
        
        loss_log.append(avg_train_loss)
        
        # Validation
        if epoch % opt.eval_freq == 0:
            print(f" Running validation...")
            model.eval()
            
            with torch.no_grad():  # Disable gradients for validation
                val_start = time.time()
                dices, mean_dice, _, val_losses = get_eval(valloader, model, criterion=criterion, opt=opt, args=args)
                val_time = time.time() - val_start
            
            print(f" Validation Results:")
            print(f"   Val Loss: {val_losses:.4f}")
            print(f"   Val Dice: {mean_dice:.4f}")
            print(f"   Val Time: {val_time:.1f}s")
            
            dice_log.append(mean_dice)
            
            # WandB logging
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
                
            # Model saving
            if mean_dice > best_dice:
                if best_model_path and os.path.exists(best_model_path):
                    try:
                        os.remove(best_model_path)
                    except:
                        pass
                
                best_dice = mean_dice
                timestr = time.strftime('%m%d%H%M')
                
                if not os.path.isdir(opt.save_path):
                    os.makedirs(opt.save_path)
                
                best_model_path = os.path.join(
                    opt.save_path, 
                    f'{args.modelname}{opt.save_path_code}_best_{timestr}_epoch{epoch}_dice{best_dice:.4f}_finetune.pth'
                )
                
                # Save comprehensive checkpoint
                checkpoint_data = {
                    'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'best_dice': best_dice,
                    'loss_log': loss_log,
                    'dice_log': dice_log,
                    'args': vars(args),
                    'model_config': {
                        'modelname': args.modelname,
                        'encoder_input_size': args.encoder_input_size,
                        'low_image_size': args.low_image_size
                    }
                }
                
                torch.save(checkpoint_data, best_model_path)
                print(f"💾 NEW BEST MODEL SAVED!")
                print(f"   Dice: {best_dice:.4f}")
                print(f"   Path: {os.path.basename(best_model_path)}")
                
                if args.use_wandb:
                    wandb.log({"best_dice": best_dice, "best_epoch": epoch + 1})
                    # Save model artifact to WandB
                    try:
                        artifact = wandb.Artifact(f"model-{epoch}", type="model")
                        artifact.add_file(best_model_path)
                        wandb.log_artifact(artifact)
                    except Exception as e:
                        print(f"Warning: Could not save model artifact to WandB: {e}")
            
            # Early stopping check
            if early_stopping(mean_dice):
                print(f"🛑 Early stopping triggered at epoch {epoch + 1}")
                if args.use_wandb:
                    wandb.log({"early_stopped": True, "early_stop_epoch": epoch + 1})
                break
        else:
            dice_log.append(dice_log[-1] if dice_log else 0.0)
        
        # Progress summary
        total_elapsed = time.time() - training_start_time
        completed_epochs = (epoch + 1) - args.start_epoch
        remaining_epochs = opt.epochs - (epoch + 1)
        
        if completed_epochs > 0:
            avg_epoch_time = total_elapsed / completed_epochs
            eta = remaining_epochs * avg_epoch_time
            
            print(f"\n⏱  Progress: {((epoch + 1) / opt.epochs) * 100:.1f}% complete")
            print(f"   Elapsed: {total_elapsed/60:.1f} min")
            print(f"   ETA: {eta/60:.1f} min")
            print(f"   Best Dice: {best_dice:.4f}")
        
        # Traditional logging
        if args.keep_log:
            TensorWriter.add_scalar('train_loss', avg_train_loss, epoch)
            TensorWriter.add_scalar('learning_rate', current_lr, epoch)
            if epoch % opt.eval_freq == 0:
                TensorWriter.add_scalar('val_loss', val_losses, epoch)
                TensorWriter.add_scalar('val_dice', mean_dice, epoch)
    
    # Training completion
    total_time = time.time() - training_start_time
    final_epoch = epoch + 1
    completed_epochs = final_epoch - args.start_epoch
    
    print(f"\n{'='*80}")
    print(f"TRAINING COMPLETED!")
    print(f"{'='*80}")
    print(f"Epochs: {args.start_epoch + 1} → {final_epoch} (Completed: {completed_epochs})")
    print(f"⏱Total Time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    print(f"Avg Time/Epoch: {total_time/completed_epochs:.1f}s")
    print(f"Best Dice Score: {best_dice:.4f}")
    
    if best_model_path:
        print(f"Best Model: {os.path.basename(best_model_path)}")
    
    print(f"📊 Final Training Loss: {loss_log[-1]:.4f}")
    print(f"📈 Final Validation Dice: {dice_log[-1]:.4f}")
    print(f"{'='*80}")
    
    # Final WandB logging
    if args.use_wandb:
        wandb.log({
            "training_completed": True,
            "total_training_time": total_time,
            "total_epochs_completed": completed_epochs,
            "final_train_loss": loss_log[-1],
            "final_val_dice": dice_log[-1],
            "best_dice_final": best_dice
        })
        
        # Create summary table
        summary_data = {
            "Total Training Time (hours)": f"{total_time/3600:.2f}",
            "Epochs Completed": completed_epochs,
            "Best Dice Score": f"{best_dice:.4f}",
            "Final Train Loss": f"{loss_log[-1]:.4f}",
            "Final Val Dice": f"{dice_log[-1]:.4f}",
            "Model Path": os.path.basename(best_model_path) if best_model_path else "None"
        }
        
        wandb.summary.update(summary_data)
        print("📊 WandB summary updated!")
        
        # Finish WandB run
        wandb.finish()
        print("✅ WandB run completed!")
    
    # Save final training logs
    if args.keep_log:
        log_data = {
            'loss_log': loss_log,
            'dice_log': dice_log,
            'best_dice': best_dice,
            'total_time': total_time,
            'epochs_completed': completed_epochs,
            'args': vars(args)
        }
        
        timestr = time.strftime('%m%d%H%M')
        log_path = os.path.join(opt.save_path, f'training_log_{timestr}.pkl')
        
        try:
            import pickle
            with open(log_path, 'wb') as f:
                pickle.dump(log_data, f)
            print(f"📋 Training logs saved: {log_path}")
        except Exception as e:
            print(f"⚠️  Could not save training logs: {e}")
        
        # Close tensorboard writer
        TensorWriter.close()
        print("📊 TensorBoard logs closed!")
    
    # Memory cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("🧹 GPU memory cleared!")
    
    print(f"\n🚀 Training script completed successfully!")
    
    return {
        'best_dice': best_dice,
        'best_model_path': best_model_path,
        'loss_log': loss_log,
        'dice_log': dice_log,
        'total_time': total_time,
        'epochs_completed': completed_epochs
    }


if __name__ == '__main__':
    try:
        results = main()
        print(f"✅ Script finished with best dice: {results['best_dice']:.4f}")
    except KeyboardInterrupt:
        print(f"\n⏹  Training interrupted by user!")
        if 'args' in locals() and args.use_wandb:
            wandb.finish(exit_code=1)
    except Exception as e:
        print(f"\n Training failed with error: {str(e)}")
        if 'args' in locals() and args.use_wandb:
            wandb.finish(exit_code=1)
        raise