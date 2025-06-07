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


def main():

    #  ============================================================================= parameters setting ====================================================================================

    parser = argparse.ArgumentParser(description='Networks')
    parser.add_argument('--modelname', default='SAMUS', type=str, help='type of model, e.g., SAM, SAMFull, MedSAM, MSA, SAMed, SAMUS...')
    parser.add_argument('-encoder_input_size', type=int, default=256, help='the image size of the encoder input, 1024 in SAM and MSA, 512 in SAMed, 256 in SAMUS')
    parser.add_argument('-low_image_size', type=int, default=128, help='the image embedding size, 256 in SAM and MSA, 128 in SAMed and SAMUS')
    parser.add_argument('--task', default='US30K', help='task or dataset name')
    parser.add_argument('--vit_name', type=str, default='vit_b', help='select the vit model for the image encoder of sam')
    parser.add_argument('--sam_ckpt', type=str, default='checkpoints/sam_vit_b_01ec64.pth', help='Pretrained checkpoint of SAM')
    parser.add_argument('--batch_size', type=int, default=8, help='batch_size per gpu') # Set to 16 as requested
    parser.add_argument('--data_subset_ratio', type=float, default=0.02, help='Use only a fraction of training data (0.25 = 25% of data)')
    parser.add_argument('--val_subset_ratio', type=float, default=0.02, help='Use only a fraction of validation data (0.2 = 20% of data)')
    parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
    parser.add_argument('--base_lr', type=float, default=0.0005, help='segmentation network learning rate, 0.005 for SAMed, 0.0001 for MSA') #0.0006
    parser.add_argument('--warmup', type=bool, default=False, help='If activated, warp up the learning from a lower lr to the base_lr') 
    parser.add_argument('--warmup_period', type=int, default=250, help='Warp up iterations, only valid whrn warmup is activated')
    parser.add_argument('-keep_log', type=bool, default=False, help='keep the loss&lr&dice during training or not')
    parser.add_argument('--epochs', type=int, default=5, help='number of training epochs')

    args = parser.parse_args()
    opt = get_config(args.task) 
    
    # Override config epochs with command line argument
    if hasattr(opt, 'epochs'):
        opt.epochs = args.epochs
    else:
        # If opt doesn't have epochs attribute, add it
        setattr(opt, 'epochs', args.epochs)

    device = torch.device(opt.device)
    if args.keep_log:
        logtimestr = time.strftime('%m%d%H%M')  # initialize the tensorboard for record the training process
        boardpath = opt.tensorboard_path + args.modelname + opt.save_path_code + logtimestr
        if not os.path.isdir(boardpath):
            os.makedirs(boardpath)
        TensorWriter = SummaryWriter(boardpath)

    #  =============================================================== add the seed to make sure the results are reproducible ==============================================================

    seed_value = 1234  # the number of seed
    np.random.seed(seed_value)  # set random seed for numpy
    random.seed(seed_value)  # set random seed for python
    os.environ['PYTHONHASHSEED'] = str(seed_value)  # avoid hash random
    torch.manual_seed(seed_value)  # set random seed for CPU
    torch.cuda.manual_seed(seed_value)  # set random seed for one GPU
    torch.cuda.manual_seed_all(seed_value)  # set random seed for all GPU
    torch.backends.cudnn.deterministic = True  # set random seed for convolution

    #  =========================================================================== model and data preparation ============================================================================
    
    # register the sam model
    model = get_model(args.modelname, args=args, opt=opt)
    opt.batch_size = args.batch_size * args.n_gpu

    tf_train = JointTransform2D(img_size=args.encoder_input_size, low_img_size=args.low_image_size, ori_size=opt.img_size, crop=opt.crop, p_flip=0.0, p_rota=0.5, p_scale=0.5, p_gaussn=0.0,
                                p_contr=0.5, p_gama=0.5, p_distor=0.0, color_jitter_params=None, long_mask=True)  # image reprocessing
    tf_val = JointTransform2D(img_size=args.encoder_input_size, low_img_size=args.low_image_size, ori_size=opt.img_size, crop=opt.crop, p_flip=0, color_jitter_params=None, long_mask=True)
    train_dataset = ImageToImage2D(opt.data_path, opt.train_split, tf_train, img_size=args.encoder_input_size)
    val_dataset = ImageToImage2D(opt.data_path, opt.val_split, tf_val, img_size=args.encoder_input_size)  # return image, mask, and filename
    
    # NEW: Use subset of training data to reduce batches per epoch
    if args.data_subset_ratio < 1.0:
        subset_size = int(len(train_dataset) * args.data_subset_ratio)
        indices = torch.randperm(len(train_dataset))[:subset_size]
        train_dataset = torch.utils.data.Subset(train_dataset, indices)
        print(f"Using {args.data_subset_ratio*100}% of training data: {len(train_dataset)} samples")
    
    # NEW: Use subset of validation data to reduce validation time
    if args.val_subset_ratio < 1.0:
        val_subset_size = int(len(val_dataset) * args.val_subset_ratio)
        val_indices = torch.randperm(len(val_dataset))[:val_subset_size]
        val_dataset = torch.utils.data.Subset(val_dataset, val_indices)
        print(f"Using {args.val_subset_ratio*100}% of validation data: {len(val_dataset)} samples")
    
    trainloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    valloader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    model.to(device)
    if opt.pre_trained:
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
    
    # NEW: Enhanced progress tracking setup
    print(f"\n{'='*80}")
    print(f"TRAINING SETUP:")
    print(f"Total Epochs: {opt.epochs}")
    print(f"Batches per Epoch: {len(trainloader)}")
    print(f"Total Iterations: {opt.epochs * len(trainloader)}")
    print(f"Batch Size: {opt.batch_size}")
    print(f"Model: {args.modelname}")
    print(f"{'='*80}\n")

    #  ========================================================================= begin to train the model ============================================================================
    iter_num = 0
    max_iterations = opt.epochs * len(trainloader)
    best_dice, loss_log, dice_log = 0.0, np.zeros(opt.epochs+1), np.zeros(opt.epochs+1)
    
    # NEW: Track best model path to replace when better model is found
    best_model_path = None
    
    # NEW: Training start time
    training_start_time = time.time()
    
    for epoch in range(opt.epochs):
        #  --------------------------------------------------------- training ---------------------------------------------------------
        model.train()
        # if args.data_subset_ratio < 1.0:
        #     subset_size = int(len(train_dataset.dataset) * args.data_subset_ratio)
        #     indices = torch.randperm(len(train_dataset.dataset))[:subset_size]
        #     epoch_dataset = torch.utils.data.Subset(train_dataset.dataset, indices)
        # else:
        #     epoch_dataset = train_dataset

        # trainloader = DataLoader(epoch_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=8, pin_memory=True)

        train_losses = 0
        epoch_start_time = time.time()
        train_losses = 0
        epoch_start_time = time.time()
        
        # NEW: Enhanced progress tracking for each epoch
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch+1}/{opt.epochs} STARTED")
        print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")
        
        for batch_idx, (datapack) in enumerate(trainloader):
            imgs = datapack['image'].to(dtype = torch.float32, device=opt.device)
            masks = datapack['low_mask'].to(dtype = torch.float32, device=opt.device)
            bbox = torch.as_tensor(datapack['bbox'], dtype=torch.float32, device=opt.device)
            pt = get_click_prompt(datapack, opt)
            # -------------------------------------------------------- forward --------------------------------------------------------
            pred = model(imgs, pt, bbox)
            train_loss = criterion(pred, masks) 
            # -------------------------------------------------------- backward -------------------------------------------------------
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            train_losses += train_loss.item()
            
            # NEW: Enhanced batch progress tracking (reduced frequency)
            if (batch_idx + 1) % 50 == 0 or batch_idx == 0:
                elapsed_time = time.time() - epoch_start_time
                progress_percent = ((batch_idx + 1) / len(trainloader)) * 100
                avg_loss = train_losses / (batch_idx + 1)
                print(f"  Batch [{batch_idx+1:4d}/{len(trainloader):4d}] "
                      f"({progress_percent:5.1f}%) | "
                      f"Loss: {train_loss.item():.4f} | "
                      f"Avg Loss: {avg_loss:.4f} | "
                      f"Time: {elapsed_time:.1f}s")
            
            # ------------------------------------------- adjust the learning rate when needed-----------------------------------------
            if args.warmup and iter_num < args.warmup_period:
                lr_ = args.base_lr * ((iter_num + 1) / args.warmup_period)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            else:
                if args.warmup:
                    shift_iter = iter_num - args.warmup_period
                    assert shift_iter >= 0, f'Shift iter is {shift_iter}, smaller than zero'
                    lr_ = args.base_lr * (1.0 - shift_iter / max_iterations) ** 0.9  # learning rate adjustment depends on the max iterations
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_
            iter_num = iter_num + 1

        #  -------------------------------------------------- log the train progress --------------------------------------------------
        epoch_time = time.time() - epoch_start_time
        avg_train_loss = train_losses / (batch_idx + 1)
        
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch+1}/{opt.epochs} COMPLETED")
        print(f"Training Loss: {avg_train_loss:.4f}")
        print(f"Epoch Time: {epoch_time:.2f}s")
        print(f"Learning Rate: {optimizer.state_dict()['param_groups'][0]['lr']:.8f}")
        
        if args.keep_log:
            TensorWriter.add_scalar('train_loss', avg_train_loss, epoch)
            TensorWriter.add_scalar('learning rate', optimizer.state_dict()['param_groups'][0]['lr'], epoch)
            loss_log[epoch] = avg_train_loss

        #  --------------------------------------------------------- evaluation ----------------------------------------------------------
        if epoch % opt.eval_freq == 0:
            print(f"Running validation...")
            model.eval()
            val_start_time = time.time()
            dices, mean_dice, _, val_losses = get_eval(valloader, model, criterion=criterion, opt=opt, args=args)
            val_time = time.time() - val_start_time
            
            print(f"Validation Loss: {val_losses:.4f}")
            print(f"Validation Dice: {mean_dice:.4f}")
            print(f"Validation Time: {val_time:.2f}s")
            
            if args.keep_log:
                TensorWriter.add_scalar('val_loss', val_losses, epoch)
                TensorWriter.add_scalar('dices', mean_dice, epoch)
                dice_log[epoch] = mean_dice
                
            # MODIFIED: Only keep the best model, replace previous best if new one is better
            if mean_dice > best_dice:
                # Remove previous best model if it exists
                if best_model_path and os.path.exists(best_model_path):
                    try:
                        os.remove(best_model_path)
                        print(f"Removed previous best model: {os.path.basename(best_model_path)}")
                    except OSError as e:
                        print(f"Warning: Could not remove previous model file: {e}")
                
                # Update best dice score
                best_dice = mean_dice
                
                # Create new best model path
                timestr = time.strftime('%m%d%H%M')
                if not os.path.isdir(opt.save_path):
                    os.makedirs(opt.save_path)
                
                best_model_path = opt.save_path + args.modelname + opt.save_path_code + '_best_' + timestr + '_epoch' + str(epoch) + '_dice' + f'{best_dice:.4f}' + '.pth'
                
                # Save new best model
                torch.save(model.state_dict(), best_model_path, _use_new_zipfile_serialization=False)
                print(f"NEW BEST MODEL SAVED! Dice: {best_dice:.4f}")
                print(f"Saved as: {os.path.basename(best_model_path)}")
                
        # NEW: Overall progress tracking
        total_elapsed_time = time.time() - training_start_time
        remaining_epochs = opt.epochs - (epoch + 1)
        avg_epoch_time = total_elapsed_time / (epoch + 1)
        estimated_remaining_time = remaining_epochs * avg_epoch_time
        
        print(f"Progress: {((epoch + 1) / opt.epochs) * 100:.1f}% Complete")
        print(f"Total Elapsed: {total_elapsed_time/60:.1f} minutes")
        print(f"Estimated Remaining: {estimated_remaining_time/60:.1f} minutes")
        print(f"Best Dice So Far: {best_dice:.4f}")
        print(f"{'='*60}")
        
        # REMOVED: Regular epoch saving - only keep best model
        # The original code saved models every opt.save_freq epochs, this is now removed
        
        if args.keep_log:
            with open(opt.tensorboard_path + args.modelname + opt.save_path_code + logtimestr + '/trainloss.txt', 'w') as f:
                for i in range(len(loss_log)):
                    f.write(str(loss_log[i])+'\n')
            with open(opt.tensorboard_path + args.modelname + opt.save_path_code + logtimestr + '/dice.txt', 'w') as f:
                for i in range(len(dice_log)):
                    f.write(str(dice_log[i])+'\n')
    
    # NEW: Final training summary
    total_training_time = time.time() - training_start_time
    print(f"\n{'='*80}")
    print(f"TRAINING COMPLETED!")
    print(f"Total Training Time: {total_training_time/60:.2f} minutes ({total_training_time/3600:.2f} hours)")
    print(f"Final Best Dice Score: {best_dice:.4f}")
    print(f"Best Model Saved As: {os.path.basename(best_model_path) if best_model_path else 'No model saved'}")
    print(f"Total Epochs Completed: {opt.epochs}")
    print(f"Average Time per Epoch: {total_training_time/opt.epochs:.2f} seconds")
    print(f"{'='*80}")

if __name__ == '__main__':
    main()