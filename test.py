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
import cv2

from torch.nn.modules.loss import CrossEntropyLoss
from monai.losses import DiceCELoss
from einops import rearrange
from models.model_dict import get_model
from utils.data_us import JointTransform2D, ImageToImage2D
from utils.loss_functions.sam_loss import get_criterion
from utils.generate_prompts import get_click_prompt
from thop import profile


def main():

    #  =========================================== parameters setting ==================================================

    parser = argparse.ArgumentParser(description='Networks')
    parser.add_argument('--modelname', default='SAMUS', type=str, help='type of model, e.g., SAM, SAMFull, SAMHead, MSA, SAMed, SAMUS...')
    parser.add_argument('-encoder_input_size', type=int, default=256, help='the image size of the encoder input, 1024 in SAM and MSA, 512 in SAMed, 256 in SAMUS') 
    parser.add_argument('-low_image_size', type=int, default=128, help='the image embedding size, 256 in SAM and MSA, 128 in SAMed and SAMUS') 
    parser.add_argument('--task', default='BUSI', help='task or dataset name')
    parser.add_argument('--vit_name', type=str, default='vit_b', help='select the vit model for the image encoder of sam')
    parser.add_argument('--sam_ckpt', type=str, default='/Users/lakshitvedant/Desktop/main/checkpoints/sam_vit_b_01ec64.pth', help='Pretrained checkpoint of SAM')
    parser.add_argument('--batch_size', type=int, default=8, help='batch_size per gpu') # 8 # SAMed is 12 bs with 2n_gpu and lr is 0.005
    parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
    parser.add_argument('--base_lr', type=float, default=0.0001, help='segmentation network learning rate, 0.005 for SAMed, 0.0001 for MSA') #0.0006
    parser.add_argument('--warmup', type=bool, default=False, help='If activated, warp up the learning from a lower lr to the base_lr') # True
    parser.add_argument('--warmup_period', type=int, default=250, help='Warp up iterations, only valid whrn warmup is activated')
    parser.add_argument('-keep_log', type=bool, default=False, help='keep the loss&lr&dice during training or not')
    
    # NEW: Add GradCAM specific arguments
    parser.add_argument('--enable_gradcam', action='store_true', default=True, help='Enable GradCAM visualization')
    parser.add_argument('--gradcam_output_dir', type=str, default='./gradcam_results-test6', help='Directory to save GradCAM visualizations')
    parser.add_argument('--gradcam_threshold', type=float, default=0.5, help='Threshold (0-1) to binarize GradCAM for prompt generation')
    

    args = parser.parse_args()
    opt = get_config(args.task)  # please configure your hyper-parameter
    print("task", args.task, "checkpoints:", opt.load_path) 
    opt.mode = "val"
    #opt.classes=2
    opt.visual = True
    
    # MODIFIED: Enable GradCAM visualization based on argument
    opt.gradcam_visualization = True
    
    # MODIFIED: Set custom result path for GradCAM if specified
    if args.gradcam_output_dir:
        opt.result_path = args.gradcam_output_dir
    else:
        opt.result_path = "./visualization_results10"
    
    # Create the output directory if it doesn't exist
    os.makedirs(opt.result_path, exist_ok=True)
    
    # opt.eval_mode = "patient"
    opt.modelname = args.modelname
    device = torch.device(opt.device)
    
    # NEW: Print GradCAM status
    if opt.gradcam_visualization:
        print(f"GradCAM visualization enabled. Results will be saved to: {opt.result_path}")
    else:
        print("GradCAM visualization disabled.")

     #  =============================================================== add the seed to make sure the results are reproducible ==============================================================

    seed_value = 300 # the number of seed
    np.random.seed(seed_value)  # set random seed for numpy
    random.seed(seed_value)  # set random seed for python
    os.environ['PYTHONHASHSEED'] = str(seed_value)  # avoid hash random
    torch.manual_seed(seed_value)  # set random seed for CPU
    torch.cuda.manual_seed(seed_value)  # set random seed for one GPU
    torch.cuda.manual_seed_all(seed_value)  # set random seed for all GPU
    torch.backends.cudnn.deterministic = True  # set random seed for convolution

    #  =========================================================================== model and data preparation ============================================================================
    
    # register the sam model
    opt.batch_size = args.batch_size * args.n_gpu

    tf_val = JointTransform2D(img_size=args.encoder_input_size, low_img_size=args.low_image_size, ori_size=opt.img_size, crop=opt.crop, p_flip=0, color_jitter_params=None, long_mask=True)
    val_dataset = ImageToImage2D(opt.data_path, opt.test_split, tf_val, img_size=args.encoder_input_size, class_id=1)  # return image, mask, and filename
    valloader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    if args.modelname=="SAMed":
        opt.classes=2
    model = get_model(args.modelname, args=args, opt=opt)
    model.to(device)
    model.train()

    checkpoint = torch.load(opt.load_path, map_location=device)
    #------when the load model is saved under multiple GPU
    new_state_dict = {}
    for k,v in checkpoint.items():
        if k[:7] == 'module.':
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    

    optimizer = optim.Adam(model.parameters(), lr=args.base_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    #optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=b_lr, betas=(0.9, 0.999), weight_decay=0.1)
    
    criterion = get_criterion(modelname=args.modelname, opt=opt)

#  ========================================================================= begin to evaluate the model ============================================================================

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total_params: {}".format(pytorch_total_params))
    input = torch.randn(1, 1, args.encoder_input_size, args.encoder_input_size).to(device)
    points = (
    torch.tensor([[[1, 2]]]).float().to(device),
    torch.tensor([[1]]).float().to(device)
)

    flops, params = profile(model, inputs=(input, points), )
    print('Gflops:', flops/1000000000, 'params:', params)
    print(opt.eval_mode)

    model.eval()

    if opt.mode == "train":
        dices, mean_dice, _, val_losses = get_eval(valloader, model, criterion=criterion, opt=opt, args=args)
        print("mean dice:", mean_dice)
    else:
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
        
        # NEW: Print GradCAM completion message
        if opt.gradcam_visualization:
            gradcam_dir = os.path.join(opt.result_path, f"gradcam-{opt.modelname}")
            print(f"\nGradCAM visualizations saved to: {gradcam_dir}")

        # with open("experiments.txt", "a+") as file:
        #     file.write(args.task + " " + args.modelname + "-pt10 " + '%.2f'%(mean_dice[1]) + "±" + '%.2f'%std_dice[1] + " ")
        #     file.write('%.2f'%mean_hdis[1] + "±" + '%.2f'%std_hdis[1] + " ")
        #     file.write('%.2f'%(mean_iou[1]) + "±" + '%.2f'%std_iou[1] + " ")
        #     file.write('%.2f'%(mean_acc[1]) + "±" + '%.2f'%std_acc[1] + " ")
        #     file.write('%.2f'%(mean_se[1]) + "±" + '%.2f'%std_se[1] + " ")
        #     file.write('%.2f'%(mean_sp[1]) + "±" + '%.2f'%std_sp[1] + "\n")

        #     # file.write("MYO " + args.modelname + " " + '%.2f'%(mean_dice[2]) + "±" + '%.2f'%std_dice[2] + " ")
        #     # file.write('%.2f'%mean_hdis[2] + "±" + '%.2f'%std_hdis[2] + " ")
        #     # file.write('%.2f'%(mean_iou[2]) + "±" + '%.2f'%std_iou[2] + " ")
        #     # file.write('%.2f'%(mean_acc[2]) + "±" + '%.2f'%std_acc[2] + " ")
        #     # file.write('%.2f'%(mean_se[2]) + "±" + '%.2f'%std_se[2] + " ")
        #     # file.write('%.2f'%(mean_sp[2]) + "±" + '%.2f'%std_sp[2] + "\n")
        #     # file.write("LA " + args.modelname + " " + '%.2f'%(mean_dice[3]) + "±" + '%.2f'%std_dice[3] + " ")
        #     # file.write('%.2f'%mean_hdis[3] + "±" + '%.2f'%std_hdis[3] + " ")
        #     # file.write('%.2f'%(mean_iou[3]) + "±" + '%.2f'%std_iou[3] + " ")
        #     # file.write('%.2f'%(mean_acc[3]) + "±" + '%.2f'%std_acc[3] + " ")
        #     # file.write('%.2f'%(mean_se[3]) + "±" + '%.2f'%std_se[3] + " ")
        #     # file.write('%.2f'%(mean_sp[3]) + "±" + '%.2f'%std_sp[3] + "\n")
        #     file.close()

        ##
    
if __name__ == '__main__':
    main()