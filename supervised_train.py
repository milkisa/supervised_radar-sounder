# ---------------------------
# Imports & environment setup
# ---------------------------
import os
os.environ['HOME'] = '/home/milkisayebasse/sparse/.cache'  # set HOME so any code using ~/.cache writes here

import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, utils
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

# Custom dataset utilities (tensor conversion & dataset wrapper)
from data_loader import ToTensorLab, SalObjDataset

# One of the available model definitions (not necessarily used unless selected by args)
from literature.aspp import UNetASPP

# Dataset loaders (mc10/sharad) – here you load from a .pt instead, but imports kept for flexibility
from implmentation.dataset import mc10_data_model
# Experiment configuration & model/loss builders
from implmentation.inputs import parse_args, apply_presets, build_model, muti_bce_loss_fusion, PRESETS
from implmentation.metrics import calc_metrics
import time
import scipy.io as sio
import torchvision.transforms as T
from sklearn.model_selection import KFold
from supervised_foldtest import test
# ---------------------------
# Load dataset from a .pt file
# ---------------------------
data = torch.load('/mnt/data/dataset/greenland_250multi.pt')  # load serialized dict with tensors
# data = data['test']  # (optional) example if your file has a split dict

# Move to CPU numpy arrays for the SalObjDataset path
rs_image = data['data'].to('cpu').numpy()   # images as numpy (H, W) or (N, H, W)
rs_label = data['label'].to('cpu').numpy()  # labels as numpy (matching image spatial size)

# Add a channel dimension (expects [N, H, W, 1])
rs_image_fold = np.expand_dims(rs_image, axis=-1)

# ⚠️ Likely a bug: you expand labels from *image* again.
# You probably intended: rs_label_fold = np.expand_dims(rs_label, axis=-1)
rs_label_fold = np.expand_dims(rs_label, axis=-1)  

# ---------------------------
# Build args, model, optimizer
# ---------------------------
start_time = time.time()

args = parse_args()                      # parse CLI args/presets selection
model_kwargs, criterion = apply_presets(args)  # get model hyperparams and loss function based on preset
net = build_model(args, model_kwargs)    # instantiate the selected model

# Optimizer kwargs from args (lr, weight_decay always; betas/eps optional)
opt_kwargs = {
    "lr": args.lr,
    "weight_decay": args.weight_decay,
}

preset = PRESETS[args.model]             # get current model preset dict
save_dir = preset.get("model_dir")       # where to save checkpoints

# Ensure checkpoint directory exists
if not os.path.exists(save_dir):
    os.makedirs(save_dir, exist_ok=True)

# Add optional Adam hyperparameters if present in args
if getattr(args, "betas", None) is not None:
    opt_kwargs["betas"] = tuple(args.betas)
if getattr(args, "eps", None) is not None:
    opt_kwargs["eps"] = args.eps

# Choose device (CUDA if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net.to(device)

# Create optimizer
optimizer = optim.Adam(net.parameters(), **opt_kwargs)

# Simple run header logs
print(f"[OK] Model created: {args.model} with {sum(p.numel() for p in net.parameters() if p.requires_grad)} params")
print(f"[OK] Optimizer created with: {opt_kwargs}")
print("---start training...")

# ---------------------------
# Training config & dataloader
# ---------------------------
ite_num = 0                # global iteration counter
running_loss = 0.0         # running average loss accumulator
running_tar_loss = 0.0     # (unused in this script) placeholder for target loss if you split losses
ite_num4val = 0            # iterations since last log/save
epoch_num = args.epochs          # number of epochs
batch_size_train = 8       # batch size (dataset/arch suggests 1)

# Wrap numpy arrays in your custom  validaiondataset (handles tensor conversion, dtype, etc.)
salobj_dataset = SalObjDataset(
    img_name_list=rs_image_fold[:400],  # expects array-like with last dim channel
    lbl_name_list=rs_label_fold[:400],  # same shape convention as images
    transform=transforms.Compose([
        # RescaleT(288),  # (optional) if you need resizing; currently commented out
        ToTensorLab(flag=0)       # custom transform to produce torch tensors
    ])
)

# Dataloader shuffles each epoch; num_workers=1 is safe for small setups / Windows
salobj_dataloader = DataLoader(
    salobj_dataset,
    batch_size=1,
    shuffle=True,
    num_workers=1
)
val_salobj_dataset = SalObjDataset(    img_name_list=rs_image_fold[400:500],  # expects array-like with last dim channel
                                            lbl_name_list=rs_label_fold[400:500],  # same shape convention as images
                                            # lbl_name_list = [],
                                            transform=transforms.Compose([
                                                                        ToTensorLab(flag=0)])
                                            )
val_salobj_dataloader = DataLoader(val_salobj_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=1)

# ---------------------------
# Training loop
# ---------------------------
net.train()
best_val_f1=0.0
for epoch in range(0, epoch_num+1):
    net.train()  # ensure train mode (dropout/bn)
    for i, data in enumerate(salobj_dataloader):
        ite_num += 1
        ite_num4val += 1

        # Unpack a batch from dataset
        inputs, labels = data['image'], data['label']

        # If using TransSounder you may want to resize inputs to a fixed size
        """
        if args.model == 'transsounder':
            inputs = F.interpolate(inputs, size=(400, 400), mode='bilinear', align_corners=False)
        """

        # Cast to float32 and move to target device
        inputs = inputs.type(torch.FloatTensor).to(device)
        labels = labels.type(torch.FloatTensor).to(device)

        # Clear gradients
        optimizer.zero_grad()

        # -----------------------
        # Forward & loss
        # -----------------------
        if args.model == 'eu':
            # Efficient U²-Net returns multiple side outputs (d0..d6)
            d0, d1, d2, d3, d4, d5, d6 = net(inputs)
            # muti_bce_loss_fusion should return (y_pred, loss) according to your usage
            _, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels)
            # free references to intermediate outputs
            del d0, d1, d2, d3, d4, d5, d6
        else:
            # Single-output models: UNet, UNetASPP, TransSounder
            output = net(inputs)

            # Pred logits (B, C, H, W); CrossEntropy expects Long labels of shape (B, H, W)
            pred = output
            loss = criterion(pred, labels.squeeze(1).long())

        # Backprop & update
        loss.backward()
        optimizer.step()

        # -----------------------
        # Track loss
        # -----------------------
        running_loss += loss.data.item()

        # Drop the scalar to free graph
        del loss

        # -----------------------
        # Periodic checkpointing
        # -----------------------
    if epoch % 10 == 0:
            
        avg_loss = running_loss / max(1, ite_num4val)
        avg_tar  = running_tar_loss / max(1, ite_num4val)
        rspred, rs_lab= test(val_salobj_dataloader, net, device, 1 , case='val', model_name =args.model)
  
        avg_recall, avg_precision, f1_scores, avg_accuracy, avg_iou, avg_class_oa, average_f1 = calc_metrics(rspred, rs_lab)
        # ----- Build safe save path -----
        if average_f1 > best_val_f1:
            best_val_f1 = average_f1
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            ckpt_name = (
                f"greenland_{args.model}__epoch{epoch}"
                f"_valf1{average_f1:.4f}_time{time.time()-start_time:.1f}_{timestamp}.pth"
            )

            print(f"  >> Val @ epoch {epoch:03d}: acc={avg_accuracy:.4f}, f1={average_f1:.4f}")


            ckpt_path = os.path.join(save_dir, ckpt_name)
            torch.save(net.state_dict(), ckpt_path)
            print(f"  >> ✅ Saved (best val f1 so far: {best_val_f1:.4f}) to: {ckpt_path}")


    # reset trackers
    running_loss = 0.0
    running_tar_loss = 0.0
    ite_num4val = 0