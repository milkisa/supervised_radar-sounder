# =========================================
# 🔎 Optuna: pick optimizer + learning rate
# =========================================

import os
os.environ['HOME'] = '/home/milkisayebasse/sparse/.cache'
print('yaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa||||||||||||||||||||||||||||||||||||')
import torch
import torchvision
import copy
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, utils
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from data_loader import ToTensorLab,SalObjDataset
from literature.aspp import UNetASPP
from implmentation.dataset import mc10_data_model
from implmentation.inputs import parse_args, apply_presets, build_model,muti_bce_loss_fusion, PRESETS
from supervised_foldtest import test
from implmentation.metrics import calc_metrics
from implmentation.metrics import calc_metrics
import time
import scipy.io as sio

import torchvision.transforms as T
from sklearn.model_selection import KFold
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
data = torch.load('/mnt/data/dataset/greenland_64multi.pt')  # load serialized dict with tensors
# data = data['test']  # (optional) example if your file has a split dict

# Move to CPU numpy arrays for the SalObjDataset path
rs_image = data['data'].to('cpu').numpy()   # images as numpy (H, W) or (N, H, W)
rs_label = data['label'].to('cpu').numpy()  # labels as numpy (matching image spatial size)
def make_dataloaders_simple(rs_image, rs_label, batch_size):
    # fixed split to match your current script
    train_slice = slice(0, 1500)
    val_slice   = slice(1500, 2100)

    train_ds = SalObjDataset(
        img_name_list=np.expand_dims(rs_image[train_slice], axis=-1),
        lbl_name_list=np.expand_dims(rs_label[train_slice], axis=-1),
        transform=transforms.Compose([ToTensorLab(flag=0)])
    )
    val_ds = SalObjDataset(
        img_name_list=np.expand_dims(rs_image[val_slice], axis=-1),
        lbl_name_list=np.expand_dims(rs_label[val_slice], axis=-1),
        transform=transforms.Compose([ToTensorLab(flag=0)])
    )
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=1)
    val_dl   = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=1)
    return train_dl, val_dl

def train_one(args, train_dl, val_dl, device, eval_every=10):
    """Train once and return the best validation F1 achieved (no saving)."""
    model_kwargs, criterion = apply_presets(args)
    net = build_model(args, model_kwargs).to(device)

    # pick optimizer
    if args.optimizer == "Adam":
        optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "AdamW":
        optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "SGD":
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, nesterov=True, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer {args.optimizer}")

    best_f1 = 0.0
    for epoch in range(0, args.epochs + 1):
        net.train()
        for batch in train_dl:
            x, y = batch['image'], batch['label']
            x = x.type(torch.FloatTensor).to(device)
            y = y.type(torch.FloatTensor).to(device)

            optimizer.zero_grad()
            if args.model == 'eu':
                d0, d1, d2, d3, d4, d5, d6 = net(x)
                _, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, y)
                del d0, d1, d2, d3, d4, d5, d6
            else:
                logits = net(x)
                loss = criterion(logits, y.squeeze(1).long())
            loss.backward()
            optimizer.step()
            del loss

        if epoch % eval_every == 0:
            with torch.no_grad():
                rspred, rs_lab = test(val_dl, net, device, 1, case='val', model_name=args.model)
                _, _, _, _, _, _, avg_f1 = calc_metrics(rspred, rs_lab)
                best_f1 = max(best_f1, float(avg_f1))

    # cleanup
    del net, optimizer
    torch.cuda.empty_cache()
    return best_f1

def optuna_objective(trial, rs_image, rs_label, base_args, device):
    # choose optimizer first
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "AdamW", "SGD"])

    # optimizer-specific spaces
    if optimizer_name in ["Adam", "AdamW"]:
        lr = trial.suggest_float("lr", 1e-5, 5e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-7, 1e-3, log=True)
        momentum = 0.0  # unused
    else:  # SGD
        lr = trial.suggest_float("lr", 1e-3, 5e-1, log=True)
        momentum = trial.suggest_float("momentum", 0.7, 0.99)
        weight_decay = trial.suggest_float("weight_decay", 1e-7, 1e-3, log=True)

    batch_size = trial.suggest_categorical("batch_size_train", [2, 4, 8, 16])
    epochs     = trial.suggest_int("epochs", 60, 160, step=20)
    eval_every = trial.suggest_categorical("eval_every", [5, 10, 20])

    # clone your args and override
    args = parse_args()
    args.optimizer = optimizer_name
    args.lr = lr
    args.weight_decay = weight_decay
    args.batch_size_train = batch_size
    args.epochs = epochs
    args.momentum = momentum

    train_dl, val_dl = make_dataloaders_simple(rs_image, rs_label, batch_size)
    best_f1 = train_one(args, train_dl, val_dl, device, eval_every=eval_every)

    trial.set_user_attr("final_f1", best_f1)
    return best_f1

def pick_optimizer_and_lr(rs_image, rs_label, n_trials=30, study_name="pick_opt_lr"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_args = parse_args()  # we only use its model/preset hooks; hyperparams get overridden

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=42, n_startup_trials=8),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=2),
        study_name=study_name
    )
    study.optimize(lambda t: optuna_objective(t, rs_image, rs_label, base_args, device),
                   n_trials=n_trials, gc_after_trial=True)

    print("\n=== Best settings (no checkpoint saved) ===")
    print(f" Best mean Val F1: {study.best_value:.4f}")
    for k, v in study.best_trial.params.items():
        print(f"  - {k}: {v}")
    return study.best_trial.params, study.best_value

# —— Run the search (example) ——
best_params, best_score = pick_optimizer_and_lr(rs_image, rs_label, n_trials=40)
# You can now set these into your args before your normal training:
# args = parse_args()
# args.optimizer = best_params["optimizer"]
# args.lr = best_params["lr"]
# args.weight_decay = best_params["weight_decay"]
# if args.optimizer == "SGD":
#     args.momentum = best_params["momentum"]
# args.batch_size_train = best_params["batch_size_train"]
# args.epochs = best_params["epochs"]
