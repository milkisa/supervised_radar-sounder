import argparse, json, importlib, re
import torch.nn as nn
import torch.optim as optim
import torch.nn as nn
# --- import your models exactly as you already have them ---
from literature.aspp import UNetASPP

from literature.eu2net import EU2NET, EU2NETP
from literature.u2net import U2NET, U2NETP
from literature.sounder import Decoder
from literature.unet import UNet  # <- if you already have a UNet; else remove
import torch
def parse_args():
    p = argparse.ArgumentParser()
    # core toggles (start as None so presets can fill them)
    p.add_argument("--model", type=str, default="unetaspp",
                   choices=["aspp", "eu", "unet", "transsounder", "u2net"],)
    p.add_argument("--in-ch", type=int, default=1)
    p.add_argument("--num-classes", type=int, default=6)

    # hyperparams (None means “use model preset”)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size-train", type=int, default=None)
    p.add_argument("--batch-size-val", type=int, default=None)
    p.add_argument("--betas", type=str, default=None, help='e.g. "[0.9,0.999]"')
    p.add_argument("--eps", type=float, default=None)

    # model-specific kwargs (optional overrides)
    p.add_argument("--model-args", type=str, default="{}",
                   help='JSON dict for constructor kwargs, e.g. {"hc":512}')

    # io
    p.add_argument("--save-dir", type=str, default="/mnt/data/supervised/elena_aspp/")
    p.add_argument("--tag", type=str, default="run")
    return p.parse_args()
ce_loss = nn.CrossEntropyLoss(size_average=True, ignore_index=0)
def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):
        # print(d0.shape)
        # print(labels_v.shape)
        labels_v= labels_v.squeeze(1)
        # print(labels_v.shape)
        # labels_v= torch.tensor(labels_v, dtype=torch.long)
        labels_v= labels_v.type(torch.long)
        # print(labels_v)
        # print(d0)
        

        loss0 = ce_loss(d0, labels_v)
    
        loss1 = ce_loss(d1, labels_v)
        loss2 = ce_loss(d2, labels_v)
        loss3 = ce_loss(d3, labels_v)
        loss4 = ce_loss(d4, labels_v)
        loss5 = ce_loss(d5, labels_v)
        loss6 = ce_loss(d6, labels_v)

        loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
      
        return loss0, loss
# --------- per-model presets ----------
PRESETS = {
    "aspp": {
        "lr": 1e-4,
        "weight_decay": 0.0,
        "epochs": 200,
        "batch_size_train": 16,
        "batch_size_val": 1,
        "model_kwargs": {"hc": 128},  # UNetASPP(in_channels, out_channels, hc=...)
        "criterion": nn.CrossEntropyLoss(),
        "model_dir": "/mnt/data/supervised/aspp/"
    },
    "eu": {
        "lr": 0.000031,
        "weight_decay": 0,
        "epochs": 400,
        "batch_size_train": 16,
        "batch_size_val": 1,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "model_dir": "/mnt/data/supervised/eu/"
    },
    "unet": {
        "lr": 1e-4,
        "weight_decay": 0.0,
        "epochs": 200,
        "batch_size_train": 16,
        "batch_size_val": 1,
        
        "model_kwargs": {"hidden_channels": 128},  # UNet(in_channels, num_classes, base=...)
        "criterion": nn.CrossEntropyLoss(),
        "model_dir": "/mnt/data/supervised/unet/"
    },
    "u2net": {
        "lr": 0.001,
        "weight_decay": 0.0,
        "betas": [0.9, 0.999],
        "eps": 1e-08,
        "epochs": 400,
        "batch_size_train": 16,
        "batch_size_val": 1,
        "model_dir": "/mnt/data/supervised/u2net/"
    },
    "transsounder": {
        "lr": 1e-5,
        "weight_decay": 0.0,
        "epochs": 200,
        "batch_size_train": 4,
        "batch_size_val": 1,

        "criterion": nn.CrossEntropyLoss(),
        "model_dir": "/mnt/data/supervised/transsounder/"
    },
}

def apply_presets(args):
    preset = PRESETS[args.model]

    # helper to safely read preset keys
    def get(k, default=None):
        return preset[k] if k in preset else default

    # fill missing args only if user didn’t override them
    if args.lr is None: args.lr = get("lr")
    if args.weight_decay is None: args.weight_decay = get("weight_decay", 0.0)
    if args.epochs is None: args.epochs = get("epochs", 40)
    if args.batch_size_train is None: args.batch_size_train = get("batch_size_train", 4)
    if args.batch_size_val is None: args.batch_size_val = get("batch_size_val", 1)
    if args.betas is None and get("betas") is not None: args.betas = get("betas")
    elif isinstance(args.betas, str):  # user gave something like "[0.9,0.999]"
        import json
        args.betas = json.loads(args.betas)
    if args.eps is None and "eps" in preset:
        args.eps = preset["eps"]

    # merge model_kwargs (preset + CLI)
    import json
    mk = dict(get("model_kwargs", {}))
    mk.update(json.loads(args.model_args))

    # get criterion if available
    criterion = get("criterion", nn.CrossEntropyLoss())

    return mk, criterion


def build_model(args, model_kwargs):
    if args.model == "aspp":
        return UNetASPP(in_channels=args.in_ch, out_channels=args.num_classes, **model_kwargs)
    elif args.model == "unet":
        return UNet(in_channels=args.in_ch, out_channels=args.num_classes, **model_kwargs)
    elif args.model == "eu":
        return EU2NETP(args.in_ch, args.num_classes)
    elif args.model == "u2net":
        return U2NETP(args.in_ch, args.num_classes)
    elif args.model == "transsounder":
        return Decoder(0.5, 0.3, 0.2, args.num_classes)
    else:
        raise ValueError(args.model)
