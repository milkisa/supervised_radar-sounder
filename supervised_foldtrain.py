
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
seed = 42
np.random.seed(seed)
rs_image, rs_label, folds, _ = mc10_data_model()



#kf.split(rs_image)
for fold in folds:
    print(f"\nFold {fold['fold']}")
    
    # Split images and labels into train/test for the current fold
    train_images, val_images, test_images = rs_image[fold['train_idx']], rs_image[fold['val_idx']], rs_image[fold['test_idx']]
    train_labels, val_labels,  test_labels = rs_label[fold['train_idx']], rs_label[fold['val_idx']], rs_label[fold['test_idx']]
    
    # Display the shapes of the training and testing data
    print("Train Images shape:", train_images.shape)
    print("Train val shape:", val_images.shape)
    print("Train Labels shape:", train_labels.shape)


    rs_image_fold= np.expand_dims(train_images, axis=-1)
    rs_label_fold= np.expand_dims(train_labels, axis=-1)

    start_time= time.time()
    args = parse_args()
    model_kwargs, criterion = apply_presets(args)
    net = build_model(args, model_kwargs)
    opt_kwargs = {
    "lr": args.lr,
    "weight_decay": args.weight_decay,
    }
    preset = PRESETS[args.model]
    save_dir = preset.get("model_dir") 
    # make sure the folder exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    # Only add betas/eps if they exist
    if getattr(args, "betas", None) is not None:
        opt_kwargs["betas"] = tuple(args.betas)
    if getattr(args, "eps", None) is not None:
        opt_kwargs["eps"] = args.eps
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    optimizer = optim.Adam(net.parameters(), **opt_kwargs)
    print(f"[OK] Model created: {args.model} with {sum(p.numel() for p in net.parameters() if p.requires_grad)} params")
    print(f"[OK] Optimizer created with: {opt_kwargs}")
    print("---start training...")
    ite_num = 0
    running_loss = 0.0
    running_tar_loss = 0.0
    ite_num4val = 0
    epoch_num = args.epochs
    batch_size_train = args.batch_size_train
    salobj_dataset = SalObjDataset(
        img_name_list=rs_image_fold,
        lbl_name_list= rs_label_fold,
        transform=transforms.Compose([
            # RescaleT(288),
            ToTensorLab(flag=0)]))
    salobj_dataloader = DataLoader(salobj_dataset, batch_size=batch_size_train, shuffle=True, num_workers=1)
    val_salobj_dataset = SalObjDataset(img_name_list = np.expand_dims(val_images, axis=-1),
                                            lbl_name_list= np.expand_dims(val_labels, axis=-1),
                                            # lbl_name_list = [],
                                            transform=transforms.Compose([
                                                                        ToTensorLab(flag=0)])
                                            )
    val_salobj_dataloader = DataLoader(val_salobj_dataset,
                                            batch_size=1,
                                            shuffle=False,
                                            num_workers=1)
            
    net.train()
    best_val_f1 = 0.0
    best_state_dict = None  # will hold a deepcopy of the best weights

    for epoch in range(0, epoch_num+1):
        net.train()
        for i, data in enumerate(salobj_dataloader):
            ite_num = ite_num + 1
            ite_num4val = ite_num4val + 1

            inputs, labels = data['image'], data['label']
            """
            if args.model== 'transsounder':
                         inputs = F.interpolate(inputs, size=(400, 400), mode='bilinear', align_corners=False)
            """
            inputs = inputs.type(torch.FloatTensor)
            #inputs = inputs.repeat(1, 3, 1, 1)
            labels = labels.type(torch.FloatTensor)
            inputs_v, labels_v = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            

            if args.model == 'eu':
                d0, d1, d2, d3, d4, d5, d6 = net(inputs_v)
                _, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v)
                del d0, d1, d2, d3, d4, d5, d6
            else:

                output= net(inputs_v)
   
                pred = output # Extract the 'out' tensor from the OrderedDic
                labels_v = labels_v.squeeze(1)  # Shape: [batch_size, height, width]
                labels_v = labels_v.long()  # Converts to torch.int64
                loss = criterion(pred, labels_v)


            loss.backward()
            optimizer.step()

            # # print statistics
            running_loss += loss.data.item()

            # del temporary outputs and loss
            del  loss

            

        if epoch % 20 == 0:
            
            avg_loss = running_loss / max(1, ite_num4val)
            avg_tar  = running_tar_loss / max(1, ite_num4val)
            rspred, rs_lab= test(val_salobj_dataloader, net, device, fold['fold'], case='val', model_name =args.model)
            avg_recall, avg_precision, f1_scores, avg_accuracy, avg_iou, avg_class_oa, average_f1 = calc_metrics(rspred, rs_lab)
            # ----- Build safe save path -----
            if average_f1 > best_val_f1:
                best_val_f1 = average_f1
                best_epoch = epoch
                best_avg_accuracy = avg_accuracy
                best_state_dict = copy.deepcopy(net.state_dict())  # cache weights (no disk I/O)
                print(f"  >> Val @ epoch {epoch:03d}: acc={avg_accuracy:.4f}, f1={average_f1:.4f}")
                


        # reset trackers
        running_loss = 0.0
        running_tar_loss = 0.0
        ite_num4val = 0
    print(f"=== Fold {fold['fold']} training complete in {(time.time()-start_time)/60:.2f} mins ===")
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    ckpt_name = (
        f"greenland_{args.model}_fold{fold['fold']}_epoch{best_epoch}"
        f"_valf1_{best_val_f1:.4f}_time{time.time()-start_time:.1f}_{timestamp}.pth"
    )

    print(f"  >> Val @ epoch {best_epoch:03d}: acc={best_avg_accuracy:.4f}, f1={best_val_f1:.4f}")


    ckpt_path = os.path.join(save_dir, ckpt_name)
    torch.save(best_state_dict, ckpt_path)
    print(f"  >> ✅ Saved (best val f1 so far: {best_val_f1:.4f}) to: {ckpt_path}")


        


        

       
       
        


