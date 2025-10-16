
import os
os.environ['HOME'] = '/home/milkisayebasse/sparse/.cache'
print('yaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa||||||||||||||||||||||||||||||||||||')
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
from data_loader import ToTensorLab,SalObjDataset
from literature.aspp import UNetASPP
from implmentation.dataset import mc10_data_model, sharad_data_model
from implmentation.inputs import parse_args, apply_presets, build_model,muti_bce_loss_fusion, PRESETS
import time
import scipy.io as sio

import torchvision.transforms as T
from sklearn.model_selection import KFold
seed = 42
np.random.seed(seed)
rs_image, rs_label, folds, _ = mc10_data_model()



#kf.split(rs_image)
for fold, (train_index, test_index) in enumerate(folds):
    print(f"\nFold {fold + 1}")
    
    # Split images and labels into train/test for the current fold
    train_images, test_images = rs_image[train_index], rs_image[test_index]
    train_labels, test_labels = rs_label[train_index], rs_label[test_index]
    
    # Display the shapes of the training and testing data
    print("Train Images shape:", train_images.shape)
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
    save_frq = 20000 # save the model every 2000 iterations
    epoch_num = 5000
    batch_size_train = 1
    salobj_dataset = SalObjDataset(
        img_name_list=rs_image_fold,
        lbl_name_list= rs_label_fold,
        transform=transforms.Compose([
            # RescaleT(288),
            ToTensorLab(flag=0)]))
    salobj_dataloader = DataLoader(salobj_dataset, batch_size=batch_size_train, shuffle=True, num_workers=1)
            
    net.train()
    for epoch in range(0, epoch_num):
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

            

            if ite_num % save_frq == 0:
              
                avg_loss = running_loss / max(1, ite_num4val)
                avg_tar  = running_tar_loss / max(1, ite_num4val)

                print(
                    f"[epoch: {epoch+1:03d}/{epoch_num:03d}, "
                    f"ite: {ite_num:05d}] train loss: {avg_loss:.6f}, tar: {avg_tar:.6f}"
                )

                # ----- Build safe save path -----
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                ckpt_name = (
                    f"greenland_{args.model}_fold{fold}_iter{ite_num}_epoch{epoch}"
                    f"_loss{avg_loss:.6f}_time{time.time()-start_time:.1f}_{timestamp}.pth"
                )
                ckpt_path = os.path.join(save_dir, ckpt_name)

                # ----- Save -----
                torch.save(net.state_dict(), ckpt_path)


                # reset trackers
                running_loss = 0.0
                running_tar_loss = 0.0
                ite_num4val = 0



        


        

       
       
        


