import os
from skimage import io, transform
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms#, utils
# import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.image as mpimage
import numpy as np
from PIL import Image
import glob

import cv2
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from literature.u2net import U2NET # full size version 173.6 MB
from literature.u2net import U2NETP # small version u2net 4.7 MB
from skimage.transform import rotate

import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

import pandas as pd
import scipy.io as sio


from implmentation.output import save_output
from implmentation.metrics import calculate_recall_precision
from implmentation.metrics import average_recall_precision
from implmentation.metrics import calc_metrics
from implmentation.metrics import cv_calc
from implmentation.dataset import mc10_data_model,antarctica_datapatch_model,greenland_datapatch_model,sharad_datapatch_model,sharad_manual_data_model
from implmentation.merged import merge_and_resize_folds
from implmentation.inputs import parse_args, apply_presets, build_model,muti_bce_loss_fusion, PRESETS
seed = 42
np.random.seed(seed)
def  test(test_salobj_dataloader, net, device, fold, case='test',model_name= 'eu'):
    rs_pred = []
    rs_lab = []
    num=0
    for i_test, data_test in enumerate(test_salobj_dataloader):


                # print("inferencing:",img_name_list[i_test].split(os.sep)[-1])

                inputs_test = data_test['image'].to(device, dtype=torch.float32)
                labels= data_test['label']
                # print(type(inputs_test))

                if model_name == 'eu' or model_name == 'u2net':
                    
                    d1,d2,d3,d4,d5,d6,d7= net(inputs_test)

                    # normalization

                    pred = d1
                    del d1,d2,d3,d4,d5,d6,d7,
            
                else:
                    pred= net(inputs_test)


                p,l = save_output(inputs_test, pred,labels,num ,fold,case)
                
                rs_pred.append(p)

                rs_lab.append(l)

                num= num+1

                del p,l
    return rs_pred, rs_lab


   

p= 427
def main():
    all_fold_recalls = []
    all_fold_precisions = []
    all_fold_accuracies = []
    all_fold_f1 = []
    all_fold_ious = []
    all_fold_OAs = []   


    folds_a, model_dir = antarctica_datapatch_model()
    folds_g, _ = greenland_datapatch_model()
    folds_s, _ = sharad_datapatch_model()
    folds_s, _ = sharad_manual_data_model()
    merged_folds = merge_and_resize_folds([folds_a, folds_g, folds_s], target_h=800, target_w=64, shuffle=False, seed=42)
   # merged_folds= folds_s
    #print("Total merged folds:", len(merged_folds))

    #kf.split(rs_image)
    for fold in merged_folds:
        print(f"\nFold {fold['fold']}")
        
        # Split images and labels into train/test for the current fold
        train_images, val_images, test_images = fold['train_images'], fold['val_images'], fold['test_images']
        train_labels, val_labels,  test_labels = fold['train_labels'], fold['val_labels'], fold['test_labels']
        
        # Display the shapes of the training and testing data
        print("Train Images shape:", train_images.shape)
        print("Train val shape:", val_images.shape)
        print("test_images shape:", test_images.shape)

 


        test_salobj_dataset = SalObjDataset(img_name_list = test_images,
                                            lbl_name_list= test_labels,
                                            # lbl_name_list = [],
                                            transform=transforms.Compose([
                                                                        ToTensorLab(flag=0)])
                                            )
        test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                            batch_size=1,
                                            shuffle=False,
                                            num_workers=1)

        args = parse_args()
        model_kwargs, criterion = apply_presets(args)
        net = build_model(args, model_kwargs)

        if torch.cuda.is_available():
            net.load_state_dict(torch.load(model_dir[fold['fold'] -1], map_location='cuda'))
            net.cuda()
        else:
            net.load_state_dict(torch.load(model_dir[fold['fold'] -1], map_location='cpu'))



        rs_pred=[]
        rs_lab=[] 
        net.eval()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with torch.inference_mode():
            rs_pred, rs_lab = test(test_salobj_dataloader, net, device, fold['fold'], case='test', model_name =args.model)
            
        avg_recall, avg_precision, f1_scores, avg_accuracy, avg_iou, avg_class_oa, average_f1 = calc_metrics(rs_pred, rs_lab)
        print(f"Average F1 Score: {average_f1:.4f}")
        print(f"\nOverall Accuracy (including background): {avg_accuracy * 100:.2f}%")
        for i, (r, p, iou, oa,f1) in enumerate(zip(avg_recall, avg_precision, avg_iou, avg_class_oa,f1_scores)):
            print(f"Class {i+1}: Recall = {r:.4f}, Precision = {p:.4f}, IoU = {iou:.4f}, OA = {oa:.4f}, F1 Score: {f1:.4f}")

        print('||||||||||||||||||||||||||||||||||||||FOLD||||||||||||||||||||||||||||||')

        print(model_dir[fold['fold'] -1])
   
        all_fold_recalls.append(avg_recall)
        all_fold_precisions.append(avg_precision)
        all_fold_accuracies.append(avg_accuracy)
        all_fold_f1.append(f1_scores)
        all_fold_ious.append(avg_iou)
        all_fold_OAs.append(avg_class_oa)
        #||||||||||||||||||||||||||||||||||||||||||||overalll |||||||||||||||||||||||||||||||||||||||||||||||||
    print(np.array(all_fold_recalls).shape,'all fold recall shape')
    print(np.array(all_fold_f1).shape,'all fold recall shape')
    cv_calc(all_fold_recalls,all_fold_precisions,all_fold_accuracies, all_fold_f1, all_fold_ious, all_fold_OAs)
    print("number of test sample is ", test_images.shape)
    print(args.model, " model")
if __name__ == "__main__":
    main()

