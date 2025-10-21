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

from implmentation.dataset import mc10_data_model
from implmentation.inputs import parse_args, apply_presets, build_model,muti_bce_loss_fusion, PRESETS
seed = 42
np.random.seed(seed)


p= 427
def main():
    all_fold_recalls = []
    all_fold_precisions = []
    all_fold_accuracies = []
    all_fold_f1 = []
    all_fold_ious = []
    all_fold_OAs = []   
    rs_pred = []
    rs_lab = []

    rs_image, rs_label, folds, model_dir = mc10_data_model()

    #kf.split(rs_image)

    for fold, (train_index, test_index) in enumerate(folds):
        torch.cuda.empty_cache()
        print(fold)
        print(type(fold))
        print(f"\nFold {fold + 1}")
        
        # Split images and labels into train/test for the current fold
        train_images, test_images = rs_image[train_index], rs_image[test_index]
        train_labels, test_labels = rs_label[train_index], rs_label[test_index]


        rs_image_fold= np.expand_dims(test_images, axis=-1)
        rs_label_fold= np.expand_dims(test_labels, axis=-1)
        print(rs_image_fold.shape,'testing image size')

 


        test_salobj_dataset = SalObjDataset(img_name_list = rs_image_fold,
                                            lbl_name_list= rs_label_fold,
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
            net.load_state_dict(torch.load(model_dir[fold]))
            net.cuda()
        else:
            net.load_state_dict(torch.load(model_dir[fold], map_location='cpu'))



        rs_pred=[]
        rs_lab=[] 
        net.eval()
        num=0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with torch.inference_mode():
            for i_test, data_test in enumerate(test_salobj_dataloader):


                # print("inferencing:",img_name_list[i_test].split(os.sep)[-1])

                inputs_test = data_test['image'].to(device, dtype=torch.float32)
                labels= data_test['label']
                # print(type(inputs_test))

                if args.model == 'eu':
                    
                    d1,d2,d3,d4,d5,d6,d7= net(inputs_test)

                    # normalization

                    pred = d1
                    del d1,d2,d3,d4,d5,d6,d7,
            
                else:
                    pred= net(inputs_test)


                p,l = save_output(inputs_test, pred,labels,num ,fold)
                
                rs_pred.append(p)

                rs_lab.append(l)

                num= num+1

                del p,l

        avg_recall, avg_precision , f1, avg_accuracy, iou , OA  = calc_metrics(rs_pred, rs_lab)
        print(model_dir[fold])
        all_fold_recalls.append(avg_recall)
        all_fold_precisions.append(avg_precision)
        all_fold_accuracies.append(avg_accuracy)
        all_fold_f1.append(f1)
        all_fold_ious.append(iou)
        all_fold_OAs.append(OA)
        #||||||||||||||||||||||||||||||||||||||||||||overalll |||||||||||||||||||||||||||||||||||||||||||||||||
    cv_calc(all_fold_recalls,all_fold_precisions,all_fold_accuracies, all_fold_f1, all_fold_ious, all_fold_OAs)
    print("number of test sample is ", rs_image_fold.shape)
    print(args.model, " model")
if __name__ == "__main__":
    main()

