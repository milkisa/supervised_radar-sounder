from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
seed = 42
np.random.seed(seed)
import torch

def mc10_data_model():
    import pandas as pd
    data = torch.load('/mnt/data/dataset/greenland_64multi.pt')
    #data = data['test']
    rs_image = data['data'].to('cpu').numpy()
    rs_label = data['label'].to('cpu').numpy()
    
    
    kf = KFold(n_splits=3, shuffle=True, random_state=seed)
    
    
    folds_1 = [
    # Fold 1: test = [1470–2102], train = [0–1469]
    (np.arange(0, 1470), np.arange(1470, 2103)),

    # Fold 2: test = [0–632], train = [633–2102]
    (np.arange(633, 2103), np.arange(0, 633)),

    # Fold 3: test = [633–1265], train = [0–632, 1266–2102]
    (np.concatenate((np.arange(0, 633), np.arange(1266, 2103))), np.arange(633, 1266))
        ]
    
    folds_2 = [
  
    ]
    
    folds_3 = [

    ]
    fold= [folds_1, folds_2, folds_3]


    model_dir = ['/mnt/data/supervised/unet/greenland_unet_fold0_epoch100_loss0.012035_time3072.8_20251020-223133.pth',
        '/mnt/data/supervised/unet/greenland_unet_fold1_epoch100_loss0.012113_time3072.6_20251020-232245.pth',
        '/mnt/data/supervised/unet/greenland_unet_fold2_epoch100_loss0.009736_time3071.4_20251021-001357.pth'
                ]
    return rs_image,rs_label,fold[0], model_dir