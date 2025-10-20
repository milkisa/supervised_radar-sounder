from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
seed = 42
np.random.seed(seed)

import torch
def sharad_data_model():
    print('start')
    import pandas as pd
    p= 453
    rs_image= pd.read_csv('dataset/sharad_rs.csv', header = None).to_numpy().reshape(p,-1,64)
    
    rs_label= pd.read_csv('dataset/sharad_gt.csv', header = None).to_numpy().reshape(p,-1,64)


   # rs_label[rs_label==5]=3
    kf = KFold(n_splits=3, shuffle=True, random_state=seed)
    
    
    
    folds_1 = [
    (np.arange(151, 453), np.arange(0, 151)),          # First 151 as test set
    (np.concatenate((np.arange(0, 151), np.arange(302, 453))), np.arange(151, 302)),  # Middle 151 as test set
    ( np.arange(0, 302), np.arange(302, 453))           # Last 151 as test set
    ]

    
    
    folds_2 = [
    (np.arange(0, 136), np.arange(136,453)), # First 136 as training, remaining 317 as test
    (np.arange(159, 295), np.concatenate((np.arange(0, 159), np.arange(295, 453)))),  # Middle 136 as training
    (np.arange(317, 453), np.arange(0, 317))  # Last 136 as training, first 317 as test
    ]
    
    
    
    folds_3 = [
    (np.arange(0, 45), np.arange(45, 453)),  # First 45 as training, remaining 408 as test
    (np.arange(45, 90), np.concatenate((np.arange(0, 45), np.arange(90, 453)))),  # Middle 45 as training
    (np.arange(408, 453), np.arange(0, 408))  # Last 45 as training, first 408 as test
    ]

    fold= [folds_1, folds_2, folds_3]
    model_dir = ['/mnt/data/elena_Scribble/sharad_aspp_0_fold__before_bce_itr_30000_train_0.017542_time_2739.805360.pth',
        '/mnt/data/elena_Scribble/sharad_aspp_1_fold__before_bce_itr_30000_train_0.020923_time_2739.562086.pth',
        '/mnt/data/elena_Scribble/sharad_aspp_2_fold__before_bce_itr_30000_train_0.024793_time_2739.157192.pth']
    return rs_image,rs_label,fold[0], model_dir

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
    (np.arange(0, 128), np.arange(128, 427)),           # First 128 as training, remaining 299 as test
    (np.arange(150, 278), np.concatenate((np.arange(0, 150), np.arange(278, 427)))),  # Middle 128 as training
    (np.arange(299, 427), np.arange(0, 299))            # Last 128 as training, first 299 as test
    ]
    
    folds_3 = [
    (np.arange(0, 43), np.arange(43, 427)),                 # First 43 as training, remaining as test
    (np.arange(192, 235), np.concatenate((np.arange(0, 192), np.arange(235, 427)))),  # Middle 43 as training
    (np.arange(384, 427), np.arange(0, 384))                # Last 43 as training, first 384 as test
    ]
    fold= [folds_1, folds_2, folds_3]


    model_dir = ['/mnt/data/supervised/EU/greenland_eu_fold0_epoch60_loss0.209974_time3452.2_20251020-121034.pth'
        #'/nt/data/elena_Scribble/mcpoly_1_fold__before_bce_itr_30000_train_0.010252_time_1391.316641.pth',
        #'/mnt/data/elena_Scribble/mc_2_fold__before_bce_itr_90000_train_0.000015_time_4177.519399.pth'
                ]
    return rs_image,rs_label,fold[0], model_dir