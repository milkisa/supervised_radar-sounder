from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
seed = 42
np.random.seed(seed)


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


    model_dir = ['/mnt/data/supervised/EU/greenland_eu_fold0_epoch60_loss0.209974_time3452.2_20251020-121034.pth'
        #'/nt/data/elena_Scribble/mcpoly_1_fold__before_bce_itr_30000_train_0.010252_time_1391.316641.pth',
        #'/mnt/data/elena_Scribble/mc_2_fold__before_bce_itr_90000_train_0.000015_time_4177.519399.pth'
                ]
    return rs_image,rs_label,fold[0], model_dir