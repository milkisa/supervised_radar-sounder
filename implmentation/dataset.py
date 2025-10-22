from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
seed = 42
np.random.seed(seed)
import torch
from sklearn.model_selection import train_test_split, KFold
def mc10_data_model():
    import pandas as pd
    data = torch.load('/mnt/data/dataset/greenland_64multi.pt')
    #data = data['test']
    rs_image = data['data'].to('cpu').numpy()
    rs_label = data['label'].to('cpu').numpy()
    
    
   # We’ll generate 3 random folds manually
    folds_1 = []
    
    n_samples = rs_image.shape[0]
    indices = np.arange(n_samples)

    for i in range(3):
        # Randomly shuffle and split into train (70%), val (15%), test (15%)
        train_idx, temp_idx = train_test_split(indices, test_size=0.3, random_state=seed + i)
        val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=seed + i)

        folds_1.append({
            'fold': i + 1,
            'train_idx': train_idx,
            'val_idx': val_idx,
            'test_idx': test_idx
        })

    
    folds_2 = [
  
    ]
    
    folds_3 = [

    ]
    fold= [folds_1, folds_2, folds_3]


    model_dir = ['/mnt/data/supervised/aspp/greenland_aspp_fold0_epoch0_loss0.995213_time37.5_20251020-181803.pth',
        '/mnt/data/supervised/aspp/greenland_aspp_fold1_epoch0_loss0.927630_time37.3_20251020-192039.pth',
        '/mnt/data/supervised/aspp/greenland_aspp_fold2_epoch0_loss0.944996_time37.2_20251020-202314.pth'
                ]
    return rs_image,rs_label,fold[0], model_dir