# 🛰 Radargram Segmentation — Supervised Framework
git config --global user.name "Milkisa T. Yebasse"
git config --global user.email "milkisa.yebasse@gmail.com"

### 📘 Overview
This project provides a **PyTorch-based supervised training framework** for **semantic segmentation of radar sounder (RS) data**, such as MCoRDS or SHARAD radargrams.  
It enables training and evaluation across multiple architectures under a unified interface.

Supported model  literature architectures:
- 🧩 **UNet** – classic encoder-decoder segmentation model  
- 🌊 **UNet-ASPP** – UNet enhanced with Atrous Spatial Pyramid Pooling  
- ⚡ **Efficient U²-Net** – multi-task nested U-structure for efficient feature learning  
- 🛰 **TransSounder** – transformer-based model for radar sounder data  

The script performs **fold-wise training**, manages checkpoints automatically, and allows for flexible configuration via argument presets.

---

### 🧩 Folder Structure
```
project_root/
│
├── implmentation/
│   ├── dataset.py                  # mc10_data_model(), sharad_data_model(), etc.
│   ├── inputs.py                   # Presets, argument parser, build_model(), loss functions
│   ├── metrics.py                  # 
|   ├── output.py                   #
|
├── literature/
│   ├── unet.py                     # Standard UNet architecture
│   ├── aspp.py                     # UNetASPP with Atrous Spatial Pyramid Pooling
│   ├── transounder.py              # Transformer-based TransSounder model
│   ├── u2net.py                    # Efficient U²-Net model
│
├── data_loader.py                  # SalObjDataset, ToTensorLab utilities
├── supervised_foldtrain.py         # Main training script (this file)
└── README.md                       # Documentation (this file)
```

---

### ⚙️ Main Functionalities
✅ Loads radargram data and cross-validation folds from `mc10_data_model()` or `sharad_data_model()`  
✅ Dynamically builds the model via `parse_args()`, `apply_presets()`, and `build_model()`  
✅ Supports multi-output and single-output architectures  
✅ Performs iterative training with automatic model checkpointing  
✅ Compatible with CUDA acceleration  
✅ Modular design – easily switch datasets or model types  

---

### 🧠 Data Preparation
Data loading example:
```python
rs_image, rs_label, folds, _ = mc10_data_model()
```

- `rs_image`: NumPy array of radargram images  
- `rs_label`: NumPy array of label masks  
- `folds`: list of (train_index, test_index) pairs from K-Fold cross-validation  

Each fold is processed as:
```python
for fold, (train_index, test_index) in enumerate(folds):
    train_images, test_images = rs_image[train_index], rs_image[test_index]
    train_labels, test_labels = rs_label[train_index], rs_label[test_index]
```

---

### 🧱 Model Setup
The framework dynamically builds and configures the model:
```python
args = parse_args()
model_kwargs, criterion = apply_presets(args)
net = build_model(args, model_kwargs)
```

Supported models via `args.model`:
| Argument | Model | File |
|-----------|--------|------|
| `unet` | UNet | `literature/unet.py` |
| `aspp` | UNet-ASPP | `literature/aspp.py` |
| `eu` | Efficient U²-Net | `literature/u2net.py` |
| `transsounder` | TransSounder | `literature/transounder.py` |

Example:
```bash
python train_crossval.py --model eu --lr 1e-4 --weight_decay 5e-4
```

---

### 🏋️ Training Details
- **Epochs:** 5000  
- **Batch size:** 1  
- **Optimizer:** Adam (`lr`, `weight_decay` from presets)  
- **Loss functions:**
  - `muti_bce_loss_fusion()` for Efficient U²-Net (multi-output)
  - `criterion` (e.g., `CrossEntropyLoss`) for other models  

Checkpoint saving frequency:
```python
save_frq = 20000  # saves every 20k iterations
```

---

### 💾 Checkpoints
Checkpoints are automatically stored in the model directory defined in the preset:
```
model_dir/
│
└── greenland_<model>_fold<k>_iter<ite>_epoch<ep>_loss<val>_time<sec>_<timestamp>.pth
```

Example:
```
greenland_eu_fold1_iter20000_epoch10_loss0.125400_time134.2_20251016-143820.pth
```

---

### 📊 Example Console Output
```
Fold 1
Train Images shape: (200, 512, 512)
Train Labels shape: (200, 512, 512)
[OK] Model created: eu with 47,832,409 params
[OK] Optimizer created with: {'lr': 0.0001, 'weight_decay': 0.0005}
---start training...
[epoch: 005/5000, ite: 20000] train loss: 0.134521, tar: 0.000000
```

---

### 🧩 Customization
| Parameter | Description | Default |
|------------|--------------|----------|
| `epoch_num` | Number of training epochs | 5000 |
| `batch_size_train` | Training batch size | 1 |
| `save_frq` | Model saving frequency | 20000 |
| `args.model` | Model type | `"eu"` |
| `dataset` | Choose between `mc10_data_model()` or `sharad_data_model()` | `mc10` |

---

### 🧮 Requirements
Install dependencies:
```bash
pip install torch torchvision numpy matplotlib scikit-learn scipy
```

or using conda:
```bash
conda install pytorch torchvision numpy matplotlib scikit-learn scipy -c pytorch
```

---

### ⚡ GPU Usage
The script automatically uses CUDA if available:
```python
if torch.cuda.is_available():
    net.cuda()
```

To force CPU only:
```bash
CUDA_VISIBLE_DEVICES="" python train_crossval.py
```

---


### 👤 Author
**Milkisa T. Yebasse**  
Ph.D. Researcher — RS Lab, University of Trento  
📧 milkisa.yebasse@unitn.it  
🌐 [GitHub](https://github.com/milkisayebasse)
# supervised_radar-sounder

