#  Radargram Segmentation — Supervised Framework

###  Overview
This project provides a **PyTorch-based supervised training framework** for **semantic segmentation of radar sounder (RS) data**, such as MCoRDS or SHARAD radargrams.  
It enables training and evaluation across multiple architectures under a unified interface.

Supported model  literature architectures:

-  **UNet** – classic encoder-decoder segmentation model  
-  **UNet-ASPP** – UNet enhanced with Atrous Spatial Pyramid Pooling  
-  **Efficient U²-Net** – multi-task nested U-structure for efficient feature learning  
-  **TransSounder** – transformer-based model for radar sounder data  

The script performs **fold-wise training**, manages checkpoints automatically, and allows for flexible configuration via argument presets.

---

###  Folder Structure
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
├── supervised_train.py             # Main training script 
└── README.md                       # Documentation (this file)
```


### 🧮 Requirements
Install dependencies:
```bash
pip install requirements.txt

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


---


### 👤 Author
**Milkisa T. Yebasse**  
Ph.D. Researcher — RS Lab, University of Trento  
📧 milkisa.yebasse@unitn.it  
🌐 [GitHub](https://github.com/milkisa)
# supervised_radar-sounder

# supervised_radar-sounder
# supervised_radar-sounder
