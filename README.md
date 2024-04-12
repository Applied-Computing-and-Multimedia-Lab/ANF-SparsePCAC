# ANF-Sparse-PCAC
### [**[paper]**](https://ieeexplore.ieee.org/document/10317255)
This repository contains source code for Sparse Tensor-based point cloud attribute compression using Augmented Normalizing Flows (APSIPA ASC 2023).


# Abstract
The large amount of data of point cloud poses challenges for efficient storage and transmission. To address this problem, various learning-based techniques, in addition to rule-based solutions, have been developed for point cloud compression.While many previous works employed the variational autoencoder (VAE) structure, they have failed to achieve promising performance at high bitrates. In this paper, we propose a novel point cloud attribute compression technique based on the Augmented Normalizing Flow (ANF) model, which incorporates sparse convolutions where a sparse tensor is used to represent the point cloud attribute. The invertibility of the NF model provides better reconstruction compared to VAE-based coding schemes.ANF provides a more flexible way to model the input distribution by introducing additional conditioning variables into the flow. Not only comparable to G-PCC, the experimental results demonstrate the effectiveness and superiority of the proposed method over several learning-based point cloud attribute compression techniques, even without requiring sophisticated context modeling.
![architecture](https://github.com/kai0416s/ANF-Sparse-PCAC/blob/main/architecture.png)
# Requirments environment
* Create env：
```
conda create -n ANFPCAC python=3.7
conda activate ANFPCAC
conda install ANFPCAC devel -c anaconda
```
- Install torch：
```
pip install torch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
```

* ## install MinkowskiEngine：
[MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine)
* ## Requirements
Step 1. Install requirements:
```
pip install -r requirements.txt
```
Step 2. Use the torchacc
```
cd torchacc/
python3 setup.py install
cd ../
```

# dataset
* Training：
[ScanNet dataset](https://github.com/ScanNet/ScanNet), which is a large open-source dataset of indoor scenes.
>cube division with size 64* 64 *64. We randomly selected 50,000 cubes and used them for training.

![trainingdata](https://github.com/kai0416s/ANF-Sparse-PCAC/blob/main/trainingdata.png)

- Testing：
8iVFB dataset(longdress, loot, redandblack, and soldier.)

![testingdata](https://github.com/kai0416s/ANF-Sparse-PCAC/blob/main/testingdata.png)

# check point download：
| check point  | [Link](https://drive.google.com/drive/folders/1De7zUg2WWiax_u-Z5HvlhI-AfD8N3Hnd?usp=sharing)|
| ---------- | -----------|


## ❗ After data preparation, the overall directory structure should be：
```
│ANF-Sparse-PCAC/
├──results/
├──output/
├──ckpts/
│   ├──/final_result/
│                 ├──/R7.pth
├──.......
```

# Training
* The default setting：

| High rate check point  | setting|
| ---------- | -----------|
| learning rate   | 1×10^(-4) which gradually decreases to 1×10^(-6)   |
| lamda   | 0.05   |
| Epoch   | 500  |

## Train
```
python train.py
```
- You need to change the check point location and then can train the low rate check point.
```
parser.add_argument("--init_ckpt", default='/ANFPCAC/ckpts/final_result/R7.pth')
```
# Testing

* input the orignal point cloud path：
```
filedir_list = [
  './testdata/8iVFB/longdress_vox10_1300.ply',
  './testdata/8iVFB/loot_vox10_1200.ply',
  './testdata/8iVFB/redandblack_vox10_1550.ply',
  './testdata/8iVFB/soldier_vox10_0690.ply',
]
```
- output path and check point location：
```
Output = '/0222'
Ckpt = '/final_result'
```
* The check point we have provide：
```
ckptdir_list = [
  './ckpts' + Ckpt + '/R0.pth',
  './ckpts' + Ckpt + '/R1.pth',
  './ckpts' + Ckpt + '/R2.pth',
  './ckpts' + Ckpt + '/R3.pth',
  './ckpts' + Ckpt + '/R4.pth',
  './ckpts' + Ckpt + '/R5.pth',
  './ckpts' + Ckpt + '/R6.pth',
  './ckpts' + Ckpt + '/R7.pth',
]
```
> R7.pth is the high rate.

Then you can run the test and get the result in folder 0222
and we also provide the experiment result.

## Test
```
python test.py
```

# Result
compared our approach with three methods, including two learning-based methods：Deep PCAC and SparsePCAC, and a traditional point cloud compression standard G-PCC(TMC13v14).

![BD-Rate (%) and BD-PSNR (dB) with respect to abchors](https://github.com/kai0416s/ANF-Sparse-PCAC/blob/main/result.png)

# Authors
These files are provided by National Chung Cheng University [Applied Computing and Multimedia Lab](https://chiang.ccu.edu.tw/index.php).

Please contact us (s69924246@gmail.com and yimmonyneath@gmail.com) if you have any questions.

