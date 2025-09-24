# Prompt-Based Dynamic Text Detector for Real-Time Industrial Panel Monitoring

### Haowen Zheng, Hua Lin, Haobo Zuo, and Changhong Fu

## Overview
![Overview of the proposed PTDet framework.](/fig/total.jpg)
## About Code
### 1. Requirements:
- Python3
- torch == 1.13.1
- GCC >= 4.9 (This is important for PyTorch)
- CUDA >= 9.0 (11.7 is recommended)
Please install related libraries before running this code: 
```bash
  # first, make sure that your conda is setup properly with the right environment
  # for that, check that `which conda`, `which pip` and `which python` points to the
  # right path. From a clean conda env, this is what you need to do

  conda create --name PTD -y
  conda activate PTD

  # this installs the right pip and dependencies for the fresh python
  conda install ipython pip
  # CUDA 11.7
  pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
  # python dependencies
  pip install -r requirement.txt

  # clone repo
  git clone https://github.com/vision4robotics/PTD.git
  cd PTD/
```


### 2. Test

```bash 
python demo.py ./experiments/seg_detector/panel_resnet18_deform_thre.yaml                                
	--source test_v5.mp4          # video path
```
### 3. Contact
If you have any questions, please contact me.

Haowen zheng

Email: [@tongji.edu.cn](2211329@tongji.edu.cn)

For more evaluations, please refer to our paper.

## References 

```

```
## Acknowledgement
This code has been modified based on the foundation laid by DB.
Thanks for their great work!
