# Prompt-Based Dynamic Text Detector for Real-Time Industrial Panel Monitoring

### Haowen Zheng, Hua Lin, Haobo Zuo, Changhong Fu

## Abstract
As an important part of industrial production test equipment, industrial panels often display critically important data and information. 
However, traditional manual methods can no longer achieve the required levels of efficiency and precision for the real-time monitoring and recording of a large volume of dynamic text on panels. To reduce the labor cost of the task and improve monitoring accuracy, this work develops an intelligent visual monitoring system to monitor recorded data in real-time. Specifically, a prompt-based dynamic text detector (PTDet) is proposed for real-time high-precision detection of dynamic text targets in prompt regions. A dynamic adaptive prompt framework is designed to skillfully encode the point prompt into the detector. It can strengthen the prediction of key areas and redundant targets by weakening the background. Moreover, compound direction aggressive network (CDAN) is proposed to enhance the extraction of compound directional feature texture and edges to reduce screen reflections and text blurring. Set data and challenging benchmarks in over 50 real-world industrial test scenarios to prove the validity and robustness of PTDet experimentally. Real-world applications also show its practicability with 33.1 FPS.

## Overview
![Overview of the proposed PTDet framework.](/fig/total7.pdf)
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