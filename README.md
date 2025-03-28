# Intelligent Fish Detection System with Similarity-Aware Transformer 

### Shengchen Li, Haobo Zuo, Changhong Fu, Zhiyong Wang, Zhiqiang Xu

## Abstract
The monitoring of industrial panels and instruments has intelligent implementation schemes under the development of computer vision. The previous manual monitoring methods for industrial panels had the disadvantages of being laborious, costly, and having inaccurate records with poor real-time performance. An intelligent visual monitoring system was developed using a high-frequency camera for real-time data recording to address these issues. In addition, this paper proposes a text detection framework PTD that can be prompted at will to detect key areas in real-time based on prompt information. Specifically, processing the point prompt into an adaptive Gaussian mask can filter the miscellaneous information in the scene. Moreover, the proposed compound directional difference convolution with the medium self-attention in CDAN can extract feature textures and edges under screen reflection and blurred text conditions to accurately obtain effective information. Experiments were conducted on a large amount of challenging industrial panel test scene data according to strict benchmarks, proving the superiority of the proposed method. It is deployed on the i9 + NVIDIA RTX3060 platform with an inference speed of 33.1 FPS, confirming the effectiveness and feasibility of the method. 
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

Email: [@tongji.edu.cn](shengcli@tongji.edu.cn)

For more evaluations, please refer to our paper.

## References 

```

```
## Acknowledgement
This code has been modified based on the foundation laid by DB.
Thanks for their great work!