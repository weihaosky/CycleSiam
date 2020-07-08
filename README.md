# CycleSiam

This is the official implementation code for CycleSiam. It is built based on [SiamMask](https://github.com/foolwood/SiamMask). For technical details, please refer to :

**Self-supervised Object Tracking and Segmentation with Cycle-consistent Siamese Networks** <br />
[Weihao Yuan](https://weihao-yuan.com), Michael Yu Wang, Qifeng Chen <br />
**IROS2020** <br />
**[[Paper]()]** <br />


<div align="center">
  <img src="" width="600px" />
</div>

### Bibtex
If you find this code useful, please consider citing:

```
@inproceedings{yuan2020self,
  title={Self-supervised object tracking and segmentation with cycle-consistent siamese networks},
  author={Yuan, Weihao and Wang, Michael Yu and Chen, Qifeng},
  booktitle={IEEE International Conference on Robotics and Automation (ICRA)},
  pages={},
  year={2020},
  organization={IEEE}
}
```


## Contents
1. [Environment Setup](#environment-setup)
2. [Demo](#demo)
3. [Training](#training-models)

## Environment setup
This code has been tested on Ubuntu 16.04, Python 3.6, Pytorch 0.4.1, CUDA 9.2, RTX 2080 GPUs

- Clone the repository 
```
git clone https://github.com/weihaosky/CycleSiam.git && cd CycleSiam
export CycleSiam=$PWD
```
- Setup python environment
```
conda create -n cyclesiam python=3.6
source activate cyclesiam
pip install -r requirements.txt
bash make.sh
```
- Add the project to your PYTHONPATH
```
export PYTHONPATH=$PWD:$PYTHONPATH
```

## Demo
- [Setup](#environment-setup) your environment
- Download the CycleSiam model
```shell
cd $CycleSiam/experiments/siammask_sharp
wget 
wget 
```
- Run `demo.py`

```shell
cd $CycleSiam/experiments/siammask_sharp
export PYTHONPATH=$PWD:$PYTHONPATH
python ../../tools/demo.py --resume CycleSiam_plus.pth --config config_davis.json
```


## Training

### Training Data 
- Download the [Youtube-VOS](https://youtube-vos.org/dataset/download/), 
[COCO](http://cocodataset.org/#download), 
[ImageNet-DET](http://image-net.org/challenges/LSVRC/2015/), 
and [ImageNet-VID](http://image-net.org/challenges/LSVRC/2015/).
- Preprocess each datasets according the [readme](data/coco/readme.md) files.

### Download the pre-trained model (174 MB)
(This model was trained on the ImageNet-1k Dataset)
```
cd $CycleSiam/experiments
wget http://www.robots.ox.ac.uk/~qwang/resnet.model
ls | grep siam | xargs -I {} cp resnet.model {}
```

### Training CycleSiam base model
- [Setup](#environment-setup) your environment
- From the experiment directory, run
```
cd $CycleSiam/experiments/siammask_base/
bash run.sh
```
- If you experience out-of-memory errors, you can reduce the batch size in `run.sh`.
- You can view progress on Tensorboard (logs are at <experiment\_dir>/logs/)
- After training, you can test checkpoints on VOT dataset.
```shell
bash test_all.sh -s 1 -e 20 -d VOT2018 -g "0 1 2 3"  # test all snapshots with 4 GPUs
```
- Select best model for hyperparametric search.
```shell
#bash test_all.sh -m [best_test_model] -d VOT2018 -n [thread_num] -g [gpu_num] # 8 threads with 4 GPUS
bash test_all.sh -m snapshot/checkpoint_e18.pth -d VOT2018 -n 8 -g "0 1 2 3" # 8 threads with 4 GPUS
```

### Training CycleSiam model with the Refine module
- [Setup](#environment-setup) your environment
- In the experiment file, train with the best CycleSiam base model
```
cd $CycleSiam/experiments/siammask_sharp
bash run.sh <best_base_model>
bash run.sh checkpoint_e18.pth
```
- You can view progress on Tensorboard (logs are at <experiment\_dir>/logs/)
- After training, you can test checkpoints on VOT dataset
```shell
bash test_all.sh -s 1 -e 20 -d VOT2018 -g "0 1 2 3"
```
- Select best model for hyperparametric search.
```shell
#bash test_all.sh -m [best_test_model] -d VOT2018 -n [thread_num] -g [gpu_num] # 8 threads with 4 GPUS
bash test_all.sh -m snapshot/checkpoint_e19.pth -d VOT2018 -n 8 -g "0 1 2 3" # 8 threads with 4 GPUS
```

### Pretrained models
| <sub> Model </sub> | <sub>VOT2016</br>EAO / A / R</sub> | <sub>VOT2018</br>EAO / A / R</sub>  | <sub>DAVIS2016</br>J / F</sub>  | <sub>DAVIS2017</br>J / F</sub>  | <sub> Speed </sub>
|:-------------------------:|:-------------------------------:|:-------------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|
| <sub> CycleSiam </sub> | <sub>0.371 / 0.603 / 0.294</sub> | <sub>0.294 / 0.562 / 0.389</sub>  | <sub>- / -</sub>  | <sub>- / -</sub>  | <sub> 59 </sub> |
| <sub> CycleSiam+ </sub> | <sub>0.398 / 0.601 / 0.247</sub> | <sub>0.317 / 0.549 / 0.314</sub>  | <sub>64.9 / 62.0</sub>  | <sub>50.9 / 56.8</sub>  | <sub> 44 </sub>  |


## License
Licensed under an MIT license.

