## Usage

1. Install Python 3.8. For convenience, execute the following command.

```
pip install -r requirements.txt
```

2. Prepare Data. You can obtain the well pre-processed datasets from [[Google Drive]](https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2?usp=sharing) or [[Baidu Drive]](https://pan.baidu.com/s/1r3KhGd0Q9PJIUZdfEYoymg?pwd=i9iy), Then place the downloaded data in the folder`./dataset`. 


3. Train and evaluate model. We provide the experiment scripts under the folder `./scripts/`. You can reproduce the experiment results as the following examples:

```
bash ./scripts/ETTh1.sh
```
##  Citation
Visit our paper at (https://www.sciencedirect.com/science/article/pii/S0950705125009578)
Please cite the following paper if you use the code in your work:
```
@article{dou2025multivariate,
  title={Multivariate time series forecasting based on time-frequency transform mixed convolution},
  author={Dou, Jiaxin and Xun, Yaling and Yang, Haifeng and Cai, Jianghui and Li, Yanfeng and Han, Shuo},
  journal={Knowledge-Based Systems},
  pages={113912},
  year={2025},
  publisher={Elsevier}
}
```
