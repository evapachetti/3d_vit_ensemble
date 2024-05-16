# 3D-Vision-Transformer Stacking Ensemble for Assessing Prostate Cancer Aggressiveness from T2w Images

Official code for [**3D-Vision-Transformer Stacking Ensemble for Assessing Prostate Cancer Aggressiveness from T2w Images**](https://www.mdpi.com/2306-5354/10/9/1015) based on [Pytorch reimplementation](https://github.com/jeonsworld/ViT-pytorch) by [jeonsworld](https://github.com/jeonsworld) of [Google's repository for the ViT model](https://github.com/google-research/vision_transformer) [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) by Alexey Dosovitskiy. 

![vit_ensemble](./img/vit_ensemble.png)

## Dataset
We utilized the [Prostate-X 2](https://www.cancerimagingarchive.net/collection/prostatex/) dataset for our experiments. To see pre-processing details, please refer to our [paper](https://www.mdpi.com/2306-5354/10/9/1015).
According to our code, data should be stored according to the following structure:
```
├── dataset
│   └── ProstateX-YYYY
│       ├── original                             
│       ├── rotation
│       ├── horizontal_flip
│       ├── vertical_flip
```
The ProstateX-YYYY folder refers to single patient acquisition, while the four subfolders contain the original and augmented versions of the images.

## Usage

### 1. Train baseline ViTs on the whole dataset or using CV/bootstrapping techniques
```
python train_baseline_cv.py
```
```
python train_baseline_bootstrap.py
```
Re-train best baseline ViT on the whole dataset:
```
python train_baseline_whole_dataset.py
```


### 2. Train ensemble ViT on the whole dataset or using CV/bootstrapping techniques
```
python train_ensemble_cv.py
```
```
python train_ensemble_bootstrap.py
```
Re-train best ensemble ViT on the whole dataset:
```
python train_ensemble_whole_dataset.py --ensemble_conf_list 5,9,11
```
The --ensemble_conf_list parameter defines the combination of baseline ViTs that compose the best-performing ensemble to re-train on the whole dataset.

### 3. Test baseline and ensemble ViTs trained according to CV
```
python test.py
```

## Citations

```bibtex
@article{pachetti20233d,
  title={3D-Vision-Transformer Stacking Ensemble for Assessing Prostate Cancer Aggressiveness from T2w Images},
  author={Pachetti, Eva and Colantonio, Sara},
  journal={Bioengineering},
  volume={10},
  number={9},
  pages={1015},
  year={2023},
  publisher={MDPI}
}
```
