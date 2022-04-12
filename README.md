# Learning spatially varying pixel exposures for motion deblurring
[Project Page](www.computationalimaging.org/) |
[Paper](www.computationalimaging.org/) |
[Arxiv](www.computationalimaging.org/)

> Learning Spatially Varying Pixel Exspoures for Motion Deblurring
> [Cindy M. Nguyen](https://ccnguyen.github.io/), [Julien N.P. Martel](https://twitter.com/jnpmartel), [Gordon Wetzstein](https://stanford.edu/~gordonwz/)
> Under review - 2022

If you find this work useful, please consider citing us!
```python
TBD
```

## Abstract
Computationally removing the motion blur introduced by camera shake or object motion in a captured image remains a challenging task in computational photography. 
Deblurring methods are often limited by the fixed global exposure time of the image capture process. The post-processing algorithm either must deblur a longer exposure that contains relatively little noise or denoise a short exposure that intentionally removes the opportunity for blur at the cost of increased noise. 
We present a novel approach of leveraging spatially varying pixel exposures for motion deblurring using next-generation focal-plane sensor--processors along with an end-to-end design of these exposures and a machine learning--based motion-deblurring framework. We demonstrate in simulation and a physical prototype that learned spatially varying pixel exposures (L-SVPE) can successfully deblur scenes while recovering high frequency detail. Our work illustrates the promising role that focal-plane sensor--processors can play in the future of computational imaging.

## Getting started
#### 0) Install conda environment
```python
conda env create -f environmental.yml
conda activate coded
```

#### 1) Download the dataset
Set up the dataset so that it is located outside of the repo.
Use `dataprep/process_nfs_zip.sh` to download the Need for Speed dataset.

#### 2) Preprocess the dataset
Run `python dataprep/init_data.py` to process the dataset.
The data is processed into a `.pt` file that is a dictionary in which each key is a video number,
and each value is a dictionary where the key is the ID of the clip generated and the value is the video.

#### 3) Train a model
To run a pretrained model, download the [models](https://drive.google.com/drive/folders/107ZTxAJOMY7zWbaoo-N-aQpMx5Au-OIS?usp=sharing)
and place the folder in `logs`. Then use the following code examples to continue training a model:
```
python train.py --shutter=short --interp=none --resume=21-10-31 --exp_name=short
python train.py --shutter=nonad --interp=scatter --resume=21-10-31 --exp_name=nonad_bilinear
```

If you would like to train from scratch, use the following example:
```python
python train.py --shutter=lsvpe --interp=scatter --exp_name=my_model
```

#### 4) Test
Run `python test.py` to test all models

### Save structure
1. logs
    1. 22-04-01
        1. 22-04-01-unet
            1. individual experiment
            2. individual experiment
        2. 22-04-01-dncnn
            1. individual experiment
            2. individual experiment
    2. 22-04-02
        1. 22-04-02-unet
            1. individual experiment

## References
1. Zhang et al. - DnCNN
2. Zamir et al. - MPRNet