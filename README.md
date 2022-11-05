# Learning spatially varying pixel exposures for motion deblurring
[Project Page](https://ccnguyen.github.io/lsvpe/) |
[Paper](https://arxiv.org/pdf/2204.07267.pdf) |
[Arxiv](https://arxiv.org/abs/2204.07267)

> Learning Spatially Varying Pixel Exspoures for Motion Deblurring
> [Cindy M. Nguyen](https://ccnguyen.github.io/), [Julien N.P. Martel](https://twitter.com/jnpmartel), [Gordon Wetzstein](https://stanford.edu/~gordonwz/)
> ICCP 2022

If you find this work useful, please consider citing us!
```python
@INPROCEEDINGS{9887786,
  author={Nguyen, Cindy M. and Martel, Julien N. P. and Wetzstein, Gordon},
  booktitle={2022 IEEE International Conference on Computational Photography (ICCP)}, 
  title={Learning Spatially Varying Pixel Exposures for Motion Deblurring}, 
  year={2022},
  volume={},
  number={},
  pages={1-11},
  doi={10.1109/ICCP54855.2022.9887786}}
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
To run a pretrained model, download the [model checkpoints](https://drive.google.com/drive/folders/1LWROt3RXNZom_8pBuYy_WkIyZTYCcJsp)
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
```
@article{zhang2017beyond,
  title={Beyond a gaussian denoiser: Residual learning of deep cnn for image denoising},
  author={Zhang, Kai and Zuo, Wangmeng and Chen, Yunjin and Meng, Deyu and Zhang, Lei},
  journal={IEEE transactions on image processing},
  volume={26},
  number={7},
  pages={3142--3155},
  year={2017},
  publisher={IEEE}
}
@inproceedings{zamir2021multi,
  title={Multi-stage progressive image restoration},
  author={Zamir, Syed Waqas and Arora, Aditya and Khan, Salman and Hayat, Munawar and Khan, Fahad Shahbaz and Yang, Ming-Hsuan and Shao, Ling},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={14821--14831},
  year={2021}
}
```
