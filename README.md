# CUMD

This is the released code and models for CUMD.

![image](https://github.com/tolearnmuch/CUMD/blob/main/figs/Figure%201_9.png)

## 1. Important updates

* 2022-07-11: This project is going to be released, please waiting.

* 2022-07-12: The inference and training is go.

## 2. Getting started

The requirements of the hardware and software are given as below.

### 2.1. Prerequisites

> CPU: Intel(R) Core(TM) i7-6900K CPU @ 3.20GHz
>
> GPU: GeForce GTX 1080 Ti
> 
> CUDA Version: 10.2
> 
> OS: Ubuntu 16.04.6 LTS

### 2.2. Installing

Configure the virtual environment on Ubuntu.

* Create a virtual with python 3.6

```
conda create -n asvp python=3.6
conda activate asvp
```

* Install requirements (Please pay attention that we use tensorflow-gpu==1.10.0)

```
pip install -r requirements.txt
```

* Additionally install ffmpeg

```
conda install x264 ffmpeg -c conda-forge
```

* Besides, the requirements for the pre-trained depth esitmation models are necessary 

Please refer to the installing of [Pre-trained Depth Estimation](https://github.com/aim-uofa/AdelaiDepth/tree/main/LeReS).

Here the virtual env is created on Ubuntu.

### 2.3. Dataset Preparation

Datasets contain: [KTH human action dataset](https://www.csc.kth.se/cvap/actions/) & [BAIR action-free robot pushing dataset](https://sites.google.com/view/sna-visual-mpc/). For reproducing the experiment, the processed dataset should be downloaded:

* For KTH, raw data and subsequence file should be downloaded firstly. In this turn of submission, please temporarily download from:

[raw data](https://mega.nz/folder/JREhlAKB#U26ufSZcVSiw0EOOlW6pMw) and [subsequence file](https://mega.nz/folder/EVMiRJhB#Gboh1r5PmbqGv97db2974w). After downloading, drag all .zip and .tar.gz files into ./data directory, and run

```
bash data/preprocess_kth.sh
```

Then all preprocessed and subsequence splitted frames are obtained in ./data/kth/processed.

* Depth data obtained by estimating with the pre-trained parameters

```
cd LeRes
python ./tools/test_depth_kth.py --load_ckpt res101.pth --backbone resnext101
cd ..
```

* Run the code below for converting images into tfrecords, the details can be referred to [ASVP](https://github.com/tolearnmuch/ASVP).

```
bash data/kth2tfrecords.sh 
...
```

## 3. Inference with released models

For downloading the [released models](https://mega.nz/folder/pFsBiDwa#3k4qgxMbHidNmEQfTBkqGw), the released models should be placed as:

>——./pretrained/pretrained_models/kth/ours_cumd
>
>——./pretrained/pretrained_models/bair_action_free/ours_cumd

and the pre-trained models for baseline should be referred to [ASVP](https://github.com/tolearnmuch/ASVP).

### 3.1. Inference on KTH human action

* For running our released model, please run

```
CUDA_VISIBLE_DEVICES=1 python scripts/evaluate.py --input_dir data/kth --dataset_hparams sequence_length=30 --checkpoint logs/kth/ours_cumd/model-kth --mode test --results_dir results_test_samples/kth --batch_size 3
```

* For running the baseline, please refer to [ASVP](https://github.com/tolearnmuch/ASVP).

### 3.2. Inference on BAIR action-free robot pushing

* For running our released model, please run

```
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py --input_dir data/bair --dataset_hparams sequence_length=22 --checkpoint logs/bair_action_free/ours_cumd/model-bair --mode test --results_dir results_test_samples/bair_action_free --batch_size 8
```

* For running the baseline, please refer to [ASVP](https://github.com/tolearnmuch/ASVP).

## 4. Training

Active pattern mining is necessary only for training and there is no need to do this if only with respective to inference with released model.

* When trying to separate active patterns along with non-active ones from videos, please refer to [details](https://github.com/Anonymous-Submission-ID/Anonymous-Submission/tree/main/separating_active_patterns/).

* After all active patters and non-active patterns are mined, these images are convertted to tfrecords for training.

```
bash data/kth2tfrecords_ap.sh
```

The final data could be downloaded from [drive](https://mega.nz/folder/VVlUiZII#kqCMjIRfCoS4IoOuMjTXZg/).

### 4.1. Training to Disentangle Motion in RGB space

* For training to do svp in RGB space with MMDN

```
CUDA_VISIBLE_DEVICES=0,1 python scripts/train_vp2.py --input_dir data/kth --dataset kth --model cumd --model_hparams_dict hparams/kth/ours_cumd/model_hparams.json --output_dir logs/kth/save_rgb
```

Note that, in this turn of training, all training data should be prepared by RGB videos as in 2.3 which is the same situation as [ASVP](https://github.com/tolearnmuch/ASVP).

* For training our model with active patterns and non-active patterns on BAIR action-free, please run

```
CUDA_VISIBLE_DEVICES=0,1 python scripts/train_vp2.py --input_dir data/bair --dataset bair --model cumd --model_hparams_dict hparams/bair_action_free/ours_cumd/model_hparams.json --output_dir logs/bair_action_free/save_rgb
```

### 4.2. Training to Disentangle Motion in Depth

* For training to do svp in depth space with MMDN

```
CUDA_VISIBLE_DEVICES=0,1 python scripts/train_vp2.py --input_dir data/kth --dataset kth --model cumd --model_hparams_dict hparams/kth/ours_cumd/model_hparams.json --output_dir logs/kth/save_depth
```

Note that, in this turn of training, all training data should be prepared by depth videos as in 2.3.

* For training our model with active patterns and non-active patterns on BAIR action-free, please run

```
CUDA_VISIBLE_DEVICES=0,1 python scripts/train_vp2.py --input_dir data/bair --dataset bair --model cumd --model_hparams_dict hparams/bair_action_free/ours_cumd/model_hparams.json --output_dir logs/bair_action_free/save_depth
```

### 4.3. Motion Complementing and Uncertainty Complementing

* For training MCN, UCN during prediction.

```
CUDA_VISIBLE_DEVICES=0,1 python scripts/train_vp2_fusion.py --input_dir data/kth --dataset kth --model cumd --model_hparams_dict hparams/kth/ours_cumd/model_hparams.json --output_dir logs/kth/ours_cumd --checkpoint logs/kth/save_rgb/model-rgb --checkpoint_d logs/kth/save_depth/model-depth
```

Please rename the pre-trained models in 4.1 and 4.2 as model-rgb and model-depth before training.

* For training on BAIR action-free, please run

```
CUDA_VISIBLE_DEVICES=0,1 python scripts/train_vp2_fusion.py --input_dir data/bair --dataset bair --model cumd --model_hparams_dict hparams/bair_action_free/ours_cumd/model_hparams.json --output_dir logs/bair_action_free/ours_cumd --checkpoint logs/bair_action_free/save_rgb/model-rgb --checkpoint_d logs/bair_action_free/save_depth/model-depth
```

It needs to pay attention that, the source code here is not well organized, and the perfect code for training will be released in future.

## 5. More cases

Add additional notes about how to deploy this on a live system

## 6. License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments



