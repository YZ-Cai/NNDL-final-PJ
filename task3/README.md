# Task 3 of Final project of "Neural Network and Deep Learning"

-----

Table of Contents
=================

* [This is the task 3 of final project of "Neural Network and Deep Learning"](#this-is-the-final-project-of-neural-network-and-deep-learning)
   * [Introduction](#introduction)
   * [Dependencies](#dependencies)
   * [Structure](#structure)
   * [Key Features](#key-features)
   * [Preparing Data](#preparing-data)
   * [Using COLMAP](#using-colmap)
   * [Training NeRF](#training-nerf)
   * [Results](#results)
   

## Introduction

The project is the final project of **DATA620004**.
In this project, we use the **COLMAP** to preprocess a set of 131 images, then pass them to **NeRF** model to train and render the video.
![Positions computed by COLMAP](gif/COLMAP.gif)
![Results rendered by NeRF](gif/scene.gif)

## Dependencies

- `imageio`
- `scikit-image`
- `opencv-python`
- `configargparse`
- `imageio-ffmpeg`

## Structure
```angular2html
.
├── LLFF
├── README.md
├── gif
│   ├── COLMAP.gif
│   └── scene.gif
├── nerf
│   ├── LICENSE
│   ├── README.md
│   ├── configs
│   │   ├── *.txt
│   ├── download_example_data.sh
│   ├── imgs
│   │   └── pipeline.jpg
│   ├── load_LINEMOD.py
│   ├── load_blender.py
│   ├── load_deepvoxels.py
│   ├── load_llff.py
│   ├── requirements.txt
│   ├── run_nerf.py
│   └── run_nerf_helpers.py
├── preprocess.py
└── run.sh
```

- `LLFF`: Codes from [LLFF](https://github.com/Fyusion/LLFF) for transform the COLMAP results to LLFF data
- `README.md` and `gif`: This file
- `nerf`: The rendering codes developed based on [NeRF](https://github.com/yenchenlin/nerf-pytorch)
- `preprocess.py`: This file is to preprocess original images and construct the configuration file for NeRF
- `run.sh`: The bash script to run the whole process

## Key Features

- Automatically extract a given number of images from a video, and generate the configuration file for NeRF. The code is in `./preprocess.py`
- Supports to transform the outputs of COLMAP into LLFF data for running NeRF. The code is in `./LLFF`
- Use Tensorboard to record the loss and PSNR when training and testing NeRF model. The code is in `./nerf/run_nerf.py`

## Preparing Data

Make a new directory named `./data/scene`.
Put the `video.mp4` in `./data/scene` if the input is a video, otherwise put all the images in the folder `./data/scene/images`.

Run the following commands to extract images from the video, and also generate the configuration file for NeRF.
```bash
$ python preprocess.py 
    --name scene                    # project name
    --from_video                    # obtain images from a video
    --num_images 100                # the number of images
```

Run the following commands to directly generate the configuration file for NeRF, if the provided data is already a set of images in `./data/scene/images`.
```bash
$ python preprocess.py 
    --name scene                    # project name
```

## Using COLMAP

Use the GUI of [COLMAP](https://github.com/colmap/colmap) to preprocess all the images. Specifically,
- create a new project and add the `./data/scene/images` folder as images
- run feature extraction and feature matching with default settings in COLMAP
- start reconstruction
- export the model and save in path `./data/scene/sparse/0`

The built model in COLMAP is shown [here](#introduction).
Then, run the following command in the `./LLFF` folder to obtain the LLFF data for NeRF.
```bash
$ python imgs2poses.py 
    --scenedir ../data/scene
```

All the original images and processed files can be downloaded from [this link (password: NNDL)](https://pan.baidu.com/s/1l_gyF6x9SNJnw8hQthxrZw).
It has a folder named `task3/data` which should be put in the current folder.

## Training NeRF

Run the following command in the `./nerf` folder to train the NeRF model, and it will automatically generate the rendered video.

```bash
$ python run_nerf.py 
    --config ../data/scene/config.txt
```

You can download our trained model from [this link (password: NNDL)](https://pan.baidu.com/s/1l_gyF6x9SNJnw8hQthxrZw).
It has a folder named `task3/logs`, which should be put in the current folder.

## Results

The gif generated from the rendered video is shown [here](#introduction).
And the final PSNR in test images is 30.82.