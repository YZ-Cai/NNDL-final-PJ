import os
import cv2
import random
import argparse


def extract_frames(data_path, num_images):
    video = cv2.VideoCapture(f'{data_path}/video.mp4')
    os.makedirs(f'{data_path}/images', exist_ok=True)

    # sample images
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = random.sample(range(total_frames), num_images)
    
    count = 0
    success = True
    while success and frame_indices:
        success, image = video.read()
        if not success:
            break
        if count in frame_indices:
            cv2.imwrite(f'{data_path}/images/frame_{count}.jpg', image)
        count += 1
    video.release()
    
    
# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str)
parser.add_argument('--from_video', action='store_true', default=False)
parser.add_argument('--num_images', type=int, default=100)
args = parser.parse_args()

# prepare data
if args.from_video:
    extract_frames(f'./data/{args.name}', args.num_images)

# write configs
content = f"""\
expname = {args.name}
basedir = ../logs
datadir = ../data/{args.name}
dataset_type = llff

factor = 2
llffhold = 8

N_rand = 1024
N_samples = 64
N_importance = 64

use_viewdirs = True
raw_noise_std = 1e0
"""
with open(f'./data/{args.name}/config.txt', 'w') as f:
    f.write(content)