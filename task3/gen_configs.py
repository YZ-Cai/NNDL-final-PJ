import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str)
args = parser.parse_args()

content = f"""\
expname = {args.name}
basedir = ./logs
datadir = ../data/{args.name}
dataset_type = llff

factor = 8
llffhold = 8

N_rand = 1024
N_samples = 64
N_importance = 64

use_viewdirs = True
raw_noise_std = 1e0
"""

with open(f'./data/{args.name}/config.txt', 'w') as f:
    f.write(content)