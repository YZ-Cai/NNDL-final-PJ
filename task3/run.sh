
cd ./LLFF
python imgs2poses.py --scenedir ../data/test

python gen_configs.py --name test

cd ./nerf
python run_nerf.py --config ../data/test/config.txt