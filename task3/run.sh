python preprocess.py --name scene

cd ./LLFF
python imgs2poses.py --scenedir ../data/scene

cd ./nerf
python run_nerf.py --config ../data/scene/config.txt