# construct the configuration file for NeRF
python preprocess.py --name scene

# transform data from COLMAP to LLFF
cd ./LLFF
python imgs2poses.py --scenedir ../data/scene

# run NeRF training the rendering
cd ./nerf
python run_nerf.py --config ../data/scene/config.txt