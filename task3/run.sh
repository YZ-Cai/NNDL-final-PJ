python preprocess.py --name scene --num_train 200 --num_test 200

cd ./LLFF
python imgs2poses.py --scenedir ../data/scene

cd ./nerf
python run_nerf.py --config ../data/scene/config.txt