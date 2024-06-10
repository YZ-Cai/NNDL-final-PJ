
# test resnet18
nohup python main.py --net resnet18 --device cuda:3 > test_resnet18.txt 2>&1 &

# test vit
nohup python main.py --net vit --device cuda:2 > test_vit.txt 2>&1 &