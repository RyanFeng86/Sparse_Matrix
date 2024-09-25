import os
import sys

cmd1 = "python main_mbp_admm.py --data='cifar10' --model_type='vgg16' --epochs=150 --density=0.25 --seed=500\
        --prune='magnitude' --growth='momentum' --start_admm_ep=151 --batch_size=128"
os.system(cmd1)
'''
cmd2 = "python main_mbp_admm.py --data='cifar10' --model_type='vgg16' --epochs=20 --density=0.1 --seed=50\
        --prune='magnitude' --growth='momentum' --start_admm_ep=10"
os.system(cmd2)
cmd3 = "python main.py --data='cifar100' --model_type='vgg16' --epochs=200 --density=0.1  --seed=50 --prune='magnitude' --growth='momentum' --decay_frequency=11730"
os.system(cmd3)
'''
