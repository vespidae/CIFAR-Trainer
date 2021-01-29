#!/usr/bin/env python
# coding: utf-8

# In[2]:


from functools import partial # for trials
import numpy as np # for accuracy math
import os # for paths
import torch # for nn instantiation
import torch.nn as nn # for nn objects
import torch.nn.functional as F # for forward method
import torch.optim as optim # for optimization
from torch.utils.data import random_split # for train/test split
import torchvision # for data transforms
import torchvision.transforms as transforms # for transform methods
import ray
from ray import tune # for trialing
# from ray.tune import CLIReporter # for trial reporting
from ray.tune import JupyterNotebookReporter # for trial reporting
from ray.tune.integration.torch import is_distributed_trainable
from torch.nn.parallel import DistributedDataParallel
from ray.tune.integration.torch import DistributedTrainableCreator
from ray.tune.integration.torch import distributed_checkpoint_dir
from ray.tune.schedulers import ASHAScheduler # for trial scheduling
# from ray.tune.schedulers import HyperBandForBOHB # for trial scheduling
# from ray.tune.suggest.bohb import TuneBOHB # for trial selection/pruning
# from ray.tune.suggest.dragonfly import DragonflySearch
# from dragonfly.opt.gp_bandit import CPGPBandit
from ray.tune.schedulers import AsyncHyperBandScheduler
# from dragonfly import load_config
# from dragonfly.exd.experiment_caller import CPFunctionCaller, EuclideanFunctionCaller
from ray.tune.suggest.bayesopt import BayesOptSearch

import GPy
import sklearn
from ray.tune.schedulers import pb2
from ray.tune.schedulers.pb2 import PB2

from ray.tune.suggest import ConcurrencyLimiter

import ConfigSpace as CS # for configuration bounds
from collections import OrderedDict # for dynamic configuration definition
from pathlib import Path # for OS agnostic path definition

# import itertools package 
import itertools 
from itertools import combinations, combinations_with_replacement
from itertools import product

import math
import pandas as pd

# allow configuration copying
from copy import deepcopy

import optuna
# from optuna.samplers import TPESampler
from optuna.multi_objective.samplers import MOTPEMultiObjectiveSampler
# from optuna.pruners import SuccessiveHalvingPruner

# from ray.tune.schedulers.pb2_utils import normalize, optimize_acq, select_length, UCB, standardize, TV_SquaredExp


# In[3]:


# # set data and checkpoint locations
# p = Path('.')
# d = p / 'data'
# r = p / 'ray_results'
# l = p / 'checkpoints' / 'layers'
# n = p / 'checkpoints' / 'layers'

# # set computation location(s)
# gpus = torch.cuda.device_count()
# device = "cuda:0" if gpus else "cpu"

# # set number or fraction of processing units (per training worker) you'd like to utilize, if any at all
# # cpu_use must be grater than zero
# cpu_use = 1 if gpus else 0.5
# gpu_use = 0.25 if gpus else 0

# # set experiment hyperparameters
# num_samples = 2 ** (6 if gpus else 4)
# max_time = 10 * (4 if gpus else 1)
# gpus_per_trial = 0.5 if gpus else 0

# set data and checkpoint locations
p = Path('.')
dt = p / 'data'
r = p / 'ray_results'
l = p / 'checkpoints' / 'layers'
n = p / 'checkpoints' / 'layers'

# set computation location(s)
cpus = os.cpu_count()
gpus = torch.cuda.device_count()

# set number or fraction of processing units (per training worker) you'd like to utilize, if any at all
# cpu_use must be grater than zero
max_concurrent_trials = cpus#int(cpus/2) if cpus > 1 else cpus
cpu_use = 2#2 / gpus if gpus else 1
gpu_use = 0.5#gpus/max_concurrent_trials if gpus else 0

# set experiment hyperparameters
# oom = 2 if gpus else 3 # order of magnitude
num_samples = 2 / gpu_use#** oom
max_time = 10# * oom


# Since the neuron configuration we want is dependent upon the number of layers we have, we need to work flatten the feature space a bit. We can reduce the high-dminesional setups to a slightly less high-dminesional string of base-n nodes.

# In[ ]:


# define feature space for hashing
c_min = 3**2
c_max = 3**5
k_min = 2
k_max = 5
m_min = 0
m_max = 1
f_min = 2**2
f_max = 2**6
d_min = 5
d_max = 11

c = c_max - c_min # convolutional layer options
k = k_max - k_min # kernel options
m = m_max - m_min # max pool layer options
f = f_max - f_min # fully connected layer options
d = d_max - d_min # dropout layer options

# conv = set(range(c_max)) - set(range(c_min))
# full = set(range(f_max)) - set(range(f_min))
conv = list(range(c_max)[c_min:])
kern = list(range(k_max)[k_min:])
maxp = list(range(m_max)[m_min:])
full = list(range(f_max)[f_min:])
drop = list(range(d_max)[d_min:])

# c_comb = list(combinations_with_replacement(conv,2))
c_comb = []
k_comb = []
m_comb = []
f_comb = []
d_comb = []

for layers in range(1,4):
    cm = product(conv,kern,maxp,repeat=1)
    c_comb += combinations_with_replacement(cm,layers)
# print(conv)
# print(maxp)
# print(c_comb)
pd.DataFrame(c_comb).to_csv("c_comb.csv")
for layers in range(1,2):
    m_comb += combinations_with_replacement(maxp,layers)
for layers in range(1,5):
    fd = product(full,drop)
    f_comb += combinations_with_replacement(fd,layers)
# print(full)
# print(drop)
# print(f_comb)
pd.DataFrame(f_comb).to_csv("f_comb.csv")
for layers in range(1,2):
    d_comb += combinations_with_replacement(drop,layers)
#     print("Fully connected layer %s range: %s" % (layers,len(f_comb)) )
#     print("\n")
# [pd.DataFrame(comb).to_csv("%s.csv" % name) for name,comb in zip(["c_comb","m_comb","f_comb","d_comb"],[c_comb,m_comb,f_comb,d_comb])]

# for conversion from dec to whatever we end up using
# most to least significant digit
def numberToBase(n, b):
    if n == 0:
        return [0]
    digits = []
    while n:
        digits.append(int(n % b))
        n //= b
    rev = digits[::-1]
    return rev

def feature_spacing():
    
    # create empty list to store the 
    # combinations 
#     conv_combinations = list(combinations([c_comb,m_comb],2))
#     print("np.shape(conv_combinations[0])",np.shape(conv_combinations[0]))
# #     pd.DataFrame(conv_combinations).to_csv('conv_combinations.csv')
#     full_combinations = list(combinations([f_comb,d_comb],2))
#     print("np.shape(full_combinations[0])",np.shape(full_combinations[0]))
#     pd.DataFrame(full_combinations).to_csv('full_combinations.csv')
    unique_combinations = product(c_comb,f_comb)
    print(len(unique_combinations))
#     print("np.shape(unique_combinations)",np.shape(unique_combinations))
    pd.DataFrame(unique_combinations).to_csv('unique_combinations.csv')
#     [pd.DataFrame(comb).to_csv("%s.csv" % name) for name,comb in zip(["conv_combinations","full_combinations","unique_combinations"],[conv_combinations,full_combinations,unique_combinations])]
    total_uniques = len(unique_combinations)
    total_points = total_uniques**2
    total_cvs = len(c_comb)
    total_krn = len(k_comb)
    total_mxp = len(m_comb)
    total_fcs = len(f_comb)
    total_drp = len(d_comb)
    
    columns = ["base","nodes_req","sparcity","sparcity_pcnt","denoise_pcnt"]
    values = [1,total_uniques,total_points - total_uniques,(total_points - total_uniques) / total_points,0]
    
    cf = []
    
    for layer in [total_cvs,total_mxp,total_fcs,total_drp]:#,total_uniques]:
        results = {
            "base": [1],
            "nodes_req": [total_uniques],
            "sparcity": [total_points - total_uniques],
            "max_necc_base_value":[0],
            "nodes+_req": [0],
            "subsparcity": [0],
            "unexplained":[0],
            "sparcity_pcnt": [(total_points - total_uniques) / total_points * 100],
            "subsparcity_pcnt": [0],
            "denoise_pcnt":[0],
            "complexity":[0]
        }

        report = pd.DataFrame(results)
    
        for base in range(2,101):
            results["base"] = [base]
            results["nodes_req"] = [math.ceil(math.log(layer,(base)))]
            results["nodes+_req"] = [math.floor(math.log(layer,(base)))]
            
            results["sparcity"] = [base**math.ceil(math.log(layer,base)) - layer]
            results["subsparcity"] = [-(base**math.floor(math.log(layer,base)) - layer)]
            
            results["sparcity_pcnt"] = [(base**math.ceil(math.log(layer,(base))) - base**math.log(layer,(base)))/(base**math.ceil(math.log(layer,(base))))*100]
            results["subsparcity_pcnt"] = [-((base**math.floor(math.log(layer,(base))) - base**math.log(layer,(base)))/(base**math.floor(math.log(layer,(base))))*100)]
            
#             results["max_necc_base_value"] = [numberToBase((results["base"][0]**results["nodes+_req"][0]+results["subsparcity"][0]),results["base"][0])]
            results["max_necc_base_value"] = [numberToBase(layer,base)]
            results["unexplained"] = [(-(base**math.floor(math.log(layer,base)) - layer))*(math.floor(math.log(layer,(base))))]
            
            results["denoise_pcnt"] = [math.floor(((total_points-(math.ceil(math.log(layer,base)))**2)/total_points)*100)]
        
            results["complexity"] = [results["nodes_req"][0]*(results["sparcity"][0]+1)]

            report = report.append(pd.DataFrame(results))
            
            
        report.index = [x for x in range(1, len(report.values)+1)]
        report.drop([1],axis=0,inplace=True)
        report.sort_values(["nodes+_req","sparcity","unexplained","subsparcity","sparcity_pcnt","base"],inplace=True)
        
        cf.append(report.iloc[0])
    
    return cf

bases = feature_spacing()
[print(r,"\n") for r in bases]

base_c = bases[0]["base"]
base_m = bases[1]["base"]
base_f = bases[2]["base"]
base_d = bases[3]["base"]


# In[ ]:




