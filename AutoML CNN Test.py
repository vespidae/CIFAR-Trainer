#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from ray import tune # for trialing
from ray.tune import CLIReporter # for trial reporting
from ray.tune import JupyterNotebookReporter # for trial reporting
from ray.tune.schedulers import ASHAScheduler # for trial scheduling
from ray.tune.schedulers import HyperBandForBOHB # for trial scheduling
from ray.tune.suggest.bohb import TuneBOHB # for trial selection/pruning
from ray.tune.suggest.dragonfly import DragonflySearch
from dragonfly.opt.gp_bandit import CPGPBandit
from ray.tune.schedulers import AsyncHyperBandScheduler
from dragonfly import load_config
from dragonfly.exd.experiment_caller import CPFunctionCaller, EuclideanFunctionCaller
from ray.tune.suggest.bayesopt import BayesOptSearch

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
from optuna.samplers import TPESampler
from optuna.multi_objective.samplers import MOTPEMultiObjectiveSampler
from optuna.pruners import SuccessiveHalvingPruner

# import numpy


# In[2]:


# set data and checkpoint locations
p = Path('.')
d = p / 'data'
r = p / 'ray_results'
l = p / 'checkpoints' / 'layers'
n = p / 'checkpoints' / 'layers'

# set computation location(s)
gpus = torch.cuda.device_count()
# target = 0
# device = "cuda:%s" % target if gpus else "cpu"

# set number or fraction of processing units (per training worker) you'd like to utilize, if any at all
# cpu_use must be grater than zero
cpu_use = 1# if gpus else 1
gpu_use = 1/gpus if gpus else 0

# set experiment hyperparameters
num_samples = 2 ** (5 if gpus else 2)
max_time = 10 * (3 if gpus else 2)
gpus_per_trial = gpus#2 if gpus else 0


# Since the neuron configuration we want is dependent upon the number of layers we have, we need to work flatten the feature space a bit. We can reduce the high-dminesional setups to a slightly less high-dminesional string of base-n nodes.

# # define feature space for hashing
# c_min = 3**2
# c_max = 3**5
# f_min = 2**2
# f_max = 2**6
# 
# c = c_max - c_min
# f = f_max - f_min
# 
# # conv = set(range(c_max)) - set(range(c_min))
# # full = set(range(f_max)) - set(range(f_min))
# conv = range(c_max)[c_min:]
# full = range(f_max)[f_min:]
# 
# c_comb = list(combinations_with_replacement(conv,2))
# f_comb = []
# for layers in range(1,5):
#     f_comb += list(combinations_with_replacement(full,layers))
# #     print("Fully connected layer %s range: %s" % (layers,len(f_comb)) )
# #     print("\n")
# 
# # for conversion from dec to whatever we end up using
# # most to least significant digit
# def numberToBase(n, b):
#     if n == 0:
#         return [0]
#     digits = []
#     while n:
#         digits.append(int(n % b))
#         n //= b
#     rev = digits[::-1]
#     return rev
# 
# def feature_spacing():
#     
#     # create empty list to store the 
#     # combinations 
#     unique_combinations = list(combinations([c_comb,f_comb],2))
#     total_uniques = len(unique_combinations)
#     total_points = total_uniques**2
#     total_cvs = len(c_comb)
#     total_fcs = len(f_comb)
#     
#     columns = ["base","nodes_req","sparcity","sparcity_pcnt","denoise_pcnt"]
#     values = [1,total_uniques,total_points - total_uniques,(total_points - total_uniques) / total_points,0]
#     
#     cf = []
#     
#     for layer in [total_cvs,total_fcs]:#,total_uniques]:
#         results = {
#             "base": [1],
#             "nodes_req": [total_uniques],
#             "sparcity": [total_points - total_uniques],
#             "max_necc_base_value":[0],
#             "nodes+_req": [0],
#             "subsparcity": [0],
#             "unexplained":[0],
#             "sparcity_pcnt": [(total_points - total_uniques) / total_points * 100],
#             "subsparcity_pcnt": [0],
#             "denoise_pcnt":[0],
#             "complexity":[0]
#         }
# 
#         report = pd.DataFrame(results)
#     
#         for base in range(2,17):
#             results["base"] = [base]
#             results["nodes_req"] = [math.ceil(math.log(layer,(base)))]
#             results["nodes+_req"] = [math.floor(math.log(layer,(base)))]
#             
#             results["sparcity"] = [base**math.ceil(math.log(layer,base)) - layer]
#             results["subsparcity"] = [-(base**math.floor(math.log(layer,base)) - layer)]
#             
#             results["sparcity_pcnt"] = [(base**math.ceil(math.log(layer,(base))) - base**math.log(layer,(base)))/(base**math.ceil(math.log(layer,(base))))*100]
#             results["subsparcity_pcnt"] = [-((base**math.floor(math.log(layer,(base))) - base**math.log(layer,(base)))/(base**math.floor(math.log(layer,(base))))*100)]
#             
# #             results["max_necc_base_value"] = [numberToBase((results["base"][0]**results["nodes+_req"][0]+results["subsparcity"][0]),results["base"][0])]
#             results["max_necc_base_value"] = [numberToBase(layer,base)]
#             results["unexplained"] = [(-(base**math.floor(math.log(layer,base)) - layer))*(math.floor(math.log(layer,(base))))]
#             
#             results["denoise_pcnt"] = [math.floor(((total_points-(math.ceil(math.log(layer,base)))**2)/total_points)*100)]
#         
#             results["complexity"] = [results["nodes_req"][0]*(results["sparcity"][0]+1)]
# 
#             report = report.append(pd.DataFrame(results))
#             
#             
#         report.index = [x for x in range(1, len(report.values)+1)]
#         report.drop([1],axis=0,inplace=True)
#         report.sort_values(["sparcity","unexplained","nodes+_req","subsparcity","sparcity_pcnt","base"],inplace=True)
#         
#         cf.append(report.iloc[0])
#     
#     return cf

# bases = feature_spacing()
# [print(r,"\n") for r in bases]
# 
# base_c = bases[0]["base"]
# base_f = bases[1]["base"]

# print("For the convolutional layers, base %s seems to allow us to use the fewest nodes with the lowest number of invalid configuration indices (sparcity)." % bases[0]["base"])
# print("For the linear layers, base %s seems to allow us to use the fewest nodes with the lowest number of invalid configuration indices (sparcity)." % bases[1]["base"])
# 
# # print("We can use the ")

# def base_to_dec(num_list, base):
#     num_list = num_list[::-1]
#     num = 0
#     for k in range(len(num_list)):
#         dig = num_list[k]
#         dig = int(dig)
#         num += dig*(base**k)
#     return num
# 
# def encode(config=[(24, 64),(13, 41)]):
#     iconv = c_comb.index(config[0])
#     ifull = f_comb.index(config[1])
#     
#     conv_hash = numberToBase(iconv,base_c)
#     full_hash = numberToBase(ifull,base_f)
#     
#     return [conv_hash,full_hash]
# 
# def decode(hash=([1, 7, 5, 0], [2, 9, 7])):
#     conv = base_to_dec(hash[0], base_c)
#     full = base_to_dec(hash[1], base_f)
# 
#     
#     return [c_comb[conv],f_comb[full]]

# In[3]:


# move data into sets for loading
def load_data(data_dir=d.absolute()):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset,testset = [torchvision.datasets.CIFAR10(root=data_dir, train=is_train, download=True, transform=transform) for is_train in [True,False]]

    return trainset, testset


# In[4]:


def define_model(trial):
    # We optimize the number of layers, hidden untis and dropout ratio in each layer.
    n_convs = trial.suggest_int("n_conv_layers", 1, 3)
    n_fulls = trial.suggest_int("n_full_layers", 1, 4)
#         print(n_convs,n_fulls)
    layers = []
    pre_flat_size = 32
    in_channels = 3
    out_kernel = None

    for i in range(n_convs):
        if pre_flat_size > 7:
            out_channels = trial.suggest_int("n_conv_channels_c{}".format(i), *[3**x for x in [2,5]])
            kernel_size = trial.suggest_int("kernel_size_c{}".format(i),2,5)
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size))
            pre_flat_size = pre_flat_size - kernel_size+1
#             print("post conv: ",pre_flat_size)
            if trial.suggest_int("has_max_pool_c{}".format(i),0,1) & pre_flat_size > 3:
                layers.append(nn.MaxPool2d(2, 2))
                pre_flat_size = int(pre_flat_size / 2)
#                 print("post pool: ",pre_flat_size)

        in_channels = out_channels
        out_kernel = kernel_size

#     self.convolution = nn.Sequential(*layers)

#     layers = []

    layers.append(nn.Flatten())

#     self.flattening = nn.Sequential(*layers)

#     layers = []
#         print("pre_flat_size:",pre_flat_size)
    in_features = in_channels * pre_flat_size**2
    for i in range(n_fulls):
        out_features = trial.suggest_int("n_l_units_l{}".format(i), *[2**x for x in [2,6]])
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.ReLU())
        if trial.suggest_int("has_dropout_l{}".format(i),0,1):
            p = trial.suggest_uniform("dropout_l{}".format(i), 0.2, 0.5)
            layers.append(nn.Dropout(p))

        in_features = out_features

    layers.append(nn.Linear(in_features, 10))
    layers.append(nn.LogSoftmax(dim=1))

#     self.linearizing = nn.Sequential(*layers)
    return nn.Sequential(*layers)


# In[5]:


# dynamically-generated nn that takes a 3-channel image and outputs a label
class Net(nn.Module):
    def __init__(self, arch):
        super(Net, self).__init__()
        # We optimize the number of layers, hidden untis and dropout ratio in each layer.
#         n_convs = trial.suggest_int("n_conv_layers", 1, 3)
#         n_fulls = trial.suggest_int("n_full_layers", 1, 4)
#         print(n_convs,n_fulls)
        layers = []
        pre_flat_size = 32
        in_channels = 3
        out_kernel = None

        for i in range(arch["n_conv_layers"]):
            if pre_flat_size > 7:
#                 out_channels = trial.suggest_int("n_conv_channels_c{}".format(i), *[3**x for x in [2,5]])
                out_channels = arch["n_conv_channels_c%s" % i]
#                 kernel_size = trial.suggest_int("kernel_size_c{}".format(i),2,5)
                kernel_size = arch["kernel_size_c%s" % i]
                layers.append(nn.Conv2d(in_channels, out_channels, kernel_size))
                pre_flat_size = pre_flat_size - kernel_size+1
    #             print("post conv: ",pre_flat_size)
#                 if trial.suggest_int("has_max_pool_c{}".format(i),0,1) & pre_flat_size > 3:
                if arch["has_max_pool_c%s" % i] & pre_flat_size > 3:
                    layers.append(nn.MaxPool2d(2, 2))
                    pre_flat_size = int(pre_flat_size / 2)
    #                 print("post pool: ",pre_flat_size)
            
            in_channels = out_channels
            out_kernel = kernel_size
            
#         self.convolution = nn.Sequential(*layers)
        
#         layers = []
        
        layers.append(nn.Flatten())
        
#         self.flattening = nn.Sequential(*layers)
        
#         layers = []
#         print("pre_flat_size:",pre_flat_size)
        in_features = in_channels * pre_flat_size**2
        for i in range(arch["n_full_layers"]):
#             out_features = trial.suggest_int("n_l_units_l{}".format(i), *[2**x for x in [2,6]])
            out_features = arch["n_l_units_l%s" % i]
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.ReLU())
#             if trial.suggest_int("has_dropout_l{}".format(i),0,1):
            if arch["has_dropout_l%s" % i]:
                p = arch["dropout_l%s" % i]
                layers.append(nn.Dropout(p))

            in_features = out_features
        
        layers.append(nn.Linear(in_features, 10))
        layers.append(nn.LogSoftmax(dim=1))
    
#         self.linearizing = nn.Sequential(*layers)
        self.layers = nn.Sequential(*layers)
        
#         [print(layer) for layer in [self.convolution,self.flattening,self.linearizing]]
        
#         print("New model: %s" % hidden_layers)
    def forward(self, x):
        x = self.layers(x)
        return x


# def define_device():
#     CANDIDATES = 2 # number of GPUs being used
#     cuda_tensor = torch.Tensor([[2,0],[0,2]])
#     print(str(torch.Tensor().cuda().device))
#     cuda_tensor = cuda_tensor.cuda()
#     print(cuda_tensor.get_device())
#     
#     it = [torch.Tensor([[1,0],[0,1]]) for i in range(CANDIDATES)]
#     index_tensors = [t.to("cuda:%s" % i) for i,t in enumerate(it)]
#     
#     cpu_tensor = torch.Tensor([[1,0],[0,1]])
#     cpu_tensor = cpu_tensor.to("cpu")
#     print(cpu_tensor.device)
#     
#     index_tensors.append(cpu_tensor)
#     
#     [print(t.device) for t in index_tensors]
#     [print(cuda_tensor*t) for t in index_tensors]
# define_device()

# In[6]:


# train nn on data
def train_cifar(trial,non_arch_config,checkpoint_dir=None):
    loss,accuracy = 0,0
    
    data_dir=d.absolute()
    
#     def cv_discrim(s): return 'conv_subindex_' in s
#     def fc_discrim(s): return 'full_subindex_' in s
#     cvs = [neuron_config[hp] for hp in list(filter(cv_discrim, neuron_config.keys()))]
#     fcs = [neuron_config[hp] for hp in list(filter(fc_discrim, neuron_config.keys()))]
    
#     cfg = decode([cvs, fcs])
    
    
#     net = Net(cfg)
    net = define_model(trial)
    
    device = "cpu"
    gpus = torch.cuda.device_count()
    if gpus:
        device = str(torch.Tensor().cuda().device)
    if gpus > 1:
#         torch.cuda.set_device(gpus-1)
#         net = nn.DataParallel(net)
        net = nn.DataParallel(model, device_ids=range(gpus))
#         torch.distributed.init_process_group(
#             backend='nccl', world_size=gpus, rank=0#, init_method='...'
#         )
#         net = DistributedDataParallel(net, device_ids=[gpus-1], output_device=gpus-1)
    else:
        net.to(device)

        
    print(net)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=non_arch_config["lr"], momentum=0.9)

    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        net.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    trainset, testset = load_data()

    test_abs = int(len(trainset) * 0.8)
    train_subset, val_subset = random_split(
        trainset, [test_abs, len(trainset) - test_abs])

    trainloader,valloader = [torch.utils.data.DataLoader(
        train_subset,
        batch_size=2**int(non_arch_config["batch_size"]),
        shuffle=True,
        num_workers=1) for subset in [train_subset,val_subset]]

    for epoch in range(10**int(non_arch_config["epochs"])):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_steps = 0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()
            
#             print("Input shape: ",inputs.shape)

            # forward + backward + optimize
            outputs = net(inputs)
#             [print(y) for y in [outputs, labels]]
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_steps += 1
#             if i % 2000 == 1999:  # print every 2000 mini-batches
            if i % 4000 == 3999:  # print every 4000 mini-batches
#                 print("Model: %s, Epoch: %d, Mini-batch: %5d, Loss: %.3f" % (cfg,epoch + 1, i + 1, running_loss / epoch_steps))
#                 if str(loss) == 'nan': print("outputs: %s, labels: %s, loss: %s" % (outputs, labels, loss))
#                 [print("%s: %s" % n,v) for n,v in zip(["outputs", "labels", "loss"],[outputs, labels, loss])]
                print("Epoch: %d, Mini-batch: %5d, Loss: %.3f" % (epoch + 1, i + 1, running_loss / epoch_steps))
                running_loss = 0.0

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        for i, data in enumerate(valloader, 0):
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss = criterion(outputs, labels)
                val_loss += loss.cpu().numpy()
                val_steps += 1

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((net.state_dict(), optimizer.state_dict()), path)
            
        loss = (val_loss / val_steps)
        accuracy = (correct / total)

        tune.report(accuracy=accuracy,loss=loss,model=net)
        
    return [loss,accuracy,net]


# In[7]:


# get accuracy score
def test_accuracy(net, device="cpu"):
    trainset, testset = load_data()

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=4, shuffle=False, num_workers=1)

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total


# #determine configuration boundary for nn based on number of layers
# nodes_c = bases[0]["nodes_req"]
# nodes_f = bases[1]["nodes_req"]
# max_c = bases[0]["max_necc_base_value"]
# max_f = bases[1]["max_necc_base_value"]

# In[8]:


def search_training_hyperparameters():
    config_space = CS.ConfigurationSpace()
    config_space_dict,config_space_ray = {},{}
    lr = [1e-4,1e-1]
    batch_size = [2**x for x in range(2,6)]
    epochs = [10*x for x in range(2,5)]
    
    #start ConfigSpace API
    config_space.add_hyperparameter(
        CS.UniformFloatHyperparameter(name="lr", lower=1e-4, upper=1e-1, log=True))
    config_space.add_hyperparameter(
        CS.CategoricalHyperparameter(name="batch_size", choices=[2**x for x in range(2,6)]))
    config_space.add_hyperparameter(
        CS.CategoricalHyperparameter(name="epochs", choices=[10*x for x in range(2,5)]))
    
    #start Ray Search Space API
    config_space_ray["lr"] = tune.loguniform(lower=1e-4, upper=1e-1)
    config_space_ray["batch_size"] = tune.choice(categories=[2**x for x in range(2,6)])
    config_space_ray["epochs"] = tune.choice(categories=[10*x for x in range(2,5)])
    
    #start Dragonfly Search Space API
    param_list = [
        {"name": "lr", "type": "float", "min": lr[0], "max": lr[1]},
        {"name": "batch_size", "type": "discrete_numeric", "items": ":".join([str(2**x) for x in range(2,6)])},
        {"name": "epochs", "type": "discrete_numeric", "items": ":".join([str(10*x) for x in range(2,5)])}
    ]
    
    #start BayesOpt Search Space API
    config_space_dict["lr"] = tune.uniform(lower=1e-4, upper=1e-1)
    config_space_dict["batch_size (2**x)"] = tune.uniform(lower=2, upper=6)
    config_space_dict["epochs (10x)"] = tune.uniform(lower=2, upper=5)
    
    return config_space_dict


# print(search_training_hyperparameters())

# In[9]:


def search_neural_arch(non_arch_config,checkpoint_dir=None):
#     study = optuna.create_study(
#         direction="maximize",
#         sampler=TPESampler(multivariate=True),
#         pruner=SuccessiveHalvingPruner(reduction_factor=2)
#     )
#     print(non_arch_config)
#     training_hyperparameters = {hp:v for hp,v in zip(["lr","batch_size","epochs"],non_arch_config["point"])}
    training_hyperparameters = non_arch_config
    
#     print(optuna.logging.get_verbosity())
#     optuna.logging.set_verbosity(optuna.logging.ERROR)
#     print(optuna.logging.get_verbosity())

    optuna.logging.disable_default_handler()
    
    study = optuna.multi_objective.create_study(
        ["minimize","maximize"],
        sampler=MOTPEMultiObjectiveSampler()
    )
    
    study.optimize((lambda trial: train_cifar(trial, training_hyperparameters)), n_trials=max_time, gc_after_trial=True)
#     study.optimize((lambda trial: train_cifar(trial, training_hyperparameters, target)), n_trials=1)
    
    print("Best trial:")
    trial = study.best_trial if (type(study) == optuna.study.Study) else study.get_pareto_front_trials()[0]
#     return trial
    print("  Loss: %s  Accuracy: %s" % tuple(trial.values))
    
    print("  Params: ")
    for (key, value) in trial.params.items():
        print("    {}: {}".format(key, value))
#     tune.report(loss=loss,accuracy=accuracy,model=trial)
    tune.report(loss=trial.values[0], accuracy=trial.values[1], model=trial.params)
    return {"loss":trial.values[0], "accuracy":trial.values[1], "model":trial.params}


# key=["batch_size","epochs","lr"]
# val=[4,1,0.000522636]
# k_to_v = {k:v for k,v in zip(key,val)}
# res = search_neural_arch({"point":val})

# # print(res[0].values)
# [print({i:v}) for (i,v) in list(res.items())]

# mdl = Net(res["model"])
# print(mdl)

# In[10]:


# perform neuron configuration trials
def search_neurons():    
    neuron_config_space = search_training_hyperparameters()
    
#     param_dict = {"name": "training_hyperparameters", "domain": neuron_config_space}
#     domain_config = load_config(param_dict)
#     domain, domain_orderings = domain_config.domain, domain_config.domain_orderings

#     # define the hpo search algorithm BO
#     func_caller = CPFunctionCaller(None, domain, domain_orderings=domain_orderings)
#     optimizer = CPGPBandit(func_caller, ask_tell_mode=True)
# #     bo_search_alg = DragonflySearch(optimizer, metric="validation_mae", mode="min")
    
    experiment_metrics = dict(metric="accuracy", mode="max")
#     hpn = [p["name"] for p in neuron_config_space]
#     hpn = neuron_config_space.get_hyperparameter_names()
    hpn = list(neuron_config_space.keys())
#     hpn.append("model")
    
    #pre-load data to avoid races
    load_data()
    
    scheduler = ASHAScheduler(
        max_t=max_time,
        reduction_factor=2,
        **experiment_metrics)
    search = BayesOptSearch(
#         optimizer=optimizer,
#         neuron_config_space,
        **experiment_metrics)
    search = ConcurrencyLimiter(
        search,
        max_concurrent=8)
    reporter = JupyterNotebookReporter(
        overwrite=True,
        parameter_columns=hpn,
        metric_columns=["loss", "accuracy", "training_iteration"])
    result = tune.run(
        partial(search_neural_arch),
        verbose=3,
        name="neurons",
        local_dir=r.absolute(),
        resources_per_trial={"cpu": cpu_use, "gpu": gpu_use},
        max_failures=3,
        num_samples=num_samples,
        config=neuron_config_space,
        scheduler=scheduler,
        search_alg=search,
        queue_trials=True,
        progress_reporter=reporter)

    best_trial = result.get_best_trial("accuracy", "max", "last")
    
#     def cv_discrim(s): return 'conv_subindex_' in s
#     def fc_discrim(s): return 'full_subindex_' in s
    def other_discrim(s): return 'subindex' not in s
#     best_cvs = [best_trial.config[hp] for hp in list(filter(cv_discrim, best_trial.config.keys()))]
#     best_fcs = [best_trial.config[hp] for hp in list(filter(fc_discrim, best_trial.config.keys()))]
    best_other = [best_trial.config[hp] for hp in list(filter(other_discrim, best_trial.config.keys()))]

#     cfg = decode([best_cvs, best_fcs])
    
#     conv_report = ["Connolutional Layer %s: %s" % (i,c) for i,c in enumerate(cfg[0],1)]
#     full_report = ["Fully-connected Layer %s: %s" % (i,f) for i,f in enumerate(cfg[1],1)]
#     other_report = ["%s: %s" % (hp,f) for (hp,f) in zip(["Batch Size","Epochs","Learning Rate"],best_other)]
    
#     best_trained_model = Net(best_trial.last_result["model"])
    best_trained_model = best_trial.last_result["model"]
    best_training_hyperparameters = zip(["Batch Size","Epochs","Learning Rate"],best_other)

#     print("Best trial config: {}".format(best_trial.config))
    print("Best trial config:")
    print(best_trained_model)
    print(best_trial.config)
#     [print(best) for best in [conv_report,full_report,other_report]]
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))
    
    
    device = "cpu"
    gpus = torch.cuda.device_count()
    if gpus:
        device = str(torch.Tensor().cuda().device)
    if gpus > 1:
        net = nn.DataParallel(model, device_ids=range(gpus))
    else:
        net.to(device)
    
#     device = "cpu"
#     if gpus_per_trial:
#         device = str(torch.Tensor().cuda().device)
#     if gpus_per_trial > 1:
#         best_trained_model = nn.DataParallel(best_trained_model)
#     best_trained_model.to(device)

    best_checkpoint_dir = best_trial.checkpoint.value
    model_state, optimizer_state = torch.load(os.path.join(
        best_checkpoint_dir, "checkpoint"))
    best_trained_model.load_state_dict(model_state)

    test_acc = test_accuracy(best_trained_model, device)
    
#     if checkpoint_dir != None:
#         tune.report(accuracy=test_acc)
    
    print("Best trial test set accuracy: {}".format(test_acc))
    
    return (best_trained_model, dict(best_training_hyperparameters))


# # perform test
# if __name__ == "__main__":
#     # You can change the number of GPUs per trial here:
#     model = search_layers(num_samples=10, max_num_epochs=10, gpus_per_trial=1)
# 

# In[11]:


print("Resource usage can be viewed at port http://127.0.0.1:8265/ or higher")


# In[ ]:


model,trainers = search_neurons()


# In[ ]:


print(model)


# In[ ]:




