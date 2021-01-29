#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import math
from functools import partial # for trials
from collections import OrderedDict # for dynamic configuration definition

import os # for paths
from pathlib import Path # for OS agnostic path definition

import numpy as np # for accuracy math

# allow configuration copying
from copy import deepcopy

import torch # for nn instantiation
import torch.nn as nn # for nn objects
import torch.nn.functional as F # for forward method
import torch.optim as optim # for optimization
import torchvision # for data transforms
import torchvision.transforms as transforms # for transform methods
from torch.utils.data import random_split # for train/test split

import ray
from ray import tune # for trialing
from ray.tune import JupyterNotebookReporter # for trial reporting
from ray.tune import CLIReporter # for trial reporting
# from ray.tune.integration.torch import is_distributed_trainable
# from torch.nn.parallel import DistributedDataParallel
# from ray.tune.integration.torch import DistributedTrainableCreator
# from ray.tune.integration.torch import distributed_checkpoint_dir
from ray.tune.schedulers import ASHAScheduler # for trial scheduling
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest.bayesopt import BayesOptSearch

import GPy
import sklearn

from ray.tune.suggest import ConcurrencyLimiter

import ConfigSpace as CS # for configuration bounds

import pandas as pd

import optuna
from optuna.integration import BoTorchSampler
from optuna.pruners import SuccessiveHalvingPruner


# In[ ]:


# set data and checkpoint locations
p = Path('.')
d = p / 'data'
r = p / 'ray_results'
l = p / 'checkpoints' / 'layers'
n = p / 'checkpoints' / 'layers'

# set computation location(s)
cpus = os.cpu_count() # number of cpu cores
gpus = torch.cuda.device_count()

# set number or fraction of processing units (per training worker) you'd like to utilize, if any at all
# cpu_use must be grater than zero
max_concurrent_trials = cpus
cpu_use = 1 # number of cpu cores to dedicate to 1 series of trials
gpu_use = gpus/max_concurrent_trials if gpus else 0

# set experiment hyperparameters
oom = 8 if gpus else 2 # order of magnitude
num_samples = 2 ** oom


# In[ ]:


# move data into sets for loading
def load_data(data_dir=d.absolute()):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset,testset = [torchvision.datasets.CIFAR10(root=data_dir, train=is_train, download=True, transform=transform) for is_train in [True,False]]

    return trainset, testset


# In[ ]:


def define_model(trial):
    # We optimize the number of layers, hidden untis and dropout ratio in each layer.
    n_convs = trial.suggest_int("n_conv_layers", 1, 3)
    n_fulls = trial.suggest_int("n_full_layers", 1, 4)

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
            if trial.suggest_int("has_max_pool_c{}".format(i),0,1) & pre_flat_size > 3:
                layers.append(nn.MaxPool2d(2, 2))
                pre_flat_size = int(pre_flat_size / 2)
            layers.append(nn.BatchNorm2d(out_channels))

        in_channels = out_channels
        out_kernel = kernel_size

    layers.append(nn.Flatten())

    in_features = in_channels * pre_flat_size**2
    for i in range(n_fulls):
        out_features = trial.suggest_int("n_l_units_l{}".format(i), *[2**x for x in [2,6]])
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.ReLU())
        if trial.suggest_int("has_dropout_l{}".format(i),0,1):
            p = trial.suggest_uniform("dropout_l{}".format(i), 0.2, 0.5)
            layers.append(nn.Dropout(p))
        layers.append(nn.LayerNorm(out_features))

        in_features = out_features

    layers.append(nn.Linear(in_features, 10))
    layers.append(nn.LogSoftmax(dim=1))

    return nn.Sequential(*layers)


# In[ ]:


def Net(arch):
    layers = []
    pre_flat_size = 32
    in_channels = 3
    out_kernel = None

    for i in range(arch["n_conv_layers"]):
        if pre_flat_size > 7:
            out_channels = arch["n_conv_channels_c%s" % i]
            kernel_size = arch["kernel_size_c%s" % i]
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size))
            pre_flat_size = pre_flat_size - kernel_size+1
            if arch["has_max_pool_c%s" % i] & pre_flat_size > 3:
                layers.append(nn.MaxPool2d(2, 2))
                pre_flat_size = int(pre_flat_size / 2)
            layers.append(nn.BatchNorm2d(out_channels))

        in_channels = out_channels
        out_kernel = kernel_size

    layers.append(nn.Flatten())

    in_features = in_channels * pre_flat_size**2
    for i in range(arch["n_full_layers"]):
        out_features = arch["n_l_units_l%s" % i]
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.ReLU())
        if arch["has_dropout_l%s" % i]:
            p = arch["dropout_l%s" % i]
            layers.append(nn.Dropout(p))
        layers.append(nn.LayerNorm(out_features))

        in_features = out_features

    layers.append(nn.Linear(in_features, 10))
    layers.append(nn.LogSoftmax(dim=1))

    return nn.Sequential(*layers)


# In[ ]:


# train nn on data
def train_cifar(non_arch_config,trial):
    loss,accuracy = 0,0
    lr = 10**-(non_arch_config["learning rate {10^(-⌊x⌋)"])
    batch_size = 2**int(non_arch_config["batch size {2^⌊x⌋}"])
    epochs = 10*int(non_arch_config["epochs {10⌊x⌋}"])

    net = define_model(trial) if type(trial) == optuna.trial.Trial else Net(trial.params)
    
    device = "cpu"

    if torch.cuda.is_available():
        device = "cuda"

    net.to(device)

    
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)

    trainset, testset = load_data()

    test_abs = int(len(trainset) * 0.8)
    train_subset, val_subset = random_split(
        trainset, [test_abs, len(trainset) - test_abs])

    trainloader,valloader = [torch.utils.data.DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2) for subset in [train_subset,val_subset]]

    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_steps = 0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

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
            
        loss = (val_loss / val_steps)
        accuracy = (correct / total)
        print("HP: ", non_arch_config,"\n", "Trial/Epoch: ", trial.number, "/", epoch, "Loss/Accuracy: ", loss,"/",accuracy)

    with tune.checkpoint_dir(step=trial.number) as checkpoint_dir:
        path = os.path.join(checkpoint_dir, "checkpoint")
        torch.save(
            (
                net.state_dict()
            ),
            path
        )
    return [loss,accuracy]


# In[ ]:


# model nn based on HPO
def model_cifar(non_arch_config,arch_config):
    loss,accuracy = 0,0
    lr = 10**-(non_arch_config["learning rate {10^(-⌊x⌋)"])
    batch_size = 2**int(non_arch_config["batch size {2^⌊x⌋}"])
    epochs = 10*int(non_arch_config["epochs {10⌊x⌋}"])
    
    print(arch_config)
    net = Net(arch_config)
    
    device = "cpu"

    if torch.cuda.is_available():
        device = "cuda"

    net.to(device)
    if gpus > 1:
        net = nn.DataParallel(net)

    
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)

    trainset, testset = load_data()

    test_abs = int(len(trainset) * 0.8)
    train_subset, val_subset = random_split(
        trainset, [test_abs, len(trainset) - test_abs])

    trainloader,valloader = [torch.utils.data.DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2) for subset in [train_subset,val_subset]]

    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_steps = 0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

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
            
        loss = (val_loss / val_steps)
        accuracy = (correct / total)
        print("HP: ", non_arch_config,"\n", "Trial/Epoch: ", Test, "/", epoch, "Loss/Accuracy: ", loss,"/",accuracy)
        
        
    return net


# In[ ]:


# get accuracy score
def test_accuracy(net, device="cpu"):
    _, testset = load_data()

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=4, shuffle=False, num_workers=2)

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

# In[ ]:


def search_training_hyperparameters():
    lr = {
        "name":"learning rate {10^(-⌊x⌋)",
        "bounds":[x for x in range(1,4)]
    }
    batch_size = {
        "name":"batch size {2^⌊x⌋}",
        "bounds":[x for x in range(6,9)]
    }
    epochs = {
        "name":"epochs {10⌊x⌋}",
        "bounds":[x for x in range(2,6)]
    }
    
    config_space = CS.ConfigurationSpace()
    config_space_dict,config_space_ray = {},{}
    
    #start ConfigSpace API
    config_space.add_hyperparameter(
        CS.UniformFloatHyperparameter(
            lr["name"],
            lr["bounds"][0],
            lr["bounds"][-1],
            log=True
        ))
    config_space.add_hyperparameter(
        CS.CategoricalHyperparameter(
            batch_size["name"], 
            batch_size["bounds"]
        ))
    config_space.add_hyperparameter(
        CS.CategoricalHyperparameter(
            epochs["name"], 
            epochs["bounds"]
        ))
    
    #start Ray Search Space API
    config_space_ray[lr["name"]] = tune.loguniform(lr["bounds"][0],lr["bounds"][-1])
    config_space_ray[batch_size["name"]] = tune.choice(batch_size["bounds"])
    config_space_ray[epochs["name"]] = tune.choice(categories=epochs["bounds"])
    
    #start Dragonfly Search Space API
    param_list = [
        {
            "name": lr["name"], 
            "type": "float", 
            "min": lr["bounds"][0], 
            "max": lr["bounds"][-1]
        },
        {
            "name": batch_size["name"], 
            "type": "discrete_numeric", 
            "items": ":".join([str(2**x) for x in batch_size["bounds"]])
        },
        {
            "name": epochs["name"], 
            "type": "discrete_numeric", 
            "items": ":".join([str(10*x) for x in epochs["bounds"]])
        }
    ]
    
    #start BayesOpt Search Space API
    config_space_dict[lr["name"]] = tune.uniform(lr["bounds"][0],lr["bounds"][-1])
    config_space_dict[batch_size["name"]] = tune.uniform(lower=batch_size["bounds"][0], upper=batch_size["bounds"][-1])
    config_space_dict[epochs["name"]] = tune.uniform(lower=epochs["bounds"][0], upper=epochs["bounds"][-1])
    
    #start Discrete Search Search Space API
    param_dict = {p["name"]:p["bounds"] for p in [lr,batch_size,epochs]}
    
    #start PB2 Space API
    min_max_param_dict = {p["name"]:[p["bounds"][0], p["bounds"][-1]] for p in [lr,batch_size,epochs]}
    
    return config_space_dict


# print(search_training_hyperparameters())

# In[ ]:


import inspect 
def nas_report(study,trial):
    best_session = study.best_trials[0]
    print("Trial stats (#{}):    Loss={}    Accuracy={}".format(trial.number,*(list(best_session.values))))
    print("Best params so far (#{}):    {}".format(best_session.number,best_session.params))

    finished_trials = list(filter(
        (lambda trial: trial.state.is_finished()),
        study.trials
    ))

    model_state = {}
    with tune.checkpoint_dir(step=best_session.number) as checkpoint_dir:
        path = os.path.join(checkpoint_dir, "checkpoint")
        model_state = torch.load(path)

    with tune.checkpoint_dir(step=trial.number) as checkpoint_dir:
        path = os.path.join(checkpoint_dir, "checkpoint")
        torch.save(
            (
                best_session.params,
                model_state
            ),
            path
        )

    
    result_zip = zip(["loss","accuracy"], list(best_session.values))
    results = {p:v for p,v in result_zip}
    tune.report(**results)


# In[ ]:


def search_neural_arch(non_arch_config,checkpoint_dir=None):

    optuna.logging.set_verbosity(optuna.logging.FATAL)
    
    study = optuna.create_study(
        directions=["minimize","maximize"],
        study_name=str(non_arch_config),
        sampler=BoTorchSampler(),
        pruner=SuccessiveHalvingPruner(),
        storage='sqlite:///na.db',
        load_if_exists=True
    )
    
    study.optimize(
        partial(train_cifar, non_arch_config),
        n_trials=oom,
#         n_jobs=2,
        gc_after_trial=True,
        callbacks=[nas_report]
    )


# In[ ]:


# perform neuron configuration trials
def search_neurons():
    neuron_config_space = search_training_hyperparameters()
    
    experiment_metrics = dict(metric="accuracy", mode="max")

    hpn = list(neuron_config_space.keys())
    
    #pre-load data to avoid races
    load_data()
    
    scheduler = ASHAScheduler(
        max_t=oom,
        reduction_factor=2,
#         grace_period=3,
        **experiment_metrics)
    search = BayesOptSearch(
        **experiment_metrics)
    search = ConcurrencyLimiter(
        search,
        max_concurrent=max_concurrent_trials)
    reporter = CLIReporter(
#         overwrite=True,
        parameter_columns=hpn,
#         max_progress_rows=num_samples,
        max_report_frequency=10,
        **experiment_metrics)
    result = tune.run(
        search_neural_arch,
        verbose=3,
        name="neurons",
        local_dir=r.absolute(),
        resources_per_trial={"cpu": cpu_use, "gpu": gpu_use},
#         max_failures=3,
        num_samples=num_samples,
        config=neuron_config_space,
        scheduler=scheduler,
        search_alg=search,
        queue_trials=True,
        progress_reporter=reporter)

    best_trial = result.get_best_trial("accuracy", "max", "last")
    

    print("Best training hyperparameters: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))

    best_checkpoint_dir = best_trial.checkpoint.value
    arch_state, model_state = torch.load(os.path.join(best_checkpoint_dir, "checkpoint"))

    best_trained_model = Net(arch_state)
    best_trained_model.load_state_dict(model_state)
    
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    best_trained_model.to(device)

    test_acc = test_accuracy(best_trained_model, device)
    print("Best trial test set accuracy: {}".format(test_acc))
    
    return best_trained_model


# In[ ]:


print("Resource usage can be viewed at port http://127.0.0.1:8265/ or higher")


# In[ ]:


model = search_neurons()


# In[ ]:


print(model)

