#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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
import ConfigSpace as CS # for configuration bounds
from collections import OrderedDict # for dynamic configuration definition
from pathlib import Path # for OS agnostic path definition

# import itertools package 
import itertools 
from itertools import permutations
from itertools import product

import math

import pandas as pd

import time


# In[ ]:


# set data and checkpoint locations
p = Path('.')
d = p / 'data'
r = p / 'ray_results'
l = p / 'checkpoints' / 'layers'
n = p / 'checkpoints' / 'layers'

## set number (or fraction) of GPUs (per training loop) you'd like to utilize if any at all
cpu_use = 1
gpu_use = 1


# #define feature space for hashing
# def feature_spacing():
#     conv = set(range(3**5)) - set(range(3**2))
#     full = set(range(2**5)) - set(range(2**2))
#     
#     c = 3**5 - 3**2
#     f = 2**5 - 2**2
#     
#     # create empty list to store the 
#     # combinations 
#     unique_combinations = []
#     total_uniques = 0
#     total_points = 1
#     
#     # do combo
# #     for combo in product(conv,conv,full):
# #         unique_combinations.append(combo)
#         
# #     for combo in product(conv,conv,full,full):
# #         unique_combinations.append(combo)
#         
# #     for combo in product(conv,conv,full,full,full):
# #         unique_combinations.append(combo)
#         
# #     for combo in product(conv,conv,full,full,full,full):
# #         unique_combinations.append(combo)
# 
#     for ls in range(0,4):
# #         print(ls)
#         unique_combinations.append((c**2)*(f*(f+1)**ls))
#         total_uniques += (c**2)*f*((f+1)**ls)
# #         total_points = ((c**2)*f*((f+1)**ls))
#     
#     total_uniques -= ((c**2)*f)
#     total_points = total_uniques**2
# #     print("number of combos: %s" % ["%s-fc model: %s" % (l,v) for l,v in enumerate(unique_combinations, 1)])
# #     print("total uniques:",total_uniques)
# #     print("number of points/indices (with sparicities/noise): %s" % total_points)
# #     print("\n")
#     
#     columns = ["base","nodes_req","sparcity","sparcity_pcnt","denoise_pcnt"]
#     values = [1,total_uniques,total_points - total_uniques,(total_points - total_uniques) / total_points,0]
#     results = {
#         "base": [1],
#         "nodes_req": [total_uniques],
#         "sparcity": [total_points - total_uniques],
#         "sparcity_pcnt": [(total_points - total_uniques) / total_points * 100],
#         "denoise_pcnt":[0]
#     }
#     
#     report = pd.DataFrame(results)
#     
# #     print(report.to_string())
#     
#     for base in range(2,11):
#         results["base"] = [base]
#         results["nodes_req"] = [math.ceil(math.log(total_uniques,(base)))]
# # #         print("number of base %s complex nodes required:" % (base), math.ceil(math.log(total_uniques,(base))))
# #         print("number of base %s complex nodes required:" % (base), results["nodes_req"])
#         results["sparcity"] = [base**math.ceil(math.log(total_uniques,base)) - total_uniques]
# # #         print("sparcity:",base**math.ceil(math.log(total_uniques,base)) - total_uniques,'points')
# #         print("sparcity:",results["sparcity"],'points')
#         results["sparcity_pcnt"] = [(base**math.ceil(math.log(total_uniques,(base))) - base**math.log(total_uniques,(base)))/(base**math.ceil(math.log(total_uniques,(base))))*100]
# # #         print("sparcity percentage:",(base**math.ceil(math.log(total_uniques,(base))) - base**math.log(total_uniques,(base)))/(base**math.ceil(math.log(total_uniques,(base))))*100,'%')
# #         print("sparcity percentage:",results["sparcity percentage"],'%')
# #         print("%s root-%s nodes per layer" % (math.ceil(math.log(total_uniques,base+1)),base+1))
# #         print("\n")
#         results["denoise_pcnt"] = [math.floor(((total_points-(math.ceil(math.log(total_uniques,base)))**2)/total_points)*100)]
# # #         print("noise reduced from total points:",math.floor(((total_points-(math.ceil(math.log(total_uniques,base)))**2)/total_points)*100),'%')
# #         print("noise reduced from total points:",results["denoise_pcnt"],'%')
#     
#         report = report.append(pd.DataFrame(results))
# #     for root in range(1,8):
# #         print("ceilinged %s-root (%s-value per number component) of combos with complex numbers: %s\n" % (root*2, root+1, [[math.ceil(combo**(1/(root*2))),"sparsity: %s%s" % ((math.ceil(combo**(1/(root*2))) - combo**(1/(root*2)))/combo**(1/(root*2))*100,'%')] for combo in unique_combinations]))
#     
#     print(report.sort_values(["sparcity_pcnt","nodes_req","base"]).to_string())
# #     report.head()
#     
# #     print(len(product(conv,conv,full)))
# #     print(f1)
# feature_spacing()

# base = 8
# c = 3**5 - 3**2
# f = 2**5 - 2**2
# def decode(code=None):
#     conv = []
#     full = []
#     
#     print(math.ceil(math.log(c,base)))
#     print(base**math.ceil(math.log(c,base)) - c)
#     print(math.ceil(math.log(f,base)))
#     print(base**math.ceil(math.log(f,base)) - f)
#     
#     model = [conv,full]
# #     return model
#     print()
#     
# decode()
# [print(math.log(278,b)) for b in range(2,9)]

# # math.sqrt(3**5)
# poss = ((2**5 - 2**2)+1)**3
# print("%s possibilities" % poss)
# [print("%s root-%s nodes per layer" % (math.ceil(math.log(poss,root)),root)) for root in range(2,5)]
# [print("%s root-%s nodes per layer" % (math.log(poss,root),root)) for root in range(2,5)]
# # print(math.ceil(poss**(1/2)))
# # print(math.log(poss,2))

# (2**5 - 2**2)+1

# a = ['1', '2', '3']
# b = ['1', '2', '3']
# c = ['1', '2', '3']
# d = ['1', '2', '3']
# 
# # for r in product(product(a, b, d),c): print(r)
# r = [comb for comb in product(a, b, d)]
# print(r)

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


# dynamically-generated nn that takes a 3-channel image and outputs a label
class Net(nn.Module):
    def __init__(self, hidden_layers=[[6, 16],[120,84]]):
        super(Net, self).__init__()
        hidden_convs,hidden_fcs = hidden_layers
        print(hidden_convs)
        print(hidden_fcs)
        uf_input = 0
        layer_list = OrderedDict()
        
        layer_list['conv1'] = nn.Conv2d(3, hidden_convs[0], 5)
        layer_list['pool1'] = nn.MaxPool2d(2, 2)

        layer_input = layer_list['conv1'].out_channels
        
        for layer_num, channels in enumerate(hidden_convs[1:], 2):
            layer_list["conv%s" % layer_num]  = nn.Conv2d(layer_input, channels, 5)
            layer_list["pool%s" % layer_num] = nn.MaxPool2d(2, 2)
            layer_input = layer_list["conv%s" % layer_num].out_channels
        
        
        layer_list["flat"] = nn.Flatten()
        
        layer_list['fc1'] = nn.Linear(layer_input*5*5, hidden_fcs[0])
        layer_list["relu1"]  = nn.ReLU()
        
        layer_input = layer_list['fc1'].out_features
        for (layer_num, features) in enumerate(hidden_fcs[1:], 2):
            layer_list["fc%s" % layer_num]  = nn.Linear(layer_input, features)
            layer_list["relu%s" % layer_num]  = nn.ReLU()
            layer_input = layer_list["fc%s" % layer_num].out_features
            
        
        layer_list['fco'] = nn.Linear(hidden_fcs[-1], 10)
    
        self.layers = nn.Sequential(layer_list)

    def forward(self, x):
        x = self.layers(x)
        return x


# In[ ]:


# train nn on data
def train_cifar(neuron_config, checkpoint_dir=None):
    
    data_dir=d.absolute()
    
    def cv_discrim(s): return 'cv' in s
    def fc_discrim(s): return 'fc' in s
    cvs = [neuron_config[hp] for hp in list(filter(cv_discrim, neuron_config.keys()))]
    fcs = [neuron_config[hp] for hp in list(filter(fc_discrim, neuron_config.keys()))]
#     cvs = neuron_config["cvs"]
#     fcs = neuron_config["fcs"]
    
    net = Net([cvs, fcs])

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=neuron_config["lr"], momentum=0.9)

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
        batch_size=int(neuron_config["batch_size"]),
        shuffle=True,
        num_workers=1) for subset in [train_subset,val_subset]]

    for epoch in range(neuron_config["epochs"]):  # loop over the dataset multiple times
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

            # print statistics
            running_loss += loss.item()
            epoch_steps += 1
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1,
                                                running_loss / epoch_steps))
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

        tune.report(loss=(val_loss / val_steps), accuracy=(correct / total))
    print("Finished Training")


# In[ ]:


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


# In[ ]:


#determine configuration boundary for nn based on number of layers
def configure_neurons(num_convs,num_fcs):
    config_space = CS.ConfigurationSpace()
    
    config_space.add_hyperparameter(
        CS.UniformFloatHyperparameter(name="lr", lower=1e-4, upper=1e-1, log=True))
    config_space.add_hyperparameter(
        CS.CategoricalHyperparameter(name="batch_size", choices=[4, 8, 16, 32]))
    config_space.add_hyperparameter(
        CS.CategoricalHyperparameter(name="epochs", choices=[20, 30, 40]))
    
    for hidden in range(2):
        config_space.add_hyperparameter(
            CS.UniformIntegerHyperparameter("cv%s" % hidden, lower=3, upper=3**4))
    
    for hidden in range(num_fcs):
        config_space.add_hyperparameter(
            CS.UniformIntegerHyperparameter("fc%s" % hidden, lower=2**2, upper=2**6))
        
    return config_space

# def configure_neurons():
#     config_space = {
#         "batch_size_seed": tune.randint(2, 6),
#         "cv_seed": tune.grid_search([2]),
#         "fc_seed": tune.randint(2, 4),
        
#         "lr": tune.loguniform(1e-4,1e-1),
#         "batch_size": tune.sample_from(lambda spec: 2**spec.config.batch_size_seed),
#         "epochs": tune.qrandint(20, 40, 10),
        
#         "cvs": tune.sample_from(lambda spec: [tune.randint(3, 3**4) for layer in range(spec.config.cv_seed)]),
#         "fcs": tune.sample_from(lambda spec: [tune.randint(2**2, 2**4) for layer in range(spec.config.fc_seed)])        
#     }
        
#     return config_space


# neuron_config_space = configure_neurons()
# print(neuron_config_space)

# In[ ]:


# perform neuron configuration trials
def search_neurons(layer_config, checkpoint_dir=None):
    num_samples=20
    max_num_epochs=20
    gpus_per_trial=1
    
#     print(layer_config)
    
    neuron_config_space = configure_neurons(layer_config["num_convs"], layer_config["num_fcs"])
#     neuron_config_space = configure_neurons()
    
    experiment_metrics = dict(metric="accuracy", mode="max")
    
    scheduler = HyperBandForBOHB(
#         metric="loss",
#         mode="min",
        max_t=10,
        reduction_factor=2,
        **experiment_metrics)
    search = TuneBOHB(
        neuron_config_space,
        max_concurrent=4,
#         metric="loss",
#         mode="min",
        **experiment_metrics)
    reporter = JupyterNotebookReporter(
        overwrite=True,
#         parameter_columns=["l1", "l2", "lr", "batch_size", "epochs"],
        parameter_columns=neuron_config_space.get_hyperparameter_names(),
        metric_columns=["loss", "accuracy", "training_iteration"])
    result = tune.run(
        partial(train_cifar),
        verbose=2,
        name="neurons",
        local_dir=r.absolute(),
        resources_per_trial={"cpu": cpu_use, "gpu": gpu_use},
        max_failures=3,
#         config=neuron_config_space,
        num_samples=num_samples,
        scheduler=scheduler,
        search_alg=search,
        progress_reporter=reporter)

    best_trial = result.get_best_trial("accuracy", "max", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))

    
    def cv_discrim(s): return 'cv' in s
    def fc_discrim(s): return 'fc' in s
    best_cvs = [best_trial.config[hp] for hp in list(filter(cv_discrim, best_trial.config.keys()))]
    best_fcs = [best_trial.config[hp] for hp in list(filter(fc_discrim, best_trial.config.keys()))]
# #     best_trained_model = Net(best_trial.config["l1"], best_trial.config["l2"])
    
#     best_trained_model = Net(best_trial.config["cvs"], best_trial.config["fcs"])
    best_trained_model = Net([best_cvs, best_fcs])
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if gpus_per_trial > 1:
            best_trained_model = nn.DataParallel(best_trained_model)
    best_trained_model.to(device)

    best_checkpoint_dir = best_trial.checkpoint.value
    model_state, optimizer_state = torch.load(os.path.join(
        best_checkpoint_dir, "checkpoint"))
    best_trained_model.load_state_dict(model_state)

    test_acc = test_accuracy(best_trained_model, device)
    
    if checkpoint_dir != None:
        tune.report(accuracy=test_acc)
    
#     with tune.checkpoint_dir("nodes") as checkpoint_dir:
#         path = os.path.join(checkpoint_dir, "checkpoint")
#         torch.save(best_trained_model.state_dict(), path)
    
    print("Best trial test set accuracy: {}".format(test_acc))
    
    return best_trained_model.state_dict()


# In[ ]:


# perform layer count trials
def search_layers(num_samples=10, max_num_epochs=10, gpus_per_trial=0):
    data_dir=d.absolute()
    load_data(data_dir)
    layer_config_space = CS.ConfigurationSpace()

    layer_config_space.add_hyperparameter(
        CS.Constant("num_convs", value=2))
    layer_config_space.add_hyperparameter(
        CS.UniformIntegerHyperparameter("num_fcs", lower=2, upper=2**2))
    
    experiment_metrics = dict(metric="accuracy", mode="max")
    

    scheduler = HyperBandForBOHB(
        max_t=max_num_epochs,
        reduction_factor=2,
        **experiment_metrics)
    search = TuneBOHB(
        layer_config_space,
        max_concurrent=4,
        **experiment_metrics)
    reporter = CLIReporter(
#         overwrite=True,
        parameter_columns=layer_config_space.get_hyperparameter_names(),
        metric_columns=["loss", "accuracy", "training_iteration"])
    result = tune.run(
        partial(search_neurons),
        verbose=2,
        name="layers",
        local_dir=r.absolute(),
#         config=layer_config_space,
        resources_per_trial={"gpu": gpus_per_trial},
        max_failures=3,
        num_samples=num_samples,
        scheduler=scheduler,
        search_alg=search,
        progress_reporter=reporter)

    best_trial = result.get_best_trial("accuracy", "max", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))

    best_trained_model = Net([best_trial.config["num_convs"], best_trial.config["num_fcs"]])
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if gpus_per_trial > 1:
            best_trained_model = nn.DataParallel(best_trained_model)
    best_trained_model.to(device)

    best_checkpoint_dir = best_trial.checkpoint.value
    model_state, optimizer_state = torch.load(os.path.join(
        best_checkpoint_dir, "checkpoint"),map_location=torch.device('cpu'))
    best_trained_model.load_state_dict(model_state)

    test_acc = test_accuracy(best_trained_model, device)
    print("Best trial test set accuracy: {}".format(test_acc))
    
    return best_trained_model.state_dict()


# # perform test
# model = Net()
# if __name__ == "__main__":
#     # You can change the number of GPUs per trial here:
#     model = search_layers(num_samples=10, max_num_epochs=10, gpus_per_trial=1)
# 

# In[ ]:


layer_config_space = {}

# for hp in ["num_convs","num_fcs"]:
#     layer_config_space[hp] = np.random.randint(2,2**3)
# layer_config_space["num_convs"] = np.random.randint(2,3)
layer_config_space["num_convs"] = 2
layer_config_space["num_fcs"] = np.random.randint(3,2**2)

cpu_use = 1
gpu_use = 0.0
# data_dir = os.path.abspath("/home/grottesco/Source/RayTuneTut/data/")
# checkpoint_dir = os.path.abspath("/home/grottesco/Source/RayTuneTut/checkpoints")
print("Resource usage can be viewed at 127.0.0.1:8265")
start = time.time()
model = search_neurons(layer_config_space)
end = time.time()

print("\nProcessed in %s minutes\n" % ((end-start)/60,))


# print(model)

# !rm -rf ./data/* ./ray_results/layers/* ./ray_results/neurons/* 
