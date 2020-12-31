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


# In[2]:


# set data and checkpoint locations
p = Path('.')
d = p / 'data'
r = p / 'ray_results'
l = p / 'checkpoints' / 'layers'
n = p / 'checkpoints' / 'layers'

# set computation location(s)
gpus = torch.cuda.device_count()
device = "cuda:0" if gpus else "cpu"

# set number or fraction of processing units (per training worker) you'd like to utilize, if any at all
# cpu_use must be grater than zero
cpu_use = 1 if gpus else 0.5
gpu_use = 0.25 if gpus else 0

# set experiment hyperparameters
num_samples = 2 ** (5 if gpus else 4)
max_num_epochs = 10 * (4 if gpus else 1)
gpus_per_trial = 1 if gpus else 0


# Since the neuron configuration we want is dependent upon the number of layers we have, we need to work flatten the feature space a bit. We can reduce the high-dminesional setups to a slightly less high-dminesional string of base-n nodes.

# In[3]:


# define feature space for hashing
c_min = 3**2
c_max = 3**5
f_min = 2**2
f_max = 2**6

c = c_max - c_min
f = f_max - f_min

# conv = set(range(c_max)) - set(range(c_min))
# full = set(range(f_max)) - set(range(f_min))
conv = range(c_max)[c_min:]
full = range(f_max)[f_min:]

c_comb = list(combinations_with_replacement(conv,2))
f_comb = []
for layers in range(1,5):
    f_comb += list(combinations_with_replacement(full,layers))
#     print("Fully connected layer %s range: %s" % (layers,len(f_comb)) )
#     print("\n")

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
    unique_combinations = list(combinations([c_comb,f_comb],2))
    total_uniques = len(unique_combinations)
    total_points = total_uniques**2
    total_cvs = len(c_comb)
    total_fcs = len(f_comb)
    
    columns = ["base","nodes_req","sparcity","sparcity_pcnt","denoise_pcnt"]
    values = [1,total_uniques,total_points - total_uniques,(total_points - total_uniques) / total_points,0]
    
    cf = []
    
    for layer in [total_cvs,total_fcs]:#,total_uniques]:
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
    
        for base in range(2,17):
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
        report.sort_values(["sparcity","unexplained","nodes+_req","subsparcity","sparcity_pcnt","base"],inplace=True)
        
        cf.append(report.iloc[0])
    
    return cf

[print(r,"\n") for r in feature_spacing()]


# In[4]:


bases = feature_spacing()
print("For the convolutional layers, base %s seems to allow us to use the fewest nodes with the lowest number of invalid configuration indices (sparcity)." % bases[0]["base"])
print("For the linear layers, base %s seems to allow us to use the fewest nodes with the lowest number of invalid configuration indices (sparcity)." % bases[1]["base"])

# print("We can use the ")


# In[5]:


base_c = bases[0]["base"]
base_f = bases[1]["base"]

def base_to_dec(num_list, base):
    num_list = num_list[::-1]
    num = 0
    for k in range(len(num_list)):
        dig = num_list[k]
        dig = int(dig)
        num += dig*(base**k)
    return num

def encode(config=[(24, 64),(13, 41)]):
    iconv = c_comb.index(config[0])
    ifull = f_comb.index(config[1])
    
    conv_hash = numberToBase(iconv,base_c)
    full_hash = numberToBase(ifull,base_f)
    
    return [conv_hash,full_hash]

def decode(hash=([1, 7, 5, 0], [2, 9, 7])):
    conv = base_to_dec(hash[0], base_c)
    full = base_to_dec(hash[1], base_f)

    
    return [c_comb[conv],f_comb[full]]


# In[6]:


# move data into sets for loading
def load_data(data_dir=d.absolute()):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset,testset = [torchvision.datasets.CIFAR10(root=data_dir, train=is_train, download=True, transform=transform) for is_train in [True,False]]

    return trainset, testset


# In[7]:


# dynamically-generated nn that takes a 3-channel image and outputs a label
class Net(nn.Module):
    def __init__(self, hidden_layers=[[6, 16],[120,84]]):
        super(Net, self).__init__()
        hidden_convs,hidden_fcs = hidden_layers
#         print(hidden_convs)
#         print(hidden_fcs)
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
        
#         print("New model: %s" % hidden_layers)
    def forward(self, x):
        x = self.layers(x)
        return x


# In[8]:


# train nn on data
def train_cifar(neuron_config, checkpoint_dir=None):
    
    data_dir=d.absolute()
    
    def cv_discrim(s): return 'conv_subindex_' in s
    def fc_discrim(s): return 'full_subindex_' in s
    cvs = [neuron_config[hp] for hp in list(filter(cv_discrim, neuron_config.keys()))]
    fcs = [neuron_config[hp] for hp in list(filter(fc_discrim, neuron_config.keys()))]
    
    cfg = decode([cvs, fcs])    
    net = Net(cfg)
    
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
                print("Model: %s, Epoch: %d, Mini-batch: %5d, Loss: %.3f" % (cfg,epoch + 1, i + 1, running_loss / epoch_steps))
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


# In[9]:


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


# In[10]:


#determine configuration boundary for nn based on number of layers
nodes_c = bases[0]["nodes_req"]
nodes_f = bases[1]["nodes_req"]
max_c = bases[0]["max_necc_base_value"]
max_f = bases[1]["max_necc_base_value"]


# In[11]:


# def configure_neurons(num_convs,num_fcs):
def configure_neurons():
    config_space = CS.ConfigurationSpace()
    
    config_space.add_hyperparameter(
        CS.UniformFloatHyperparameter(name="lr", lower=1e-4, upper=1e-1, log=True))
    config_space.add_hyperparameter(
        CS.CategoricalHyperparameter(name="batch_size", choices=[4, 8, 16, 32]))
    config_space.add_hyperparameter(
        CS.CategoricalHyperparameter(name="epochs", choices=[20, 30, 40]))
    
    conv_lims,full_lims = [],[]
    
    for subindex in range(nodes_c):
        # define hyperparameter reference attributes
        rule_name = "conv_subindex_%s" % subindex
        conv_rule = CS.UniformIntegerHyperparameter(rule_name, lower=0, upper=base_c-1, default_value=subindex%(base_c-1))
        
        # add hyperparameter to collections
        config_space.add_hyperparameter(conv_rule)
    
        conv_rules = list(filter(lambda hp: "conv_subindex_" in hp.name, config_space.get_hyperparameters()))
    
        # build banlist from collections
        rl = deepcopy(config_space)
        rd = {}
        for ri,rule in enumerate(conv_rules,1):
    
            if (len(conv_rules) == 1) & (max_c[ri-1] == config_space.get_hyperparameter(rule_name).upper):
                break
            elif ri != len(conv_rules):
                rd[rule.name] = CS.ForbiddenEqualsClause(
                        rule,
                        max_c[ri-1]
                    )
            else:
                rd[rule.name] = CS.ForbiddenInClause(
                        rule,
                        range(
                            max_c[ri-1] + 1, 
                            rule.upper + 1
                        )
                    )
        
        # package banlist for addition to config space
        if rd.values():
            config_space.add_forbidden_clause(
                CS.ForbiddenAndConjunction(
                    *rd.values()
                )
            )           
    
    for subindex in range(nodes_f):
        # define hyperparameter reference attributes
        rule_name = "full_subindex_%s" % subindex
        full_rule = CS.UniformIntegerHyperparameter(rule_name, lower=0, upper=base_f-1, default_value=subindex%(base_f-1))
        
        # add hyperparameter to collections
        config_space.add_hyperparameter(full_rule)
    
        full_rules = list(filter(lambda hp: "full_subindex_" in hp.name, config_space.get_hyperparameters()))
    
        # build banlist from collections
        rl = deepcopy(config_space)
        rd = {}
        for ri,rule in enumerate(full_rules,1):
            if (len(full_rules) == 1) & (max_f[ri-1] == config_space.get_hyperparameter(rule_name).upper):
#                 print("breaking")
                break
            elif ri != len(full_rules):
                rd[rule.name] = CS.ForbiddenEqualsClause(
                        rule,
                        max_f[ri-1]
                    )
            else:
                rd[rule.name] = CS.ForbiddenInClause(
                        rule,
                        range(
                            max_f[ri-1] + 1, 
                            rule.upper + 1
                        )
                    )
        # add banlist to collection
        if rd.values():
            config_space.add_forbidden_clause(
                CS.ForbiddenAndConjunction(
                    *rd.values()
                )
            )           
        
    return config_space


# In[12]:


print(configure_neurons())


# In[13]:


# perform neuron configuration trials
def search_neurons(checkpoint_dir=None):    
    neuron_config_space = configure_neurons()
    
    experiment_metrics = dict(metric="accuracy", mode="max")
    
    #pre-load data to avoid races
    load_data()
    
    scheduler = HyperBandForBOHB(
        max_t=20,
        reduction_factor=2,
        **experiment_metrics)
    search = TuneBOHB(
        neuron_config_space,
        max_concurrent=8,
        **experiment_metrics)
    reporter = JupyterNotebookReporter(
        overwrite=True,
        parameter_columns=neuron_config_space.get_hyperparameter_names(),
        metric_columns=["loss", "accuracy", "training_iteration"])
    result = tune.run(
        partial(train_cifar),
        verbose=2,
        name="neurons",
        local_dir=r.absolute(),
        resources_per_trial={"cpu": cpu_use, "gpu": gpu_use},
        max_failures=3,
        num_samples=num_samples,
        scheduler=scheduler,
        search_alg=search,
        progress_reporter=reporter)

    best_trial = result.get_best_trial("accuracy", "max", "last")
    
    def cv_discrim(s): return 'conv_subindex_' in s
    def fc_discrim(s): return 'full_subindex_' in s
    def other_discrim(s): return 'subindex' not in s
    best_cvs = [best_trial.config[hp] for hp in list(filter(cv_discrim, best_trial.config.keys()))]
    best_fcs = [best_trial.config[hp] for hp in list(filter(fc_discrim, best_trial.config.keys()))]
    best_other = [best_trial.config[hp] for hp in list(filter(other_discrim, best_trial.config.keys()))]

    cfg = decode([best_cvs, best_fcs])
    
    conv_report = ["Connolutional Layer %s: %s" % (i,c) for i,c in enumerate(cfg[0])]
    full_report = ["Fully-connected Layer %s: %s" % (i,f) for i,f in enumerate(cfg[1])]
    other_report = ["%s: %s" % (hp,f) for (hp,f) in zip(["Batch Size","Epochs","Learning Rate"],best_other)]

#     print("Best trial config: {}".format(best_trial.config))
    print("Best trial config:")
    [print(best) for best in [conv_report,full_report,other_report]]
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))
    
    best_trained_model = Net(cfg)
    best_training_hyperparameters = zip(["Batch Size","Epochs","Learning Rate"],best_other)
    
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
    
    print("Best trial test set accuracy: {}".format(test_acc))
    
    return (best_trained_model.state_dict(), dict(best_training_hyperparameters))


# # perform test
# if __name__ == "__main__":
#     # You can change the number of GPUs per trial here:
#     model = search_layers(num_samples=10, max_num_epochs=10, gpus_per_trial=1)
# 

# In[14]:


print("Resource usage can be viewed at port http://127.0.0.1:8265/ or higher")


# In[15]:


model,trainers = search_neurons()

