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


# In[ ]:


# set data and checkpoint locations
p = Path('.')
d = p / 'data'
r = p / 'ray_results'

## set number (or fraction) of GPUs (per training loop) you'd like to utilize if any at all
gpu_use = 1


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


# determine configuration boundary for nn based on number of layers
def configure_neurons(num_convs,num_fcs):
    config_space = CS.ConfigurationSpace()
    
    config_space.add_hyperparameter(
        CS.UniformFloatHyperparameter(name="lr", lower=1e-4, upper=1e-1, log=True))
    config_space.add_hyperparameter(
        CS.CategoricalHyperparameter(name="batch_size", choices=[2, 4, 8, 16]))
    config_space.add_hyperparameter(
        CS.CategoricalHyperparameter(name="epochs", choices=[10, 20, 30]))
    
    for hidden in range(num_convs):
        config_space.add_hyperparameter(
            CS.UniformIntegerHyperparameter("cv%s" % hidden, lower=3, upper=3**4))
    
    for hidden in range(num_fcs):
        config_space.add_hyperparameter(
            CS.UniformIntegerHyperparameter("fc%s" % hidden, lower=2**2, upper=2**5))
        
    return config_space


# In[ ]:


# perform neuron configuration trials
def search_neurons(layer_config, checkpoint_dir=None):
    num_samples=10
    max_num_epochs=10
    gpus_per_trial=1
    

    neuron_config_space = configure_neurons(layer_config["num_convs"], layer_config["num_fcs"])
    
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
    reporter = CLIReporter(
#         overwrite=True,
#         parameter_columns=["l1", "l2", "lr", "batch_size", "epochs"],
        parameter_columns=neuron_config_space.get_hyperparameter_names(),
        metric_columns=["loss", "accuracy", "training_iteration"])
    result = tune.run(
        partial(train_cifar),
        verbose=2,
        name="neurons",
        local_dir=r.absolute(),
        resources_per_trial={"cpu": 0.5, "gpu": gpu_use},
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
#     best_trained_model = Net(best_trial.config["l1"], best_trial.config["l2"])
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
    
    tune.report(accuracy=test_acc)
    
#     with tune.checkpoint_dir("nodes") as checkpoint_dir:
#         path = os.path.join(checkpoint_dir, "checkpoint")
#         torch.save(best_trained_model.state_dict(), path)
    
    print("Best trial test set accuracy: {}".format(test_acc))


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
        resources_per_trial={"cpu": 0.5, "gpu": gpus_per_trial},
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
    
    return best_trained_model


# In[ ]:


# perform test
model = Net()
if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    model = search_layers(num_samples=10, max_num_epochs=10, gpus_per_trial=0)


# layer_config_space = {}
# 
# # for hp in ["num_convs","num_fcs"]:
# #     layer_config_space[hp] = np.random.randint(2,2**3)
# layer_config_space["num_convs"] = np.random.randint(2,3)
# layer_config_space["num_fcs"] = np.random.randint(2,2**3)
# 
# # data_dir = os.path.abspath("/home/grottesco/Source/RayTuneTut/data/")
# # checkpoint_dir = os.path.abspath("/home/grottesco/Source/RayTuneTut/checkpoints")
#     
# search_neurons(layer_config_space)

# In[ ]:


print(model)

