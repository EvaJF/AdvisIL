import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim import lr_scheduler
import random

from ray import tune
from ray.tune.schedulers import ASHAScheduler

import os, sys
from configparser import ConfigParser
import pprint

from MyImageFolder import ImagesListFileFolder
from Utils import DataUtils
from model_utils import model_builder, model_namer, count_params


###############
### CONFIGS ###
###############

# loading configuration file
cp = ConfigParser()
cp.read(sys.argv[1])
cp = cp[os.path.basename(__file__)]

# reading parameters
backbone = cp['backbone'] 
width_mult = float(cp['width_mult']) 
depth_mult = float(cp['depth_mult']) 
num_samples = int(cp['num_samples'])  
normalization_dataset_name = cp['normalization_dataset_name']
datasets_mean_std_file_path = cp['datasets_mean_std_file_path']
train_file_path = cp['train_file_path']
test_file_path = cp['test_file_path'] 
num_classes = int(cp['num_classes'])  
test_batch_size = int(cp['test_batch_size']) 
num_workers = int(cp['num_workers']) 
epochs = int(cp['epochs'])  
save_dir = cp['save_dir']


###############
### DATASET ###
###############

# dataset : sanity check
print('Loading train images from '+train_file_path)
print('Loading val images from '+test_file_path)
print('Dataset name for normalization = '+normalization_dataset_name)
print('Using {} classes.'.format(num_classes))

# Data loading code
utils = DataUtils()
dataset_mean, dataset_std = utils.get_dataset_mean_std(normalization_dataset_name, datasets_mean_std_file_path)
normalize = transforms.Normalize(mean=dataset_mean, std=dataset_std)

train_dataset = ImagesListFileFolder(
    train_file_path,
    transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]),
    range_classes=range(num_classes))


test_dataset = ImagesListFileFolder(
    test_file_path, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]),
    range_classes=range(num_classes))


#############
### MODEL ###
#############

#Creating model
model_name = model_namer(backbone, width_mult, depth_mult)
print('\nCreating model %s ...' %model_name)
model = model_builder(num_classes, backbone=backbone, width_mult=width_mult, depth_mult=depth_mult)
print("Network architecture is of type %s with width multiplier %s and depth multiplier %s ." %(backbone, str(width_mult), str(depth_mult)))
print("Number of classes in the output layer : %s" %num_classes)
print("Number of parameters : {:,}".format(count_params(model)))

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # In this example, we don't change the model architecture
        # due to simplicity.
        self.conv1 = nn.Conv2d(1, 3, kernel_size=3)
        self.fc = nn.Linear(192, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 3))
        x = x.view(-1, 192)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


################################
### TRAINING & TESTING UTILS ###
################################

def train(model, optimizer, train_loader, criterion, scheduler):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    scheduler.step(loss.cpu().data.numpy())


def test(model, data_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    return correct / total

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    return 

def custom_train(config, model = model, train_dataset = train_dataset, test_dataset = test_dataset, epochs = epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(config["seed"])
    model.to(device)

    # Data Setup
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config["train_batch_size"], shuffle=True,
        num_workers=num_workers, pin_memory=False)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=test_batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=False)
    print("Number of batches in Training-set = " + str(len(train_loader)))
    print("Number of batches in Validation-set = " + str(len(test_loader)))

    # Define Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config["lr"], momentum=config["momentum"], 
                            weight_decay=config['weight_decay'])
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones = config["lr_strat"], gamma=config["lr_factor"]) 

    # Train & test
    for i in range(epochs):
        if i % 5 == 0:
            print("Epoch {} out of {}".format(i, epochs))
        train(model, optimizer, train_loader, criterion, scheduler)
        acc = test(model, test_loader)
        # Send the current training result back to Tune
        tune.report(mean_accuracy=acc)   

    # This saves the model to the trial directory
    #torch.save(model.state_dict(), "./model.pth")


####################
### SEARCH SPACE ###
####################

search_space = {
    "lr": tune.choice([0.005, 0.01, 0.05, 0.1]), #tune.sample_from(lambda spec: 10 ** (-10 * np.random.rand())),
    "momentum": tune.choice([0.75, 0.80, 0.85, 0.90, 0.95]), #tune.uniform(0.75, 0.95),
    "weight_decay": tune.choice([0.0001, 0.0005]),
    "lr_strat" : tune.choice([ [int(4/7*epochs//10*10), int(6/7*epochs//10*10)]]), # [int(3/7*epochs//10*10), int(5/7*epochs//10*10)],
    "lr_factor" : tune.choice([0.1]),
    "train_batch_size" : tune.choice([32]), # , 64, 128
    "seed": tune.randint(0, 10000)
}


##############
### TUNING ###
##############

analysis = tune.run(
    custom_train,
    num_samples=num_samples,
    scheduler=ASHAScheduler(metric="mean_accuracy", mode="max"),
    config=search_space,
    max_failures=5,
    resources_per_trial={'gpu': 1, 'cpu':4}
    
)

###############
### RESULTS ###
###############

# Obtain a trial dataframe from all run trials of this `tune.run` call.
dfs = analysis.trial_dataframes
df = analysis.results_df

# TODO print best config and save csv with detailed logs
print("save_dir %s" %save_dir)
df.to_csv(os.path.join(save_dir, "tuning_{model_name}_{dataset}_n{num_classes}_samples{num_samples}.csv".format(
    model_name = model_name, dataset=normalization_dataset_name, num_classes = num_classes, num_samples=num_samples)))

best_config = analysis.get_best_config("mean_accuracy", mode="max")
print("\nBest performance is obtained with config : ")
pp = pprint.PrettyPrinter()
pp.pprint(best_config)

# NB : To instantiate the model with the best config
#logdir = analysis.get_best_logdir("mean_accuracy", mode="max")
#state_dict = torch.load(os.path.join(logdir, "model.pth"))
#model = model_builder(backbone, width_mult, depth_mult)
#model.load_state_dict(state_dict)