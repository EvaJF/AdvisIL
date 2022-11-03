#####################
### PRELIMINARIES ###
#####################

# custom models 
from itertools import count
from modified_mobilenet import MobileNetV2
from modified_shufflenet import ShuffleNetV2
from modified_resnet import ResNet #, BasicBlock
import sys

from myMobileNet import myMobileNetV2
from myShuffleNet import myShuffleNetV2
from myResNet import myResNet, BasicBlock

# regular imports
from torchvision import models
import torch
print('Successful imports')
print("torch version : ", torch.__version__)

# utility function to get the number of parameters

def count_params(model, trainable=True):
    """
    Computes the number of (trainable) parameters of a model.

    Args
        model : PyTorch
    """
    if trainable : 
        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    else : 
        pytorch_total_params = sum(p.numel() for p in model.parameters())
    return pytorch_total_params

########################
### REFERENCE MODELS ###
########################

mobilenet_base = models.mobilenet_v2()
resnet_base = models.resnet18()
shufflenet_base = models.shufflenet_v2_x1_0()
for net in [mobilenet_base, resnet_base, shufflenet_base]:
    print("\n\n>>> Reference network", net)
    print("Trainable parameters : ", sum(p.numel() for p in net.parameters() if p.requires_grad))
    print("All parameters       : ", sum(p.numel() for p in net.parameters() ))

N=1000
print("\n\n ***** Sanity check on network sizes for N={} classes. REF *****".format(N))
print("Reference implementation")
print(count_params(mobilenet_base))
print(count_params(resnet_base))
print(count_params(shufflenet_base))
print("Modified linear")
print(count_params(MobileNetV2(width_mult=1.0, depth_mult=1.0, num_classes=N)))
print(count_params(ResNet(block=BasicBlock, layers=[2, 2, 2, 2], width_mult=1.0, depth_mult=1.0, num_classes=N)))
print(count_params(ShuffleNetV2(width_mult=1.0, depth_mult=1.0, num_classes=N)))
print("My networks")
print(count_params(myMobileNetV2(width_mult=1.0, depth_mult=1.0, num_classes=N)))
print(count_params(myResNet(block=BasicBlock, layers=[2, 2, 2, 2], width_mult=1.0, depth_mult=1.0, num_classes=N)))
print(count_params(myShuffleNetV2(width_mult=1.0, depth_mult=1.0, num_classes=N)))

# Comparative check
print("\n Resnet 1.0 1.0")
modified_net = ResNet(block=BasicBlock, layers=[2, 2, 2, 2], width_mult=1.0, depth_mult=1.0, num_classes=N)
my_net = myResNet(block=BasicBlock, layers=[2, 2, 2, 2], width_mult=1.0, depth_mult=1.0, num_classes=N)
modified_params = [(p.shape, p.numel()) for p in modified_net.parameters()]
my_params = [(p.shape, p.numel()) for p in my_net.parameters()]
print(len(modified_params)==len(my_params))
print(len(modified_params), "VS", len(my_params))
for i in range(len(my_params)):
    if my_params[i]!=modified_params[i]:
        print("Difference on element {i} / {L}".format(i=i+1, L = len(my_params)))
        print("My params : ", my_params[i])
        print("Modified : ", modified_params[i])

print("\n Resnet 1.0 0.5")
modified_net = ResNet(block=BasicBlock, layers=[2, 2, 2, 2], width_mult=1.0, depth_mult=0.5, num_classes=N)
my_net = myResNet(block=BasicBlock, layers=[2, 2, 2, 2], width_mult=1.0, depth_mult=0.5, num_classes=N)
modified_params = [(p.shape, p.numel()) for p in modified_net.parameters()]
my_params = [(p.shape, p.numel()) for p in my_net.parameters()]
print(len(modified_params)==len(my_params))
print(len(modified_params), "VS", len(my_params))
for i in range(len(my_params)):
    if my_params[i]!=modified_params[i]:
        print("Difference on element {i} / {L}".format(i=i+1, L = len(my_params)))
        print("My params : ", my_params[i])
        print("Modified : ", modified_params[i])
print(">>>>>>>>>>>>> DEBUG RESNET <<<<<<<<<<<<<<<<<")
print("Modified Net", modified_net)
print("My net : ", my_net)

print("\n Resnet 0.5 1.0")
modified_net = ResNet(block=BasicBlock, layers=[2, 2, 2, 2], width_mult=0.5, depth_mult=1.0, num_classes=N)
my_net = myResNet(block=BasicBlock, layers=[2, 2, 2, 2], width_mult=0.5, depth_mult=1.0, num_classes=N)
modified_params = [(p.shape, p.numel()) for p in modified_net.parameters()]
my_params = [(p.shape, p.numel()) for p in my_net.parameters()]
print(len(modified_params)==len(my_params))
print(len(modified_params), "VS", len(my_params))
for i in range(len(my_params)):
    if my_params[i]!=modified_params[i]:
        print("Difference on element {i} / {L}".format(i=i+1, L = len(my_params)))
        print("My params : ", my_params[i])
        print("Modified : ", modified_params[i])

print("\n Mobilenet 1.0 1.0")
modified_net = MobileNetV2(width_mult=1.0, depth_mult=1.0, num_classes=N)
my_net = myMobileNetV2(width_mult=1.0, depth_mult=1.0, num_classes=N)
modified_params = [(p.shape, p.numel()) for p in modified_net.parameters()]
my_params = [(p.shape, p.numel()) for p in my_net.parameters()]
print(len(modified_params)==len(my_params))
print(len(modified_params), "VS", len(my_params))
for i in range(len(my_params)):
    if my_params[i]!=modified_params[i]:
        print("Difference on element {i} / {L}".format(i=i+1, L = len(my_params)))
        print("My params : ", my_params[i])
        print("Modified : ", modified_params[i])

print("\n Mobilenet 1.0 0.2")
modified_net = MobileNetV2(width_mult=1.0, depth_mult=0.2, num_classes=N)
my_net = myMobileNetV2(width_mult=1.0, depth_mult=0.2, num_classes=N)
modified_params = [(p.shape, p.numel()) for p in modified_net.parameters()]
my_params = [(p.shape, p.numel()) for p in my_net.parameters()]
print(len(modified_params)==len(my_params))
print(len(modified_params), "VS", len(my_params))
for i in range(len(my_params)):
    if my_params[i]!=modified_params[i]:
        print("Difference on element {i} / {L}".format(i=i+1, L = len(my_params)))
        print("My params : ", my_params[i])
        print("Modified : ", modified_params[i])

print("\n Mobilenet 0.2 1.0")
modified_net = MobileNetV2(width_mult=0.2, depth_mult=1.0, num_classes=N)
my_net = myMobileNetV2(width_mult=0.2, depth_mult=1.0, num_classes=N)
modified_params = [(p.shape, p.numel()) for p in modified_net.parameters()]
my_params = [(p.shape, p.numel()) for p in my_net.parameters()]
print(len(modified_params)==len(my_params))
print(len(modified_params), "VS", len(my_params))
for i in range(len(my_params)):
    if my_params[i]!=modified_params[i]:
        print("Difference on element {i} / {L}".format(i=i+1, L = len(my_params)))
        print("My params : ", my_params[i])
        print("Modified : ", modified_params[i])


print("\n Shufflenet 1.0 1.0")
modified_net = ShuffleNetV2(width_mult=1.0, depth_mult=1.0, num_classes=N)
my_net = myShuffleNetV2(width_mult=1.0, depth_mult=1.0, num_classes=N)
modified_params = [(p.shape, p.numel()) for p in modified_net.parameters()]
my_params = [(p.shape, p.numel()) for p in my_net.parameters()]
print(len(modified_params)==len(my_params))
print(len(modified_params), "VS", len(my_params))
for i in range(len(my_params)):
    if my_params[i]!=modified_params[i]:
        print("Difference on element {i} / {L}".format(i=i+1, L = len(my_params)))
        print("My params : ", my_params[i])
        print("Modified : ", modified_params[i])

print("\n Shufflenet 1.0 0.1")
modified_net = ShuffleNetV2(width_mult=1.0, depth_mult=0.1, num_classes=N)
my_net = myShuffleNetV2(width_mult=1.0, depth_mult=0.1, num_classes=N)
modified_params = [(p.shape, p.numel()) for p in modified_net.parameters()]
my_params = [(p.shape, p.numel()) for p in my_net.parameters()]
print(len(modified_params)==len(my_params))
print(len(modified_params), "VS", len(my_params))
for i in range(len(my_params)):
    if my_params[i]!=modified_params[i]:
        print("Difference on element {i} / {L}".format(i=i+1, L = len(my_params)))
        print("My params : ", my_params[i])
        print("Modified : ", modified_params[i])

print("\n Shufflenet 0.1 1.0")
modified_net = ShuffleNetV2(width_mult=0.1, depth_mult=1.0, num_classes=N)
my_net = myShuffleNetV2(width_mult=0.1, depth_mult=1.0, num_classes=N)
modified_params = [(p.shape, p.numel()) for p in modified_net.parameters()]
my_params = [(p.shape, p.numel()) for p in my_net.parameters()]
print(len(modified_params)==len(my_params))
print(len(modified_params), "VS", len(my_params))
for i in range(len(my_params)):
    if my_params[i]!=modified_params[i]:
        print("Difference on element {i} / {L}".format(i=i+1, L = len(my_params)))
        print("My params : ", my_params[i])
        print("Modified : ", modified_params[i])


# test 1
print("\n\n ***** Sanity check on network sizes for various number of classes. CUSTOM *****".format(N))
net1 = MobileNetV2(width_mult=1.0, depth_mult=1.0, num_classes=10)
net2 = MobileNetV2(width_mult=1.0, depth_mult=1.0, num_classes=100)
net3 = MobileNetV2(width_mult=1.0, depth_mult=1.0, num_classes=1000)
print(count_params(net1), count_params(net2), count_params(net3))
assert(count_params(net1)!=count_params(net2))
assert(count_params(net2)!=count_params(net3))

# tests on CosineLinear
resnet_test1 = ResNet(block=BasicBlock, layers=[2,2,2,2], width_mult=0.5, depth_mult=0.5, num_classes=100)
print(resnet_test1)
my_list1 = [(p.shape, p.numel()) for p in resnet_test1.parameters()]
for i in range(len(my_list1)):
    print(my_list1[i])

resnet_test2 = ResNet(block=BasicBlock, layers=[2,2,2,2], width_mult=0.5, depth_mult=0.5, num_classes=10)
print(resnet_test2)
my_list2 = [(p.shape, p.numel()) for p in resnet_test2.parameters()]
for i in range(len(my_list2)):
    if my_list1[i] != my_list2[i] : 
        print(i, '/', len(my_list2), my_list1[i], my_list2[i])


#####################
### CUSTOM MODELS ###
#####################

###########################################
### Explore architectures with N = 100 ###

mem_to_config = {"mobilenet":{}, "resnet18":{}, "shufflenet":{}}
N = 100
print("\n\n ***** Assessing possibilities for N={} classes. *****".format(N))

# Resnet only (bigger networks) - Resnet18 with BasicBlocs
width_mult = [0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 2.0]
depth_mult = [0.1, 0.2, 0.4, 0.5, 0.6, 1.0]
for w in width_mult : 
    for d in depth_mult : 
        print("\nResnet18 --- Width mult : %s --- Depth mult : %s" %(w,d))
        mynet = ResNet(block=BasicBlock, layers=[2, 2, 2, 2], width_mult=w, depth_mult=d, num_classes=N)
        print(mynet.config)
        n_params = count_params(mynet)
        print(n_params)
        mem_to_config["resnet18"][n_params]=(w,d)


# Shufflenet and Mobilenet
width_mult = [0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.5, 2.6, 3.0]
depth_mult = [0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0, 2.0]
for w in width_mult : 
    for d in depth_mult :         
        # ShuffleNet
        print("\nShufflenet --- Width mult : %s --- Depth mult : %s" %(w,d))
        mynet = ShuffleNetV2(width_mult=w, depth_mult=d, num_classes=N)
        print(mynet.config)
        n_params = count_params(mynet)
        print(n_params)
        mem_to_config["shufflenet"][n_params]=(w,d)

        ### Mobilenet in (t, c, n, s) style
        print("\nMobilenet --- Width mult : %s --- Depth mult : %s" %(w,d))
        mynet = MobileNetV2(width_mult=w, depth_mult=d, num_classes=N)
        print(mynet.config)
        n_params = count_params(mynet)
        print(n_params)
        mem_to_config["mobilenet"][n_params]=(w,d)

### Memory budgets ###

budget = [0, 1.5e6, 3.0e6, 6e6] # millions of parameters
eligible_configs = {b:{"resnet18":[], "shufflenet":[], "mobilenet":[]} for b in budget}

for net in ["resnet18", "shufflenet", "mobilenet"]:
    for i in range(len(budget)-1): 
        b=budget[i+1]
        print("\nMemory budget : %s" %b)    
        for n_params in mem_to_config[net]:            
            if n_params < b and n_params > budget[i]: 
                eligible_configs[b][net].append(mem_to_config[net][n_params])
                print("Model {} is eligible with scaling {} width and {} depth, accounting for {:,} params.".format(net.upper(), 
                mem_to_config[net][n_params][0], mem_to_config[net][n_params][1], n_params))

###########################################
### Explore architectures with N = 1000 ###

mem_to_config = {"mobilenet":{}, "resnet18":{}, "shufflenet":{}}
N = 1000
print("\n\n ***** Assessing possibilities for N={} classes. *****".format(N))

# Resnet only (bigger networks) - Resnet18 with BasicBlocs
width_mult = [0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 2.0]
depth_mult = [0.1, 0.2, 0.4, 0.5, 0.6, 1.0]
for w in width_mult : 
    for d in depth_mult : 
        print("\nResnet18 --- Width mult : %s --- Depth mult : %s" %(w,d))
        mynet = ResNet(block=BasicBlock, layers=[2, 2, 2, 2], width_mult=w, depth_mult=d, num_classes=N)
        print(mynet.config)
        n_params = count_params(mynet)
        print(n_params)
        mem_to_config["resnet18"][n_params]=(w,d)


# Shufflenet and Mobilenet
width_mult = [0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.5, 2.6, 3.0]
depth_mult = [0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0, 2.0]
for w in width_mult : 
    for d in depth_mult :         
        # ShuffleNet
        print("\nShufflenet --- Width mult : %s --- Depth mult : %s" %(w,d))
        mynet = ShuffleNetV2(width_mult=w, depth_mult=d, num_classes=N)
        print(mynet.config)
        n_params = count_params(mynet)
        print(n_params)
        mem_to_config["shufflenet"][n_params]=(w,d)

        ### Mobilenet in (t, c, n, s) style
        print("\nMobilenet --- Width mult : %s --- Depth mult : %s" %(w,d))
        mynet = MobileNetV2(width_mult=w, depth_mult=d, num_classes=N)
        print(mynet.config)
        n_params = count_params(mynet)
        print(n_params)
        mem_to_config["mobilenet"][n_params]=(w,d)

### Memory budgets ###

budget = [0, 1.5e6, 3.0e6, 6e6] # millions of parameters
eligible_configs = {b:{"resnet18":[], "shufflenet":[], "mobilenet":[]} for b in budget}

for net in ["resnet18", "shufflenet", "mobilenet"]:
    for i in range(len(budget)-1): 
        b=budget[i+1]
        print("\nMemory budget : %s" %b)    
        for n_params in mem_to_config[net]:            
            if n_params < b and n_params > budget[i]: 
                eligible_configs[b][net].append(mem_to_config[net][n_params])
                print("Model {} is eligible with scaling {} width and {} depth, accounting for {:,} params.".format(net.upper(), 
                mem_to_config[net][n_params][0], mem_to_config[net][n_params][1], n_params))











