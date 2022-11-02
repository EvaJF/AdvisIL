#####################
### PRELIMINARIES ###
#####################

# custom models 
from itertools import count
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
assert(count_params(mobilenet_base)==count_params(myMobileNetV2(width_mult=1.0, depth_mult=1.0, num_classes=N)))
assert(count_params(resnet_base)==count_params(myResNet(block=BasicBlock, layers=[2, 2, 2, 2], width_mult=1.0, depth_mult=1.0, num_classes=N)))
assert(count_params(shufflenet_base)==count_params(myShuffleNetV2(width_mult=1.0, depth_mult=1.0, num_classes=N)))

# test 
print("\n\n ***** Sanity check on network sizes for various number of classes. CUSTOM *****".format(N))
net1 = myMobileNetV2(width_mult=1.0, depth_mult=1.0, num_classes=10)
net2 = myMobileNetV2(width_mult=1.0, depth_mult=1.0, num_classes=100)
net3 = myMobileNetV2(width_mult=1.0, depth_mult=1.0, num_classes=1000)
print(count_params(net1), count_params(net2), count_params(net3))
assert(count_params(net1)!=count_params(net2))
assert(count_params(net2)!=count_params(net3))

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
        mynet = myResNet(block=BasicBlock, layers=[2, 2, 2, 2], width_mult=w, depth_mult=d, num_classes=N)
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
        mynet = myShuffleNetV2(width_mult=w, depth_mult=d, num_classes=N)
        print(mynet.config)
        n_params = count_params(mynet)
        print(n_params)
        mem_to_config["shufflenet"][n_params]=(w,d)

        ### Mobilenet in (t, c, n, s) style
        print("\nMobilenet --- Width mult : %s --- Depth mult : %s" %(w,d))
        mynet = myMobileNetV2(width_mult=w, depth_mult=d, num_classes=N)
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
        mynet = myResNet(block=BasicBlock, layers=[2, 2, 2, 2], width_mult=w, depth_mult=d, num_classes=N)
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
        mynet = myShuffleNetV2(width_mult=w, depth_mult=d, num_classes=N)
        print(mynet.config)
        n_params = count_params(mynet)
        print(n_params)
        mem_to_config["shufflenet"][n_params]=(w,d)

        ### Mobilenet in (t, c, n, s) style
        print("\nMobilenet --- Width mult : %s --- Depth mult : %s" %(w,d))
        mynet = myMobileNetV2(width_mult=w, depth_mult=d, num_classes=N)
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











