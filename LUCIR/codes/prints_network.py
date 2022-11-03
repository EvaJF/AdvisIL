### imports ###
from modified_resnet import ResNet as my_resnet
from modified_resnet import BasicBlock as my_bb
from modified_resnet_original import ResNet as ref_resnet
from modified_resnet_original import BasicBlock as ref_bb
from utils_architecture import count_params

### original modified_resnet18 ###
ref = ref_resnet(block=ref_bb, layers=[2,2,2,2])
print("\nREFERENCE")
print(ref)
### my resnet ###
print("My RESNET")
my_resnet = my_resnet(block = my_bb, layers = [2,2,2,2])
print(my_resnet)

### compare number of parameters ###
print(count_params(ref))
print(count_params(my_resnet))

### compare matrices ###
ref_list = [p.numel() for p in ref.parameters()]
my_list = [p.numel() for p in my_resnet.parameters()]
print(len(ref_list))
print(len(my_list))

ref_list = [p.shape for p in ref.parameters()]
my_list = [p.shape for p in my_resnet.parameters()]

for i in range(len([p.numel() for p in ref.parameters()])):
    if ref_list[i] != my_list[i] : 
        print(i, ref_list[i], " (ref) vs (my)", my_list[i])
    else : 
        print("Same size")
