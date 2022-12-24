from __future__ import division
import torchvision.transforms as transforms
import sys, os, warnings, time
import numpy as np

from MyImageFolder import ImagesListFileFolder
import torch 


if len(sys.argv) != 2:
    print('Arguments: images_list_file_path')
    sys.exit(-1)


train_file_path = sys.argv[1]
print('Train file path = '+train_file_path)

#catching warnings
with warnings.catch_warnings(record=True) as warn_list:

    train_dataset = ImagesListFileFolder(
        train_file_path,
        transforms.ToTensor())

    num_classes = len(train_dataset.classes)
    print("Number of classes = " + str(num_classes))
    print("Training-set size = " + str(len(train_dataset)))

    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=24)
    mean = torch.zeros(3).cuda(0)
    std = torch.zeros(3).cuda(0)
    print('==> Computing mean and std..')
    cpt = 0
    for inputs, targets in dataloader:
        cpt += 1
        if cpt % 1000 == 0 : 
            print(str(cpt) +  '/'+ str(len(train_dataset)))

        inputs = inputs.cuda(0)
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(train_dataset))
    std.div_(len(train_dataset))

    print('train_file_path : '+train_file_path)
    print('mean = '+str(mean))
    print('std = '+str(std))
