import os
import numpy as np 

### NOTE ###
# The goal of this script is to build several random datasets out of Imagenet21k 
# constraints : a minimum number of images per class
# a class cannot be present in several datasets

MAP_PATH = "/home/eva/Documents/code/imagenet/synsets_words_size_map.txt"
LEAVES_PATH = "/home/eva/Documents/code/imagenet/imagenet_leaves.lst"
nb_datasets = 3 # number of datasets to create
nb_min_samples = 600 # minimum number of samples for a particular class
nb_classes = 100 # number of classes per dataset
output_dir = "/home/eva/Documents/code/imagenet/class_lists"

def get_classes(nb_min_sample = nb_min_samples, nb_classes=nb_classes, nb_datasets=nb_datasets, output_dir=output_dir):
    # select the classes with enough samples (only leaf of ImageNet21k) 
    list_of_classes = []
    f = open(MAP_PATH, "r")
    data = f.read()
    f.close()
    f = open(LEAVES_PATH, "r")
    leaf_list = f.read()
    f.close()
    leaf_list = leaf_list.split("\n")
    data = data.split("\n")
    for i in range(len(data)):
        data[i] = data[i].split("\t")
    for i in range(len(data)):
        if(len(data[i])==3 and int(data[i][2]) >= nb_min_sample and data[i][0] in leaf_list):
            list_of_classes.append(data[i][0])
    # randomly select classes
    list_of_classes = np.random.choice(list_of_classes, nb_classes*nb_datasets, replace=False)
    # write output
    for i in range(nb_datasets):
        classes_i = list_of_classes[i*nb_classes:(i+1)*nb_classes]
        print(classes_i)
        file_i = os.path.join(output_dir, "random_classes_{}.txt".format(i))
        f = open(file_i, 'w')
        for c in classes_i:
            f.write(c+'\n')
        f.close()
        print("Wrote classes into ", file_i)
    return list_of_classes

get_classes()
