import os, sys
import random
import os
import numpy as np
import tarfile
from configparser import ConfigParser
from PIL import Image

### Preliminaries ###
# loading configuration file
cp = ConfigParser()
cp.read(sys.argv[1])
cp = cp[os.path.basename(__file__)]
input_path = cp["input_path"] # path to text file with class names and image numbers
path_to_tars = cp["path_to_tars"] # path to Imagenet tars
images_list_files_path = cp["images_list_files_path"] # path to folder for writing the output image list file for training and test sets
root_image_dir = cp["root_image_dir"] # prefix for image_list_files lists, folder where to store untarred images
num_classes = int(cp["num_classes"]) # number of classes to select for the subset of ImageNet
num_train = int(cp["num_train"]) # number of train images
num_test = int(cp["num_test"]) # number of test images
# set random seed for reproductibility
seed = int(cp["seed"])
np.random.seed(seed)
# check / create folder for the new dataset
# if not os.path.isdir(root_image_dir):
os.makedirs(root_image_dir, exist_ok=True) # override by default
print("Created a folder for the new dataset : ", root_image_dir)

if not os.path.isdir(images_list_files_path):
    os.makedirs(images_list_files_path)
    print("Created a folder for the new dataset : ", images_list_files_path)

### Get the classes ###
# read text file
class_list_file = open(input_path)
class_list=[] # dict key = imagenet leaf id, value = number of images for this class
for line in class_list_file:
    class_id = line[:-1]
    class_list.append(class_id)
print("There are a total of {} available classes.".format(len(class_list)))
print(class_list)

# select a random num_classes classes 
class_subset = np.random.choice((class_list), num_classes, replace=False)
#print(class_subset)

### Get the images ###
# create image_list_files for the training and the test sets
class_counter = 0
train_list_file_path = os.path.join(images_list_files_path, "train.lst")
test_list_file_path = os.path.join(images_list_files_path, "test.lst")
bad_images = [] # a few images among the many of ImageNet are correupted, in fact zipped twice. Discard thoose.

with open(train_list_file_path, 'w') as train, open(test_list_file_path, 'w') as test:
    train.write(root_image_dir+' -1\n')
    test.write(root_image_dir+' -1\n')
    for class_id in class_subset : 
        print("\nClass {class_id} has label {class_counter}.".format(class_id=class_id, class_counter=class_counter))
        # path to tarfile
        tar_path = os.path.join(path_to_tars, class_id+'.tar') 
        my_tarfile = tarfile.open(tar_path)
        image_list = my_tarfile.getnames()
        # sanity checks
        #print(image_list[:5])
        print("Number of images ", len(image_list))
        # assert len(image_list) == class_dic[class_id] # right number of images : not always the case ! 
        assert len(image_list) > num_test+num_train # enough images
        # shuffle images 
        random.Random(seed).shuffle(image_list)
        # untar the corresponding images and create the subfolders of the new dataset
        # write path to images into text files
        train_counter = 0
        test_counter = 0
        image_index = 0
        while train_counter < num_train : 
            image = image_list[image_index]
            my_tarfile.extract(image, os.path.join(root_image_dir, class_id)) # extract specific image
            try:
                img = Image.open(os.path.join(root_image_dir, class_id, image))
                img.verify()
                train.write(os.path.join(class_id, image) + ' ' + str(class_counter)+'\n') # write relative path into text file
                train_counter +=1
            except Exception as e:
                bad_images.append(os.path.join(root_image_dir, class_id, image))
                print("Bad image encountered")
                flag=False                
            image_index +=1
        
        while test_counter < num_test : 
            image = image_list[image_index]
            my_tarfile.extract(image, os.path.join(root_image_dir, class_id))
            try:
                img = Image.open(os.path.join(root_image_dir, class_id, image))
                img.verify()
                test.write(os.path.join(class_id, image) + ' ' + str(class_counter)+'\n')
                test_counter += 1
            except Exception as e:
                bad_images.append(os.path.join(root_image_dir, class_id, image))  
                print("Bad image encountered")
                flag=False                
            image_index += 1

        class_counter+=1
        my_tarfile.close()

print("Number of bad images found : ", len(bad_images))
print("\nSuccessfully wrote image list files. Check the files!")

            

