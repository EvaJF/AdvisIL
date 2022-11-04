from __future__ import division
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import torch.cuda as tc
import torch.utils.data.distributed
from torch.optim import lr_scheduler
from configparser import ConfigParser
import sys, os, warnings, time
from datetime import timedelta
import AverageMeter as AverageMeter
import socket
from MyImageFolder import ImagesListFileFolder
import copy
import numpy as np

from Utils import DataUtils

from model_utils import model_builder, model_namer, count_params

print("\n\n >>>>>>>>>>>> NO MEM FT <<<<<<<<<<<<<<\n")


if len(sys.argv) != 2:  # We have to give 1 arg
    print('Arguments: general_config')
    sys.exit(-1)

if not os.path.exists(sys.argv[1]):
    print('No configuration file found in the specified path')
    sys.exit(-1)

# loading configuration file
cp = ConfigParser()
cp.read(sys.argv[1])
cp = cp[os.path.basename(__file__)]

# reading parameters
gpu = int(cp['gpu'])
patience = int(cp['patience'])
num_workers = int(cp['num_workers'])
dataset_files_dir = cp['dataset_files_dir']
first_model_load_path = cp['first_batch_model_load_path']
lr_decay = float(cp['lr_decay'])
lr = float(cp['lr'])
momentum = float(cp['momentum'])
weight_decay = float(cp['weight_decay'])
old_batch_size = int(cp['old_batch_size'])
new_batch_size = int(cp['new_batch_size'])
test_batch_size = int(cp['test_batch_size'])
iter_size = int(old_batch_size / new_batch_size)
starting_epoch = int(cp['starting_epoch'])
num_epochs = int(cp['num_epochs'])
normalization_dataset_name = cp['normalization_dataset_name']
first_batch_number = int(cp['first_batch_number'])
last_batch_number = int(cp['last_batch_number'])
B = int(cp['B'])
P = int(cp['P'])
datasets_mean_std_file_path = cp['datasets_mean_std_file_path']
num_batches = cp["num_batches"] # for subfolder naming
algo_name = cp['algo_name']  #nomemft
memory=0
# backbone architecture
backbone = cp['backbone']
width_mult = float(cp['width_mult'])
depth_mult = float(cp['depth_mult'])
model_name = model_namer(backbone=backbone, width_mult=width_mult, depth_mult=depth_mult)
ckp_prefix = normalization_dataset_name+'_s'+num_batches+'_k'+str(memory)+'_B'+str(B)+'_P'+str(P)
models_save_dir = os.path.join(cp['models_save_dir'], normalization_dataset_name, ckp_prefix, model_name, algo_name+"_models") # ex : /results/mininat/models
saving_intermediate_models = cp['saving_intermediate_models'] == 'True'
intermediate_models_save_dir = os.path.join(models_save_dir, "checkpoints")

# make sure folders are created, if not create them
print("models_save_dir %s" %models_save_dir)
print("intermediate_models_save_dir %s" %intermediate_models_save_dir)
if not os.path.exists(models_save_dir):
    os.makedirs(models_save_dir)
    print("Created folder %s" %models_save_dir)
if saving_intermediate_models:
    if not os.path.exists(intermediate_models_save_dir):
        os.makedirs(intermediate_models_save_dir)
        print("Created folder %s" %intermediate_models_save_dir)

# catching warnings
with warnings.catch_warnings(record=True) as warn_list:
    utils = DataUtils()

    # Data loading code
    dataset_mean, dataset_std = utils.get_dataset_mean_std(normalization_dataset_name, datasets_mean_std_file_path)

    print('normalization dataset name = ' + str(normalization_dataset_name))
    print('dataset mean = ' + str(dataset_mean))
    print('dataset std = ' + str(dataset_std))
    print('first batch number = ' + str(first_batch_number))
    print('last batch number = ' + str(last_batch_number))

    # Data loading code
    normalize = transforms.Normalize(mean=dataset_mean, std=dataset_std)

    #print parameters
    print("Number of workers = " + str(num_workers))
    print("Old Batch size = " + str(old_batch_size))
    print("New Batch size = " + str(new_batch_size))
    print("test Batch size = " + str(test_batch_size))
    print("Iter size = " + str(iter_size))
    print("Starting epoch = " + str(starting_epoch))
    print("Number of epochs = " + str(num_epochs))
    print("momentum = " + str(momentum))
    print("weight_decay = " + str(weight_decay))
    print("lr_decay = " + str(lr_decay))
    print("patience = " + str(patience))
    print("Running on " + str(socket.gethostname()) + " | gpu " + str(gpu))


    top_1_test_accuracies = []
    top_5_test_accuracies = []

    for b in range(first_batch_number, last_batch_number +1):
        print('*' * 110)
        print('*' * 51+'BATCH '+str(b)+' '+'*'*51)
        print('*' * 110)

        train_file_path = os.path.join(dataset_files_dir,normalization_dataset_name, 'train.lst')
        test_file_path = os.path.join(dataset_files_dir,normalization_dataset_name, 'test.lst')
        print('train data loaded from ' + train_file_path)
        print('test data loaded from ' + test_file_path)

        batch_lr = lr / b
        if b == 2:
            model_load_path = first_model_load_path
            
            new_train_dataset = ImagesListFileFolder(
                train_file_path,
                transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize, ]),
                range_classes=range(B, B+(b-1)*P) # (b-1)*P,b*P
            )
            print("Range of classes for new_train_dataset : ", B, B+(b-1)*P)
            old_test_dataset = ImagesListFileFolder(
                test_file_path,
                transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize, ]),
                range_classes=range(B)) #(b-1)*P)
            print("range of classes for old_test_dataset : ", 0, B)
            new_test_dataset = ImagesListFileFolder(
                test_file_path,
                transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize, ]),
                range_classes=range(B, B+(b-1)*P)) #(b-1)*P,b*P)
            print("range of classes for new_test_dataset : ", B, B+(b-1)*P)
            test_dataset = torch.utils.data.dataset.ConcatDataset((
                old_test_dataset, new_test_dataset))

            train_loader = torch.utils.data.DataLoader(
                new_train_dataset, batch_size=new_batch_size, shuffle=True,
                num_workers=num_workers, pin_memory=False)

            test_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=test_batch_size, shuffle=True,
                num_workers=num_workers, pin_memory=False)

            old_classes_number = B #(b - 1) * P
            new_classes_number = len(new_train_dataset.classes)

        else:
            model_load_path = os.path.join(models_save_dir,save_name) #os.path.join(models_save_dir, algo_name+'_b'+str(b-1)+'.pt')

            new_train_dataset = ImagesListFileFolder(
                train_file_path,
                transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize, ]),
                range_classes=range(B+(b-2)*P,B+(b-1)*P)
            )

            old_test_dataset = ImagesListFileFolder(
                test_file_path,
                transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize, ]),
                range_classes=range(B+(b-2)*P)) #(b-1)*P)

            new_test_dataset = ImagesListFileFolder(
                test_file_path,
                transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize, ]),
                range_classes=range(B+(b-2)*P,B+(b-1)*P)) # (b-1)*P,b*P)

            test_dataset = torch.utils.data.dataset.ConcatDataset((
                old_test_dataset, new_test_dataset))

            train_loader = torch.utils.data.DataLoader(
                new_train_dataset, batch_size=new_batch_size, shuffle=True,
                num_workers=num_workers, pin_memory=False)

            test_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=test_batch_size, shuffle=True,
                num_workers=num_workers, pin_memory=False)

            old_classes_number = B + (b - 2) * P # (b - 1) * P
            new_classes_number = len(new_train_dataset.classes)


        print("lr = " + str(batch_lr))
        print("Old classes number = " + str(old_classes_number))
        print("New classes number = " + str(new_classes_number))
        print("New Training-set size = " + str(len(new_train_dataset)))
        print("Test-set size = " + str(len(test_dataset)))
        print("Number of batches in Training-set = " + str(len(train_loader)))
        print("Number of batches in Test-set = " + str(len(test_loader)))

        model_ft = model_builder(num_classes=old_classes_number, backbone=backbone, width_mult=width_mult, depth_mult=depth_mult) # initialize architecture 
        # load weights from the training of the first batch/phase
        print('Loading saved model from ' + model_load_path)
        state = torch.load(model_load_path, map_location=lambda storage, loc: storage)
        model_ft.load_state_dict(state['state_dict'])
 
        #print("DEBUG state_dict")
        #for name, param in model_ft.state_dict().items():
        #    print(name, param.size())
            
        #print("DEBUG in %s out %s" %(model_ft.fc.in_features, model_ft.fc.out_features))
        model_ft.fc = nn.Linear(model_ft.fc.in_features, old_classes_number + new_classes_number) #model_ft.fc.in_features instead of hard coded 512
        #print("DEBUG in %s out %s" %(model_ft.fc.in_features, model_ft.fc.out_features))
        print("Network architecture is of type %s with width multiplier %s and depth multiplier %s ." %(backbone, str(width_mult), str(depth_mult)))
        print("Number of classes in the output layer : %s" %(old_classes_number + new_classes_number))
        print("Number of parameters : {:,}".format(count_params(model_ft)))

        if tc.is_available():
            model_ft = model_ft.cuda(gpu)
        else:
            print("GPU not available")
            sys.exit(-1)

        # Define Loss and Optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer_ft = optim.SGD(model_ft.parameters(), lr=batch_lr, momentum=momentum, weight_decay=weight_decay)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_ft, patience=patience, factor=lr_decay)


        # Training
        print("-" * 20)
        print("Training...")
        starting_time = time.time()
        best_top1_v_acc = -1
        best_top5_v_acc = -1
        best_epoch = 0
        best_model = None
        best_optimizer_ft = None
        epoch = 0
        for epoch in range(num_epochs):
            top1 = AverageMeter.AverageMeter()
            top5 = AverageMeter.AverageMeter()
            model_ft.train()
            running_loss = 0.0
            nb_batches = 0
            # zero the parameter gradients
            optimizer_ft.zero_grad()
            for i, data in enumerate(train_loader, 0):
                nb_batches += 1
                # get the data
                inputs, labels = data

                if tc.is_available():
                    inputs, labels = inputs.cuda(gpu), labels.cuda(gpu)

                # wrap it in Variable
                inputs, labels = Variable(inputs), Variable(labels)

                # forward + backward + optimize
                outputs = model_ft(inputs)
                loss = criterion(outputs, labels)

                # loss.data[0] /= iter_size
                # loss.backward()
                # running_loss += loss.data.cpu().numpy()[0]
                loss.data /= iter_size
                loss.backward()
                running_loss += loss.data.item()
                if (i+1)%iter_size == 0:
                    optimizer_ft.step()
                    optimizer_ft.zero_grad()

            scheduler.step(loss.cpu().data.numpy())

            # Model evaluation
            model_ft.eval()

            #Test on both old and new data
            for data in test_loader:
                inputs, labels = data
                if tc.is_available():
                    inputs, labels = inputs.cuda(gpu), labels.cuda(gpu)
                outputs = model_ft(Variable(inputs))
                prec1, prec5 = utils.accuracy(outputs.data, labels, topk=(1, min(5,  P * b)))
                top1.update(prec1.item(), inputs.size(0))
                top5.update(prec5.item(), inputs.size(0))
            # -------------------------------------------
            if top1.avg > best_top1_v_acc:
                best_top1_v_acc = top1.avg
                best_top5_v_acc = top5.avg
                best_model = copy.deepcopy(model_ft)
                best_optimizer_ft = copy.deepcopy(optimizer_ft)
                best_epoch = epoch


            current_elapsed_time = time.time() - starting_time
            print('{:03}/{:03} | {} | Train : loss = {:.4f}  | Test : acc@1 = {}% ; acc@5 = {}%'.
                  format(epoch + 1, num_epochs, timedelta(seconds=round(current_elapsed_time)),
                         running_loss / nb_batches, top1.avg , top5.avg))



            if saving_intermediate_models == True :
                # Saving model
                state = {
                    'epoch': best_epoch,
                    'state_dict': model_ft.state_dict(),
                    'optimizer': optimizer_ft.state_dict(),
                    'best_top1_v_acc': best_top1_v_acc
                }
                model_name = model_namer(backbone, width_mult, depth_mult)
                intermediate_save_name = '_'.join([algo_name, normalization_dataset_name, 's'+str(num_batches), model_name, "batch"+str(b), "epoch"+str(epoch)])+'.pt'
                torch.save(state, os.path.join(intermediate_models_save_dir,intermediate_save_name))
                



        #training finished
        if best_model is not None:
            model_name = model_namer(backbone, width_mult, depth_mult)
            save_name = '_'.join([algo_name, ckp_prefix, model_name, "batch"+str(b)])+'.pt'
            print('Saving best model in %s under name %s' %(models_save_dir, save_name))
            print("Path : ", os.path.join(models_save_dir,save_name))
            state = {
                'epoch': epoch,
                'state_dict': best_model.state_dict(),
                'optimizer': best_optimizer_ft.state_dict()
            }
            print('best acc = ' + str(best_top1_v_acc))
            torch.save(state, os.path.join(models_save_dir, save_name))

        top_1_test_accuracies.append(best_top1_v_acc)
        top_5_test_accuracies.append(best_top5_v_acc)
        print('top1 accuracies so far : ' + str(top_1_test_accuracies))
        print('top5 accuracies so far : ' + str(top_5_test_accuracies))

# Print warnings (Possibly corrupt EXIF files):
if len(warn_list) > 0:
    print("\n" + str(len(warn_list)) + " Warnings\n")
    # for i in range(len(warn_list)):
    #     print("warning " + str(i) + ":")
    #     print(str(i)+":"+ str(warn_list[i].category) + ":\n     " + str(warn_list[i].message))
else:
    print('No warnings.')

print('TOP1 Test accuracies = '+str([float(str(e)[:6]) for e in top_1_test_accuracies]))
print('TOP1 mean incremental accuracy = '+str(np.mean(np.array(top_1_test_accuracies))))
print('***************')
print('TOP5 Test accuracies = '+str([float(str(e)[:6]) for e in top_5_test_accuracies]))
print('TOP5 mean incremental accuracy = '+str(np.mean(np.array(top_5_test_accuracies))))