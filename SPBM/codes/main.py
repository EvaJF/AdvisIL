#!/usr/bin/env python
# coding=utf-8
import torch, math
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms
from torch.autograd import Variable
import numpy as np
import os
import sys

from MyImageFolder import ImagesListFileFolder
import copy
from PIL import Image
try:
    import cPickle as pickle
except:
    import pickle

from utils_architecture import model_namer, count_params, modified_model_builder
import modified_linear

from utils_dataset import split_images_labels, save_protosets,  merge_images_labels
from utils_incremental.compute_features import compute_features
from utils_incremental.compute_accuracy import compute_accuracy
from utils_incremental.train_eval_SPB_rot import train_eval_SPB_rot


######### Modifiable Settings ##########
from configparser import ConfigParser
from Utils import DataUtils
import warnings, socket, os

if len(sys.argv) != 2:  # We have to give 1 arg
    print('Arguments: config')
    sys.exit(-1)
with warnings.catch_warnings(record=True) as warn_list:

    ######### Modifiable Settings ##########
    utils = DataUtils()
    # loading configuration file
    cp = ConfigParser()
    cp.read(sys.argv[1])
    cp = cp[os.path.basename(__file__)]
    ######################################## ARCHITECTURE
    backbone = cp["backbone"]
    width_mult = float(cp["width_mult"])
    depth_mult = float(cp["depth_mult"])
    ######################################## FIRST STATE
    first_train_batch_size = int(cp['first_train_batch_size']) # Batch size for train
    first_base_lr = float(cp['first_base_lr'])  # Initial learning rate
    first_lr_strat = utils.from_str_to_list(cp['first_lr_strat'], 'int')  # Epochs where learning rate gets decreased
    first_lr_factor = float(cp['first_lr_factor']) # Learning rate decrease factor
    first_custom_weight_decay = float(cp['first_custom_weight_decay'])  # Weight Decay
    first_epochs = int(cp['first_epochs'])
    ######################################## SUBSEQUENT STATES
    subsequent_train_batch_size = int(cp['subsequent_train_batch_size']) # Batch size for train
    subsequent_base_lr_features_extractor = float(cp['subsequent_base_lr_features_extractor'])  # Initial learning rate
    subsequent_base_lr_classifier = float(cp['subsequent_base_lr_classifier'])  # Initial learning rate
    subsequent_lr_strat = utils.from_str_to_list(cp['subsequent_lr_strat'], 'int')  # Epochs where learning rate gets decreased
    subsequent_lr_factor = float(cp['subsequent_lr_factor']) # Learning rate decrease factor
    subsequent_custom_weight_decay = float(cp['subsequent_custom_weight_decay'])  # Weight Decay
    subsequent_epochs = int(cp['subsequent_epochs'])
    first_path_model = cp['first_path_model']    
    test_batch_size = int(cp['test_batch_size'])  # Batch size for test
    eval_batch_size = int(cp['eval_batch_size'])  # Batch size for eval
    custom_momentum = float(cp['custom_momentum'])  # Momentum
    normalization_dataset_name = cp['normalization_dataset_name']  # Momentum
    datasets_mean_std_file_path = cp['datasets_mean_std_file_path']  # Momentum
    train_file_path = cp['train_file_path']
    test_file_path = cp['test_file_path']
    num_classes = int(cp['num_classes'])
    num_workers = int(cp['num_workers'])
    last_batch_number = int(cp['last_batch_number'])
    B = int(cp['B'])
    P = int(cp['P'])
    memory_size = int(cp['memory_size'])
    nb_runs = int(cp['nb_runs'])
    output_dir = cp['output_dir']
    T = int(cp['T'])
    beta= float(cp['beta'])
    resume = cp['resume'] == 'True'
    rs_ratio = float(cp['rs_ratio'])
    random_seed= int(cp['random_seed'])
    K= int(cp['K'])
    adapt_lamda = cp['adapt_lamda'] == 'True'
    rot_em_loss = cp['rot_em_loss'] == 'True'
    lw_ms = float(cp['lw_ms'])
    lamda = float(cp['lamda'])
    dist = float(cp['dist'])
    lw_mr = float(cp['lw_mr'])
    exists_val = cp['exists_val'] == 'True'
    if exists_val: memory_size = int(math.ceil(memory_size * 0.9))
    ft_epochs = int(cp['ft_epochs'])
    ft_base_lr = float(cp['ft_base_lr'])
    ft_lr_strat = utils.from_str_to_list(cp['ft_lr_strat'], 'int')  # Epochs where learning rate gets decreased
    ft_flag = int(cp['ft_flag'])
    ################################################
    print(sys.argv[0])
    print("Running on " + str(socket.gethostname()) + " | " + str(os.environ["CUDA_VISIBLE_DEVICES"]) + '\n')
    #utils.print_parameters(cp)
    print(cp)
    ################################################
    assert(B >= P) 
    S  = int((num_classes - B) / P) + 1
    print("num_classes ", num_classes)
    print("last_batch_number ", last_batch_number) 
    print('S = '+str(S))
    ckp_prefix        = '{}_s{}_k{}_B{}_P{}'.format(normalization_dataset_name, S, memory_size, B, P)
    if random_seed!=-1:
        np.random.seed(random_seed)        # Fix the random seed
    ########################################
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset_mean, dataset_std = utils.get_dataset_mean_std(normalization_dataset_name, datasets_mean_std_file_path)
    normalize = transforms.Normalize(mean=dataset_mean, std=dataset_std)

    trainset = ImagesListFileFolder(
                train_file_path,
                transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]))

    testset = ImagesListFileFolder(
                test_file_path,
                transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize, ]))


    evalset = ImagesListFileFolder(
                test_file_path,
                transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize, ]))

    print('Training-set size = ' + str(len(trainset)))
    print('Testing-set size = '+str(len(evalset)))

    ################################
    print("memory_size = ", memory_size)
    X_train_total, Y_train_total = split_images_labels(trainset.samples)
    X_valid_total, Y_valid_total = split_images_labels(testset.samples)

    # Launch the different runs
    for iteration_total in range(nb_runs):

        top1_cnn_cumul_acc = []
        top5_cnn_cumul_acc = []

        #follow the same order than us
        order = np.arange(num_classes)
        order_list = list(order)

        # Initialization of the variables for this run
        X_valid_cumuls    = []
        X_protoset_cumuls = []
        X_train_cumuls    = []
        Y_valid_cumuls    = []
        Y_protoset_cumuls = []
        Y_train_cumuls    = []

        # The following contains all the training samples of the different classes
        # because we want to compare our method with the theoretical case where all the training samples are stored
        prototypes = [[] for i in range(num_classes)]
        for orde in range(num_classes):
            prototypes[orde] = X_train_total[np.where(Y_train_total==order[orde])]

        prototypes = np.array(prototypes)
        max_class_len = max(len(e) for e in prototypes)

        alpha_dr_herding = np.zeros((last_batch_number+ B//P-1,max_class_len,P),np.float32) # modify if use of memory


        first_batch_number = 0 #int(B/P) - 1
        
        print('b in range(first_batch_number, last_batch_number)', list(range(first_batch_number, last_batch_number)))
        for b in range(first_batch_number, last_batch_number):
            #init model
            print("\nModel init for batch ", b+1)
            if b == first_batch_number:
                ############################################################
                last_iter = 0
                ############################################################
                model_name = model_namer(backbone=backbone, width_mult=width_mult, depth_mult=depth_mult)
                print('\nCreating model %s ...' %model_name) #modified_resnet.resnet18(num_classes=B)
                print("Network architecture is of type %s with width multiplier %s and depth multiplier %s ." %(backbone, str(width_mult), str(depth_mult)))
                tg_model = modified_model_builder(num_classes=B, backbone=backbone, width_mult=width_mult, depth_mult=depth_mult)
                print("Number of parameters : {:,}".format(count_params(tg_model)))
                print(tg_model.config)
                print(tg_model)
                in_features = tg_model.fc.in_features
                out_features = tg_model.fc.out_features
                print("in_features:", in_features, "out_features:", out_features)
                ref_model = None
            elif b == first_batch_number+1:
                ############################################################
                last_iter = b
                ############################################################
                #increment classes
                ref_model = copy.deepcopy(tg_model)
                in_features = tg_model.fc.in_features
                out_features = tg_model.fc.out_features
                print("in_features:", in_features, "out_features:", out_features)
                new_fc = modified_linear.SplitCosineLinear(in_features, out_features, P)
                new_fc.fc1.weight.data = tg_model.fc.weight.data
                new_fc.sigma.data = tg_model.fc.sigma.data
                print("new_fc.fc1.weight.data.size", new_fc.fc1.weight.data.size())
                print("new_fc.fc2.weight.data.size", new_fc.fc2.weight.data.size())
                tg_model.fc = new_fc
                lamda_mult = out_features*1.0 / P
            else:
                ############################################################
                last_iter = b
                ############################################################
                ref_model = copy.deepcopy(tg_model)
                in_features = tg_model.fc.in_features
                out_features1 = tg_model.fc.fc1.out_features
                out_features2 = tg_model.fc.fc2.out_features
                print("in_features:", in_features, "out_features1:", out_features1, "out_features2:", out_features2)
                new_fc = modified_linear.SplitCosineLinear(in_features, out_features1+out_features2, P)
                #print("================== debug 1=====================")
                #print("new_fc.fc1.weight.data.size", new_fc.fc1.weight.data.size())
                #print("tg_model.fc.fc1.weight.data.size", tg_model.fc.fc1.weight.data.size())
                #print("tg_model.fc.fc2.weight.data.size", tg_model.fc.fc2.weight.data.size())
                new_fc.fc1.weight.data[:out_features1] = tg_model.fc.fc1.weight.data
                new_fc.fc1.weight.data[out_features1:] = tg_model.fc.fc2.weight.data
                new_fc.sigma.data = tg_model.fc.sigma.data
                tg_model.fc = new_fc
                lamda_mult = (out_features1+out_features2)*1.0 / (P)

            if b > first_batch_number and adapt_lamda:
                cur_lamda = lamda * math.sqrt(lamda_mult)
            else:
                cur_lamda = lamda
            if b > first_batch_number:
                print("###############################")
                print("Lamda for less forget is set to ", cur_lamda)
                print("###############################")

            # Prepare the training data for the current batch of classes
            if b==first_batch_number : 
                actual_cl        = order[range(B)]
                indices_train_10 = np.array([i in order[range(B)] for i in Y_train_total])
                indices_test_10  = np.array([i in order[range(B)] for i in Y_valid_total])               
            else : 
                actual_cl        = order[range(B+(b-1)*P,B+b*P)]
                indices_train_10 = np.array([i in order[range(B+(b-1)*P,B+b*P)] for i in Y_train_total])
                indices_test_10  = np.array([i in order[range(B+(b-1)*P,B+b*P)] for i in Y_valid_total])

            #actual_cl        = order[range(last_iter*P,(b+1)*P)]
            #indices_train_10 = np.array([i in order[range(last_iter*P,(b+1)*P)] for i in Y_train_total])
            #indices_test_10  = np.array([i in order[range(last_iter*P,(b+1)*P)] for i in Y_valid_total])

            X_train          = X_train_total[indices_train_10]
            X_valid          = X_valid_total[indices_test_10]
            X_valid_cumuls.append(X_valid)
            X_train_cumuls.append(X_train)
            X_valid_cumul    = np.concatenate(X_valid_cumuls)
            X_train_cumul    = np.concatenate(X_train_cumuls)

            Y_train          = Y_train_total[indices_train_10]
            Y_valid          = Y_valid_total[indices_test_10]
            Y_valid_cumuls.append(Y_valid)
            Y_train_cumuls.append(Y_train)
            Y_valid_cumul    = np.concatenate(Y_valid_cumuls)
            Y_train_cumul    = np.concatenate(Y_train_cumuls)

            # Add the stored exemplars to the training data
            #if b == first_batch_number:
            #    X_valid_ori = X_valid
            #    Y_valid_ori = Y_valid
            #else:
            #    X_protoset = np.concatenate(X_protoset_cumuls)
            #    Y_protoset = np.concatenate(Y_protoset_cumuls)
            #    if rs_ratio > 0:
            #        scale_factor = (len(X_train) * rs_ratio) / (len(X_protoset) * (1 - rs_ratio))
            #        rs_sample_weights = np.concatenate((np.ones(len(X_train)), np.ones(len(X_protoset))*scale_factor))
            #        #number of samples per epoch
            #        rs_num_samples = int(len(X_train) / (1 - rs_ratio))
            #        print("X_train:{}, X_protoset:{}, rs_num_samples:{}".format(len(X_train), len(X_protoset), rs_num_samples))
            #    X_train    = np.concatenate((X_train,X_protoset),axis=0)
            #    Y_train    = np.concatenate((Y_train,Y_protoset))
#
            # Launch the training loop
            print('Batch of classes number {0} arrives for training ...'.format(b+1))
            map_Y_train = np.array([order_list.index(i) for i in Y_train])
            map_Y_valid_cumul = np.array([order_list.index(i) for i in Y_valid_cumul])

            #imprint weights
            if b > first_batch_number: # P classes in the batch
                print("Imprint weights")
                #########################################
                #compute the average norm of old embdding
                old_embedding_norm = tg_model.fc.fc1.weight.data.norm(dim=1, keepdim=True)
                average_old_embedding_norm = torch.mean(old_embedding_norm, dim=0).to('cpu').type(torch.DoubleTensor)
                #########################################
                #tg_feature_model = nn.Sequential(*list(tg_model.children())[:-1])
                if backbone=="resnetBasic" or backbone=="resnetBottleneck":
                    tg_feature_model = nn.Sequential(*list(tg_model.children())[:-1])  
                elif backbone=="mobilenet" or backbone=="shufflenet":
                    tg_feature_model = nn.Sequential(*list(tg_model.children())[:-2],nn.AdaptiveAvgPool2d((1, 1)))
                else : 
                    print("wrong backbone")
                num_features = tg_model.fc.in_features
                novel_embedding = torch.zeros((P, num_features))
                for cls_idx in range(B+(b-1)*P, B+b*P) : # // originally range(b*P, (b+1)*P)
                    cls_indices = np.array([i == cls_idx  for i in map_Y_train])
                    assert(len(np.where(cls_indices==1)[0])<=max_class_len)
                    current_eval_set = merge_images_labels(X_train[cls_indices], np.zeros(len(X_train[cls_indices])))
                    evalset.imgs = evalset.samples = current_eval_set
                    evalloader = torch.utils.data.DataLoader(evalset, batch_size=eval_batch_size,
                        shuffle=False, num_workers=num_workers)
                    num_samples = len(X_train[cls_indices])
                    cls_features = compute_features(tg_feature_model, evalloader, num_samples, num_features)
                    norm_features = F.normalize(torch.from_numpy(cls_features), p=2, dim=1)
                    cls_embedding = torch.mean(norm_features, dim=0)
                    novel_embedding[cls_idx-B-(b-1)*P] = F.normalize(cls_embedding, p=2, dim=0) * average_old_embedding_norm
                tg_model.to(device)
                print("Before : ", tg_model.fc.fc2.weight.data.size())
                tg_model.fc.fc2.weight.data = novel_embedding.to(device)
                print("After : ", tg_model.fc.fc2.weight.data.size())
            ############################################################
            current_train_imgs = merge_images_labels(X_train, map_Y_train)
            trainset.imgs = trainset.samples = current_train_imgs
            if b==first_batch_number:
                train_batch_size = first_train_batch_size
                epochs = first_epochs
                base_lr = first_base_lr
                custom_weight_decay = first_custom_weight_decay
                lr_factor = first_lr_factor
                lr_strat = first_lr_strat
            else:
                train_batch_size = subsequent_train_batch_size
                epochs = subsequent_epochs
                base_lr = subsequent_base_lr_features_extractor
                custom_weight_decay = subsequent_custom_weight_decay
                lr_factor = subsequent_lr_factor
                lr_strat = subsequent_lr_strat
            #if b > first_batch_number and rs_ratio > 0 and scale_factor > 1:
            #    print("Weights from sampling:", rs_sample_weights)
            #    index1 = np.where(rs_sample_weights>1)[0]
            #    index2 = np.where(map_Y_train<B+(b-1)*P)[0] # // originally : index2 = np.where(map_Y_train<b*P)[0]
            #    assert((index1==index2).all())
            #    train_sampler = torch.utils.data.sampler.WeightedRandomSampler(rs_sample_weights, rs_num_samples)
            #    trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, \
            #        shuffle=False, sampler=train_sampler, num_workers=num_workers, pin_memory=True)
            #else:
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

            print('Training-set size = ' + str(len(trainset)))
            print('Testing-set size = '+str(len(evalset)))
            current_test_imgs = merge_images_labels(X_valid_cumul, map_Y_valid_cumul)
            testset.imgs = testset.samples = current_test_imgs
            testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size,
                shuffle=False, num_workers=num_workers)
            print('Max and Min of train labels: {}, {}'.format(min(map_Y_train), max(map_Y_train)))
            print('Max and Min of valid labels: {}, {}'.format(min(map_Y_valid_cumul), max(map_Y_valid_cumul)))
            ##############################################################
            ckp_dir = os.path.join(output_dir,
                                   normalization_dataset_name, 
                                   ckp_prefix, 
                                   model_name, 
                                   "SPB-M",
                                   "models")
            if not os.path.exists(ckp_dir):
                os.makedirs(ckp_dir)
                print("Created directory ", ckp_dir)
            ckp_name = os.path.join(ckp_dir, ckp_prefix+'_model_{}.pth'.format(b)) # TODO add number of epochs
            print('ckp_name', ckp_name)            
            if b > first_batch_number:
                first_path_model = ckp_name
            if resume and os.path.exists(ckp_name):
                print("###############################")
                print("Loading models from checkpoint")
                tg_model = torch.load(ckp_name)
                print("###############################")
            else:
                
                ###############################
                if b > first_batch_number:
                    #print("================== debug 2 =====================")
                    #print("tg_model.fc.fc1.weight.data.size", tg_model.fc.fc1.weight.data.size())
                    #print("tg_model.fc.fc2.weight.data.size", tg_model.fc.fc2.weight.data.size())
                
                    #fix the embedding of old classes
                    ignored_params = list(map(id, tg_model.fc.fc1.parameters()))+list(map(id, tg_model.fc.fc2.parameters()))
                    base_params = filter(lambda p: id(p) not in ignored_params, \
                        tg_model.parameters())
                    tg_params =[{'params': base_params, 'lr': base_lr, 'weight_decay': custom_weight_decay}, \
                        {'params': tg_model.fc.fc1.parameters(), 'lr': 0, 'weight_decay': 0}, \
                        {'params': tg_model.fc.fc2.parameters(), 'lr': subsequent_base_lr_classifier, 'weight_decay': subsequent_custom_weight_decay}]
                else:
                    tg_params = tg_model.parameters()
                ###############################
                tg_model = tg_model.to(device)
                if b > first_batch_number:
                    ref_model = ref_model.to(device)
                    #print("================== debug 3 =====================")
                    #print("tg_model.fc.fc1.weight.data.size", tg_model.fc.fc1.weight.data.size())
                    #print("tg_model.fc.fc2.weight.data.size", tg_model.fc.fc2.weight.data.size())
                
                tg_optimizer = optim.SGD(tg_params, lr=ft_base_lr, momentum=custom_momentum, weight_decay=custom_weight_decay)
                tg_lr_scheduler = lr_scheduler.MultiStepLR(tg_optimizer, milestones=lr_strat, gamma=lr_factor)

                ###############################

                print("train_eval_SPB_rot")
                print("first_path_model :", first_path_model)
                tg_model = train_eval_SPB_rot(epochs, first_path_model, tg_model, ref_model, tg_optimizer, tg_lr_scheduler, \
                    trainloader, testloader, \
                    b, first_batch_number, \
                    cur_lamda, \
                    dist, K, lw_mr, backbone)
                if b > first_batch_number:
                    #print("================== debug 4 =====================")
                    #print("tg_model.fc.fc1.weight.data.size", tg_model.fc.fc1.weight.data.size())
                    #print("tg_model.fc.fc2.weight.data.size", tg_model.fc.fc2.weight.data.size())
                    pass
                if not os.path.isdir(output_dir):
                    os.makedirs(output_dir)
                torch.save(tg_model, ckp_name)

            ### Exemplars
            nb_protos_cl = int(math.ceil(memory_size / (B + P * b)))

            print('nb_protos_cl = ' + str(nb_protos_cl))

            #tg_feature_model = nn.Sequential(*list(tg_model.children())[:-1])
            #print("\nBefore : list(tg_model.children()", list(tg_model.children()))
            if backbone=="resnetBasic" or backbone=="resnetBottleneck":
                tg_feature_model = nn.Sequential(*list(tg_model.children())[:-1])  
            elif backbone=="mobilenet" or backbone=="shufflenet":
                tg_feature_model = nn.Sequential(*list(tg_model.children())[:-2],nn.AdaptiveAvgPool2d((1, 1)))
            else : 
                print("wrong backbone")
            #print("\nAfter : ", tg_feature_model)
            num_features = tg_model.fc.in_features

            ## Herding - but here we do not use examplars
            #print('Updating exemplar set...')
            #first_index=B+(last_iter-1)*P
            #if first_index<B:
            #    first_index=0
            #for iter_dico in range(first_index, B+b*P): #  // originally range(last_iter*P, (b+1)*P)
            #    # Possible exemplars in the feature space and projected on the L2 sphere
            #    current_eval_set = merge_images_labels(prototypes[iter_dico], np.zeros(len(prototypes[iter_dico])))
            #    evalset.imgs = evalset.samples = current_eval_set
            #    evalloader = torch.utils.data.DataLoader(evalset, batch_size=eval_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
            #    num_samples = len(prototypes[iter_dico]) #number of images per class
            #    mapped_prototypes = compute_features(tg_feature_model, evalloader, num_samples, num_features)
            #    D = mapped_prototypes.T
            #    D = D/np.linalg.norm(D,axis=0)
            #    # Herding procedure : ranking of the potential exemplars
            #    mu  = np.mean(D,axis=1)
            #    index1 = int(iter_dico/P)
            #    index2 = iter_dico % P
            #    alpha_dr_herding[index1,:,index2] = alpha_dr_herding[index1,:,index2]*0
            #    w_t = mu
            #    iter_herding     = 0
            #    iter_herding_eff = 0
            #    while not(np.sum(alpha_dr_herding[index1,:,index2]!=0)==min(nb_protos_cl, num_samples)) and iter_herding_eff<1000:
            #        tmp_t   = np.dot(w_t,D)
            #        ind_max = np.argmax(tmp_t)
            #        iter_herding_eff += 1
            #        if alpha_dr_herding[index1,ind_max,index2] == 0:
            #            alpha_dr_herding[index1,ind_max,index2] = 1+iter_herding
            #            iter_herding += 1
            #        w_t = w_t+mu-D[:,ind_max]
            #if b > first_batch_number:
            #    print("================== debug 5 =====================")
            #    print("tg_model.fc.fc1.weight.data.size", tg_model.fc.fc1.weight.data.size())
            #    print("tg_model.fc.fc2.weight.data.size", tg_model.fc.fc2.weight.data.size())
            #
            ## Prepare the protoset
            #X_protoset_cumuls = []
            #Y_protoset_cumuls = []
            ## Class means for iCaRL and NCM + Storing the selected exemplars in the protoset
            #print('Computing mean-of_exemplars and theoretical mean...')
            class_means = np.zeros((num_features, num_classes, 2))
            #for b2 in range(b+1): # potentially some fixes on indexes TODO here 
            #    if b2==0:
            #        state_size=B
            #        first_index=0
            #    else:
            #        state_size=P
            #        first_index=B+(b2-1)*P
            #    for iter_dico in range(state_size):
            #        current_cl = order[range(first_index, B+b2*P)]
            #        current_eval_set = merge_images_labels(prototypes[first_index+iter_dico], \
            #            np.zeros(len(prototypes[first_index+iter_dico])))
            #        evalset.imgs = evalset.samples = current_eval_set
            #        evalloader = torch.utils.data.DataLoader(evalset, batch_size=eval_batch_size,
            #            shuffle=False, num_workers=num_workers, pin_memory=True)
            #        num_samples = len(prototypes[first_index+iter_dico])
            #        mapped_prototypes = compute_features(tg_feature_model, evalloader, num_samples, num_features)
            #        #shape mapped prototypes = 512 (features) x number of images per class = 500 for cifar
            #        D = mapped_prototypes.T
            #        D = D/np.linalg.norm(D, axis=0)
            #        D2 = D

            # class_means_name = os.path.join(output_dir, ckp_prefix + '_class_means_{}.pth'.format(b))
            class_means_name = os.path.join(ckp_dir, ckp_prefix + '_class_means_{}.pth'.format(b))

            #torch.save(class_means, class_means_name)

            current_means = class_means[:, order[range(B+(b)*P)]]
            ##############################################################
            # Calculate validation error of model on the cumul of classes:
            map_Y_valid_cumul = np.array([order_list.index(i) for i in Y_valid_cumul])
            current_eval_set = merge_images_labels(X_valid_cumul, map_Y_valid_cumul)
            evalset.imgs = evalset.samples = current_eval_set
            evalloader = torch.utils.data.DataLoader(evalset, batch_size=eval_batch_size,
                    shuffle=False, num_workers=num_workers, pin_memory=True)
            cumul_acc = compute_accuracy(tg_model, tg_feature_model, current_means, evalloader)

            ##############################################################
            top1_cnn_cumul_acc.append(float(str(cumul_acc[0])[:6]))
            top5_cnn_cumul_acc.append(float(str(cumul_acc[1])[:6]))

        print("###########################################################")
        print('TOP-1 detailed Results')
        print('SPB - CNN = '+str(top1_cnn_cumul_acc))
        print("###########################################################")
        print('TOP-5 detailed Results')
        print('SPB - CNN = '+str(top5_cnn_cumul_acc))
        print("###########################################################")
        print('mean inc accuracy')
        mean_top1_cnn_cumul_acc = np.mean(np.array(top1_cnn_cumul_acc)[1:])
        mean_top5_cnn_cumul_acc = np.mean(np.array(top5_cnn_cumul_acc)[1:])
        print('>>> results SPB_CNN | acc@1 = {:.2f} \t acc@5 = {:.2f} '.format(mean_top1_cnn_cumul_acc, mean_top5_cnn_cumul_acc))

# Print warnings (Possibly corrupt EXIF files):
if len(warn_list) > 0:
    print("\n" + str(len(warn_list)) + " Warnings\n")
    for i in range(len(warn_list)):
        print("warning " + str(i) + ":")
        print(str(i)+":"+ str(warn_list[i].category) + ":\n     " + str(warn_list[i].message))
else:
    print('No warnings.')