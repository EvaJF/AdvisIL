from __future__ import division
from configparser import ConfigParser
import numpy as np
import torch as th
import AverageMeter as AverageMeter
import sys, os
import pandas as pd
from sklearn import preprocessing
from Utils import DataUtils
utils = DataUtils()

print("\n\n >>>>>>>>>>>> inFT SIW <<<<<<<<<<<<<<\n")


if len(sys.argv) != 2:
    print('Arguments : general_config')
    sys.exit(-1)

if not os.path.exists(sys.argv[1]):
    print('No configuration file found in the specified path')
    sys.exit(-1)


# loading configuration file
cp = ConfigParser()
cp.read(sys.argv[1])
cp = cp[os.path.basename(__file__)]

#Parameters###############################


batch_size = int(cp['batch_size'])
images_list_files_path =  cp['images_list_files_path']
scores_path = cp['ft_feat_scores_path'] # for scores files and faetures files
ft_weights_dir = cp['ft_weights_dir'] # for weight_bias files .pt (saved after training)
memory_size = cp['K']
B = int(cp['B'])
P = int(cp['P'])
S = int(cp['S'])
dataset = cp['dataset']
print('*****************************************************')
print('python '+ ' '.join(sys.argv))
###########################################

top1_accuracies = []
top1_accuracies_rectified = []
top5_accuracies = []
top5_accuracies_rectified = []

batch_initial_weight_matrix = {}
batch_initial_bias_vector = {}


#ft_weights_dir = os.path.join(ft_weights_dir, dataset)
#get first batch weights and bias
first_model_classif_param = th.load(os.path.join(ft_weights_dir, 'batch1_weight_bias.pt'))
np_first_model_weights = first_model_classif_param[0].detach().numpy()
np_first_model_bias = first_model_classif_param[1].detach().numpy()


batch_initial_weight_matrix[1] = np_first_model_weights
batch_initial_bias_vector[1] = np_first_model_bias


for batch_number in range(2, S + 1):
    print('*****************************************************')

    #test 
    test_images_paths_file = os.path.join(images_list_files_path,dataset,'test.lst')
    my_range = range(B+(batch_number-1)*P) # batch_number*P

    #test
    test_images_scores_file = os.path.join(scores_path,'test','batch'+str(batch_number),'scores')
    test_images_features_file = os.path.join(scores_path,'test','batch'+str(batch_number),'features')

    batch_ft_weights_path = os.path.join(ft_weights_dir, 'batch'+str(batch_number)+'_weight_bias.pt')
    model_classif_param = th.load(batch_ft_weights_path)
    np_model_weights = model_classif_param[0].detach().numpy()
    np_model_bias = model_classif_param[1].detach().numpy()


    #insert in the dict
    batch_initial_weight_matrix[batch_number] = np_model_weights[B + (batch_number - 2) * P : B + (batch_number - 1) * P] # (batch_number - 1) * P : batch_number*P]
    batch_initial_bias_vector[batch_number] = np_model_bias[B + (batch_number - 2) * P : B + (batch_number - 1) * P] # idem

    # TODO check if ok
    #erase current weights with initial batch weights:
    for b2 in range(1, batch_number):
        if b2 == 1 : 
            np_model_weights[:B] = batch_initial_weight_matrix[b2]
            np_model_bias[:B] = batch_initial_bias_vector[b2]
        else : 
            np_model_weights[B + (b2 - 2) * P : B + (b2 - 1) * P] = batch_initial_weight_matrix[b2]
            np_model_bias[B + (b2 - 2) * P : B +( b2 - 1) * P] = batch_initial_bias_vector[b2]

    for i in range(np_model_weights.shape[0]):
        mu = np.mean(np_model_weights[i])
        std = np.std(np_model_weights[i])
        np_model_weights[i] -= mu
        np_model_weights[i] /= std


    # np_model_weights = preprocessing.normalize(np_model_weights, axis=1, norm='l2')
    # np_model_bias = preprocessing.normalize(np_model_bias.reshape(1,-1), norm='l2')

###############################################################################################
    df = pd.read_csv(test_images_paths_file, sep=' ', names=['paths','class'])
    root_folder = df['paths'].head(1)[0]
    df = df.tail(df.shape[0] -1)
    df.drop_duplicates()
    df = df.sort_values('class')
    index_to_take = my_range
    samples = [(os.path.join(root_folder, elt[0]),elt[1]) for elt in list(map(tuple, df.loc[df['class'].isin(index_to_take)].values.tolist()))]
    samples.sort(key=lambda x:x[1])
    test_images_paths_file = samples#[elt[0] for elt in samples]
    test_images_scores_file = open(test_images_scores_file, 'r').readlines()
    test_images_features_file = open(test_images_features_file, 'r').readlines()

    assert(len(test_images_paths_file) == len(test_images_scores_file) ==len(test_images_features_file) )

    #print(len(test_images_paths_file))
    top1 = AverageMeter.AverageMeter()
    top5 = AverageMeter.AverageMeter()

    top1_rectified = AverageMeter.AverageMeter()
    top5_rectified = AverageMeter.AverageMeter()
    ################################################################################

    full_np_scores = None
    full_np_rectified_scores = None
    full_labels = []
    examples_counter = 0


    for (test_path_line, test_score_line, test_feat_line) in zip(test_images_paths_file, test_images_scores_file, test_images_features_file):
        #test_path_line = test_path_line.strip().split()
        test_score_line = test_score_line.strip().split()
        test_feat_line = test_feat_line.strip().split()
        test_image_path = test_path_line[0]
        test_image_class = int(test_path_line[1])
        test_np_scores = np.array(test_score_line, dtype=np.float)
        test_np_feat = np.array(test_feat_line, dtype=np.float).reshape(-1, 1)

        predicted_class = np.argmax(test_np_scores)


        test_rectified_np_scores = test_np_feat.T.dot(np_model_weights.T) + np_model_bias
        rectified_predicted_class = np.argmax(test_rectified_np_scores)

        full_labels.append(test_image_class)
        if full_np_rectified_scores is None:
            full_np_scores = test_np_scores
            full_np_rectified_scores = test_rectified_np_scores
        else:
            full_np_scores = np.vstack((full_np_scores, test_np_scores))
            full_np_rectified_scores = np.vstack((full_np_rectified_scores, test_rectified_np_scores))

        examples_counter += 1

        if examples_counter == batch_size:
            full_labels = th.from_numpy(np.array(full_labels, dtype=int))
            full_np_scores = th.from_numpy(full_np_scores)
            full_np_rectified_scores = th.from_numpy(full_np_rectified_scores)
            #compute accuracy
            prec1, prec5 = utils.accuracy(full_np_scores, full_labels, topk=(1, min(5, batch_number * P)))
            prec1_rectified, prec5_rectified = utils.accuracy(full_np_rectified_scores, full_labels, topk=(1, min(5, batch_number * P)))
            top1.update(prec1.item(), examples_counter)
            top5.update(prec5.item(), examples_counter)
            top1_rectified.update(prec1_rectified.item(), examples_counter)
            top5_rectified.update(prec5_rectified.item(), examples_counter)
            #re-init
            full_np_scores = None
            full_np_rectified_scores = None
            full_labels = []
            examples_counter = 0

    ##############################################################################
    if full_labels != []: #still missing some examples
        full_labels = th.from_numpy(np.array(full_labels, dtype=int))
        full_np_scores = th.from_numpy(full_np_scores)
        full_np_rectified_scores = th.from_numpy(full_np_rectified_scores)
        prec1, prec5 = utils.accuracy(full_np_scores, full_labels, topk=(1, min(5, batch_number * P)))
        prec1_rectified, prec5_rectified = utils.accuracy(full_np_rectified_scores, full_labels, topk=(1, min(5, batch_number * P)))
        top1.update(prec1.item(), examples_counter)
        top5.update(prec5.item(), examples_counter)
        top1_rectified.update(prec1_rectified.item(), examples_counter)
        top5_rectified.update(prec5_rectified.item(), examples_counter)


    #Accuracy
    print('[batch {}] Before Calibration | test : acc@1 = {:.1f}% ; acc@5 = {:.1f}%'.format(batch_number, top1.avg, top5.avg))
    print('[batch {}] After Calibration  | test : acc@1 = {:.1f}% ; acc@5 = {:.1f}%'.format(batch_number, top1_rectified.avg, top5_rectified.avg))
    # print('***********************************************************************')


    top1_accuracies.append(float(str(top1.avg*0.01)[:6]))
    top5_accuracies.append(float(str(top5.avg*0.01)[:6]))
    top1_accuracies_rectified.append(float(str(top1_rectified.avg*0.01)[:6]))
    top5_accuracies_rectified.append(float(str(top5_rectified.avg*0.01)[:6]))


print('*****************************************************')
print('TOP 1 Before calibration:')
print(top1_accuracies)
print('TOP 1 After calibration:')
print(top1_accuracies_rectified)
print('TOP 5 Before calibration:')
print(top5_accuracies)
print('TOP 5 After calibration:')
print(top5_accuracies_rectified)
print('TOP1 mean incremental accuracy = {:.1f} '.format(np.mean(np.array(top1_accuracies))*100))
print('TOP5 mean incremental accuracy = {:.1f} '.format(np.mean(np.array(top5_accuracies))*100))
print('*********After rectification**********')
print('TOP1 mean incremental accuracy = {:.1f} '.format(np.mean(np.array(top1_accuracies_rectified))*100))
print('TOP5 mean incremental accuracy = {:.1f} '.format(np.mean(np.array(top5_accuracies_rectified))*100))
# same format for all methods when printing final logs
print('>>> results inFT_SIW | acc@1 = {:.2f} \t acc@5 = {:.2f} '.format(np.mean(np.array(top1_accuracies_rectified))*100, 
                                                                        np.mean(np.array(top5_accuracies_rectified))*100))
