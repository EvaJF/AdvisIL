import matplotlib
import pandas as pd
import pprint as pp
from configparser import ConfigParser
import os
import sys
import re
import matplotlib.pyplot as plt 
import numpy as np

from myMobileNet import myMobileNetV2
from myShuffleNet import myShuffleNetV2
from myResNet import myResNet, BasicBlock
from model_utils import count_params

### Preliminaries ###
# loading configuration file
cp = ConfigParser()
cp.read(sys.argv[1])
cp = cp[os.path.basename(__file__)]
# reading parameters
root_dir = cp['root_dir']
var_dataset= cp['dataset']
var_ntot_states = cp['ntot_states']
var_ntot_classes = cp['ntot_classes']
var_epochs_init = cp['epochs_init']
var_epochs_incr = cp['epochs_incr']
var_algo = cp['algo']

### Select log files to read results from ###
# method 1 : when log files are in subfolders corresponding to individual experiments
# ex : rootdir = .../expe/results/inat/inat_s10_k0
expe_list = [e for e in os.listdir(root_dir) if  e.startswith("resnet") or e.startswith("mobilenet") or e.startswith("shufflenet")] 
log_list = [(expe, os.path.join(root_dir, expe, "LUCIR", 'logs', 'log.log')) for expe in expe_list]
pp.pprint(log_list)

# method 2 : when several experiments are gathered in a singl log files, and several log files must be read too
# ex : rootdir = .../expe/scaling/imagenet_random_0/imagenet_random0_s10equi
#log_list = [os.path.join(root_dir, "unique", "LUCIR", "logs", file)
#         for file in os.listdir(root_dir) if file.endswith(".log") and file.startswith("log")]
# see result_analyser_concat.py for a script adapted to the ">>> results" flag

### Read results ###
backbone, w_mult, d_mult, acc1, acc5 = [], [], [], [], []
n_params, n_flops = [], []

# TODO adapt regex
for expe, path in log_list : 
    print(expe)
    try : 
        with open(path) as f:
            lines = f.readlines()
        results = [l for l in lines if l.startswith("LUCIR - CNN | acc@1")][0]
        acc1.append(float(results.split()[-4]))
        acc5.append(float(results.split()[-1]))
        backbone.append(expe.split('_')[0])
        w_mult.append(float(re.findall(r"[-+]?\d*\.*\d+", expe)[0]))
        d_mult.append(float(re.findall(r"[-+]?\d*\.*\d+", expe)[1]))
        print(results)
        print(acc1[-1], acc5[-1], backbone[-1], w_mult[-1], d_mult[-1])
    except:
        acc1.append(0.0)
        acc5.append(0.0)
        backbone.append(expe.split('_')[0])
        w_mult.append(float(re.findall(r"[-+]?\d*\.*\d+", expe)[0]))
        d_mult.append(float(re.findall(r"[-+]?\d*\.*\d+", expe)[1]))
L = len(log_list)
dataset = [var_dataset for i in range(L)]
ntot_states = [var_ntot_states for i in range(L)]
ntot_classes = [var_ntot_classes for i in range(L)]
epochs_init = [var_epochs_init for i in range(L)]
epochs_incr = [var_epochs_incr for i in range(L)]
algo = [var_algo for i in range(L)]

# More : compute number of params
print("\nNumber of experiments : ", L)
for i in range(L):
    b, w, d = backbone[i], w_mult[i], d_mult[i]
    #print("\%s --- Width mult : %s --- Depth mult : %s" %(b, w, d))
    if b == 'shufflenet' : 
        mynet = myShuffleNetV2(width_mult=w, depth_mult=d)
    if b == 'resnetBasic' :
        mynet = myResNet(block=BasicBlock, layers=[2, 2, 2, 2], width_mult=w, depth_mult=d)
    if b == 'mobilenet':
        mynet = myMobileNetV2(width_mult=w, depth_mult=d)
    n_params.append(count_params(mynet))

### Create & feed the dataframe ###
results_df = pd.DataFrame({'dataset':dataset, 'ntot_states':ntot_states, 'ntot_classes':ntot_classes, 
'epochs_init':epochs_init, 'epochs_incr':epochs_incr, 'algo':algo, 
'backbone':backbone, 'w_mult':w_mult, 'd_mult':d_mult, 'n_params':n_params, 
'acc1':acc1, 'acc5':acc5})


#save
results_df.to_csv(os.path.join(root_dir, 'results.csv'))

# todo :  add number of FLOPs as feature


### Let's plot! ###

# accuracy as a function of the number of parameters
print("Backbones : ", set(backbone))
for b in set(backbone): 
    fig, ax = plt.subplots()
    ordered_df = results_df[results_df['backbone']==b].sort_values('n_params', axis=0, ascending=True, inplace=False) 
    font1 = {'family':'serif','color':'blue','size':15}
    font2 = {'family':'serif','color':'black','size':15}
    ax.set_title("Accuracy = f(number of parameters) \n"+b+" architectures, e"+var_epochs_init+"/"+var_epochs_incr,
     fontdict = font1, loc='center')
    ax.set_xlabel("#params", fontdict = font2)
    ax.set_ylabel("accuracy", fontdict = font2)
    ax.plot(ordered_df['n_params'], ordered_df['acc5'], '-.', label = 'lucir acc@5')
    ax.plot(ordered_df['n_params'], ordered_df['acc1'], '-*', label = 'lucir acc@1')
    ax.legend()
    plt.savefig(os.path.join(root_dir, 'acc_nparams_'+b+'.png'))

# accuracy as a function of width for a fixed depth
for b in set(backbone): 
    fig, ax = plt.subplots() 
    W=list(set(results_df[(results_df.backbone==b)]["w_mult"]))
    colors = matplotlib.cm.get_cmap('viridis', len(W))(range(len(W)))
    color_id=0
    for w in sorted(W): 
        ordered_df = results_df[(results_df.backbone==b) & (results_df.w_mult==w)].sort_values('d_mult', axis=0, ascending=True, inplace=False) 
        ax.plot(ordered_df['d_mult'], ordered_df['acc5'], '-o', c=colors[color_id], label = 'width '+str(w))
        color_id+=1
    ax.legend()
    font1 = {'family':'serif','color':'blue','size':15}
    font2 = {'family':'serif','color':'black','size':15}
    ax.set_title("lucir acc@5 = f(depth) for fixed width multipliers \n"+b+" architectures, e"+var_epochs_init+"/"+var_epochs_incr, 
    fontdict = font1, loc='center')
    ax.set_xlabel("depth_mult", fontdict = font2)
    ax.set_ylabel("accuracy", fontdict = font2)
    plt.savefig(os.path.join(root_dir, 'acc_depth_'+b+'.png'))

# accuracy as a function of depth for a fixed width
for b in set(backbone): 
    fig, ax = plt.subplots()
    D=list(set(results_df[(results_df.backbone==b)]["d_mult"]))
    colors = matplotlib.cm.get_cmap('viridis', len(D))(range(len(D)))    
    color_id=0
    for d in sorted(list(set(results_df[(results_df.backbone==b)]["d_mult"]))):        
        ordered_df = results_df[(results_df.backbone==b) & (results_df.d_mult==d)].sort_values('w_mult', axis=0, ascending=True, inplace=False) 
        ax.plot(ordered_df['w_mult'], ordered_df['acc5'], '-o', c=colors[color_id], label = 'depth '+str(d))
        color_id+=1
    ax.legend()
    font1 = {'family':'serif','color':'blue','size':15}
    font2 = {'family':'serif','color':'black','size':15}
    ax.set_title("lucir acc@5 = f(width) for fixed depth multipliers \n"+b+" architectures, e"+var_epochs_init+"/"+var_epochs_incr,
     fontdict = font1, loc='center')
    ax.set_xlabel("width_mult", fontdict = font2)
    ax.set_ylabel("accuracy", fontdict = font2)
    plt.savefig(os.path.join(root_dir, 'acc_width_'+b+'.png'))


# accuracy in a  width by depth landscape
for b in set(backbone): 
    fig, ax = plt.subplots()       
    filtered_df = results_df[(results_df.backbone==b)]
    print(filtered_df['acc5'].tolist())
    scaled_acc = (filtered_df['acc5']-min(filtered_df['acc5']))/(max(filtered_df['acc5'])-min(filtered_df['acc5']))
    print(scaled_acc)
    ax.scatter(filtered_df['w_mult'], filtered_df['d_mult'], s=(filtered_df['acc5']+0.05)*10, alpha=0.5)
    #ax.legend()
    font1 = {'family':'serif','color':'blue','size':15}
    font2 = {'family':'serif','color':'black','size':15}
    ax.set_title("depth = f(width), area = f(acc@5) \n"+b+" architectures, e"+var_epochs_init+"/"+var_epochs_incr,
     fontdict = font1, loc='center')
    ax.set_xlabel("width_mult", fontdict = font2)
    ax.set_ylabel("depth_mult", fontdict = font2)
    plt.savefig(os.path.join(root_dir, 'scatter_acc_'+b+'.png'))

# accuracy in a  width by depth landscape
for b in set(backbone): 
    fig, ax = plt.subplots()       
    filtered_df = results_df[(results_df.backbone==b)]
    scaled_acc = (filtered_df['acc5']-min(filtered_df['acc5']))/(max(filtered_df['acc5'])-min(filtered_df['acc5']))
    ax.scatter(filtered_df['w_mult'], filtered_df['d_mult'], s=(scaled_acc+0.05)*200, c=scaled_acc, cmap='RdYlGn', alpha=0.5)
    font1 = {'family':'serif','color':'blue','size':15}
    font2 = {'family':'serif','color':'black','size':15}
    ax.set_title("depth = f(width), area = f(acc@5) \n"+b+" architectures, e"+var_epochs_init+"/"+var_epochs_incr,
     fontdict = font1, loc='center')
    ax.set_xlabel("width_mult", fontdict = font2)
    ax.set_ylabel("depth_mult", fontdict = font2)
    plt.savefig(os.path.join(root_dir, 'scatter_acc_'+b+'.png'))

# width in a  depth by accuracy landscape
for b in set(backbone): 
    fig, ax = plt.subplots()       
    filtered_df = results_df[(results_df.backbone==b)]
    scaled_w = (filtered_df['w_mult']-min(filtered_df['w_mult']))/(max(filtered_df['w_mult'])-min(filtered_df['w_mult']))
    ax.scatter(filtered_df['d_mult'], filtered_df['acc5'], s=(scaled_w+0.05)*200, c=scaled_w, cmap='PuOr', alpha=0.5)
    font1 = {'family':'serif','color':'blue','size':15}
    font2 = {'family':'serif','color':'black','size':15}
    ax.set_title("lucir acc@5 = f(depth), area = f(width) \n"+b+" architectures, e"+var_epochs_init+"/"+var_epochs_incr,
     fontdict = font1, loc='center')
    ax.set_xlabel("depth_mult", fontdict = font2)
    ax.set_ylabel("accuracy", fontdict = font2)
    plt.savefig(os.path.join(root_dir, 'scatter_width_'+b+'.png'))

