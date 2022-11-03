import os
import sys
import matplotlib.pyplot as plt
import pandas as pd

root_dir = "/home/users/efeillet/expe/AdvisIL/hp_tuning"  # local : '/home/eva/Documents/code/hp_tuning'

def name_parser(filename):
    name = filename.split('.csv')[0].split('_')
    dico = {'net':name[1], 'width':name[2], 'depth':name[3]} #'width':float(name[2][1:]), 'depth':float(name[3][1:]
    
    #marker according to architecture
    if dico['net']=='resnetBasic' : # resnet
        dico['marker']='*'
    elif dico['net']=='shufflenet':
        dico['marker']='o' # shufflenet
    else : #mobilenet
        dico['marker'] = '^'
    # color according to budget and run
    if "samples60" in filename : # darker colors for finer tuning
        if (dico['width']=='w3.0' and dico['depth']=='d0.1') or (dico['width']=='w1.0' and dico['depth']=='d0.5') or (dico['width']=='w2.2' and dico['depth']=='d0.2'):
            dico['color']='blue' # large 6.0M params
            dico['budget']='6.0M'
        elif (dico['width']=='w2.6' and dico['depth']=='d0.1') or (dico['width']=='w0.6' and dico['depth']=='d0.5') or (dico['width']=='w1.4' and dico['depth']=='d0.2'):
            dico['color']='purple' # medium 3.0
            dico['budget']='3.0M'
        else : # small 1.5
            dico['color']='orange'
            dico['budget']='1.5M'
    else : # lighter colors for coarser tuning
        if (dico['width']=='w3.0' and dico['depth']=='d0.1') or (dico['width']=='w1.0' and dico['depth']=='d0.5') or (dico['width']=='w2.2' and dico['depth']=='d0.2'):
            dico['color']='paleturquoise' # large 6.0M params
            dico['budget']='6.0M'
        elif (dico['width']=='w2.6' and dico['depth']=='d0.1') or (dico['width']=='w0.6' and dico['depth']=='d0.5') or (dico['width']=='w1.4' and dico['depth']=='d0.2'):
            dico['color']='plum' # medium 3.0
            dico['budget']='3.0M'
        else : # small 1.5
            dico['color']='bisque'
            dico['budget']='1.5M'
    print(dico)
    
    return dico

### ALL ### Coarse + Fine

# filter : samples25 : coarse tuning; samples60 --> finer tuning
csv_files = sorted([file for file in os.listdir(root_dir) if file.endswith('.csv') ]) #and "samples25" in file
print(len(csv_files))
print(csv_files)

### BEST ACCURACY ###
fig, ax = plt.subplots()
x = [] # backbone type
y = [] # best accuracy
c = [] # color i.e. budget
for file in csv_files : 
    dico = name_parser(file)
    df = pd.read_csv(os.path.join(root_dir, file))
    x.append(dico['net'])
    y.append(max(df.mean_accuracy))
    c.append(dico['color'])
ax.scatter(x, y, c=c) 
ax.set_xlabel("backbone")
ax.set_ylabel("tuning_accuracy")
plt.savefig(os.path.join(root_dir, 'best_accuracy'+'.png'))


# filtered by  training_iteration == 70
# A plot for each hyper parameter
for hp in ['config.lr', 'config.momentum', 'config.weight_decay','config.lr_strat', 'config.train_batch_size']:
    # Show a cloud of points + a moustache box for each network configuration (architecture + budget)
    # architecture --> */o/^; budget --> blue, purple, orange 
    fig, ax = plt.subplots()
    for file in csv_files : 
        dico = name_parser(file)
        df = pd.read_csv(os.path.join(root_dir, file))
        print(len(df))
        df = df[df['training_iteration']==70]
        print(len(df))
        ax.plot(df[hp], df['mean_accuracy'], dico['marker'], c=dico['color'], label = dico['net']+' '+dico['budget'])
        #ax.boxplot()
    ax.legend(loc='best')
    ax.set_xlabel(hp)
    ax.set_ylabel("tuning_accuracy")
    plt.savefig(os.path.join(root_dir, hp+'.png')) #'_coarser'+


### COARSE ONLY ###

# filter : samples25 : coarse tuning
csv_files = sorted([file for file in os.listdir(root_dir) if file.endswith('.csv') and "samples25" in file]) #
print(len(csv_files))
print(csv_files)

### BEST ACCURACY ###
fig, ax = plt.subplots()
x = [] # backbone type
y = [] # best accuracy
c = [] # color i.e. budget
for file in csv_files : 
    dico = name_parser(file)
    df = pd.read_csv(os.path.join(root_dir, file))
    x.append(dico['net'])
    y.append(max(df.mean_accuracy))
    c.append(dico['color'])
ax.scatter(x, y, c=c) 
ax.set_xlabel("backbone")
ax.set_ylabel("tuning_accuracy")
plt.savefig(os.path.join(root_dir, 'best_accuracy_coarser'+'.png'))


# filtered by  training_iteration
# A plot for each hyper parameter
for hp in ['config.lr', 'config.momentum', 'config.weight_decay','config.lr_strat', 'config.train_batch_size']:
    # Show a cloud of points + a moustache box for each network configuration (architecture + budget)
    # architecture --> */o/^; budget --> blue, purple, orange 
    fig, ax = plt.subplots()
    for file in csv_files : 
        dico = name_parser(file)
        df = pd.read_csv(os.path.join(root_dir, file))
        print(len(df))
        df = df[df['training_iteration']>64]
        print(len(df))
        ax.plot(df[hp], df['mean_accuracy'], dico['marker'], c=dico['color'], label = dico['net']+' '+dico['budget'])
        #ax.boxplot()
    ax.legend(loc='best')
    ax.set_xlabel(hp)
    ax.set_ylabel("tuning_accuracy")
    plt.savefig(os.path.join(root_dir, hp+'_coarser'+'.png')) #

### FINER ONLY ###

# filter :  samples60 --> finer tuning
csv_files = sorted([file for file in os.listdir(root_dir) if file.endswith('.csv') and "samples60" in file]) #
print(len(csv_files))
print(csv_files)

### BEST ACCURACY ###
fig, ax = plt.subplots()
x = [] # backbone type
y = [] # best accuracy
c = [] # color i.e. budget
for file in csv_files : 
    dico = name_parser(file)
    df = pd.read_csv(os.path.join(root_dir, file))
    x.append(dico['net'])
    y.append(max(df.mean_accuracy))
    c.append(dico['color'])
ax.scatter(x, y, c=c) 
ax.set_xlabel("backbone")
ax.set_ylabel("tuning_accuracy")
plt.savefig(os.path.join(root_dir, 'best_accuracy_finer'+'.png'))


# filtered by  training_iteration
# A plot for each hyper parameter
for hp in ['config.lr', 'config.momentum', 'config.weight_decay','config.lr_strat', 'config.train_batch_size']:
    # Show a cloud of points + a moustache box for each network configuration (architecture + budget)
    # architecture --> */o/^; budget --> blue, purple, orange 
    fig, ax = plt.subplots()
    for file in csv_files : 
        dico = name_parser(file)
        df = pd.read_csv(os.path.join(root_dir, file))
        print(len(df))
        df = df[df['training_iteration']>64]
        print(len(df))
        ax.plot(df[hp], df['mean_accuracy'], dico['marker'], c=dico['color'], label = dico['net']+' '+dico['budget'])
        #ax.boxplot()
    ax.legend(loc='best')
    ax.set_xlabel(hp)
    ax.set_ylabel("tuning_accuracy")
    plt.savefig(os.path.join(root_dir, hp+'_finer'+'.png')) #