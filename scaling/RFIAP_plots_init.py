import matplotlib
import pandas as pd
import pprint as pp
from configparser import ConfigParser
import os
import sys
import re
import matplotlib.pyplot as plt 
import numpy as np


save_dir = "/home/data/efeillet/expe/scaling"

input_csv = "/home/data/efeillet/expe/scaling/fauna_avg_init_results_scaling.csv" #"/home/eva/Documents/code/rfiap_plots/results_MS.csv"
dataset = "imagenet_fauna" 
prefix = "init_fauna_" 

ntot_states = 10
ntot_classes = 100
epochs_init = 70
epochs_incr = 70
algo = "lucir"


results_df = pd.read_csv(input_csv)
print(results_df.columns)

# cool, turbo colormaps

def adjust_lightness(color, amount=0.3):
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])

def plot_v1(): 
    for b in set(results_df.backbone): 
        print("\n"+b)
        fig, ax = plt.subplots()
        #fig.set_size_inches(7.5, 5.0, forward=True)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0+0.01, box.width, box.height*1.1]) # occupy a bit more space (crop upper margin for title)
        D=list(set(results_df[(results_df.backbone==b)]["d_mult"]))
        colors = matplotlib.cm.get_cmap('cool', len(D))(range(len(D)))[::-1]    
        color_id=0
        # curves without markers; with n_params as annotations 
        for d in sorted(list(set(results_df[(results_df.backbone==b)]["d_mult"]))):        
            ordered_df = results_df[(results_df.backbone==b) & (results_df.d_mult==d)].sort_values('w_mult', axis=0, ascending=True, inplace=False) 
            ax.plot(ordered_df['w_mult'], ordered_df['acc5'], '-', c=colors[color_id], label = 'depth '+str(d), linewidth=4)        
            for k, n in enumerate(ordered_df.n_params):
                ax.annotate('  '+str(round(n/1e6, 1))+'e6', (ordered_df['w_mult'].tolist()[k], ordered_df['acc5'].tolist()[k]), fontsize=14)  
                #'  {:.1e}'.format(txt)        
            color_id+=1
        # plot points with a marker code according to each budget
        mem = [0, 2.5e6, 5e6 ] # memory budget  
        marker = ['*', "^"] # associated marker '+',
        mem_label = [str(round(n_params/1e6, 1))+'e6' for n_params in mem][1:]#'{:.1e}'.format(n_params)
        for i in range(len(marker)):
            ordered_df = results_df[(results_df.backbone==b) & (results_df.n_params>mem[i]) & (results_df.n_params<=mem[i+1])].sort_values('w_mult', axis=0, ascending=True, inplace=False) 
            print("There are {} points under mem {}".format(len(ordered_df), mem[i+1]))
            ax.plot(ordered_df['w_mult'], ordered_df['acc5'], marker[i], c='black', markersize=8, label = 'budget '+mem_label[i])  
            # heuristic
            h_df = ordered_df[(ordered_df.w_mult==ordered_df.w_mult.max()) & (ordered_df.d_mult==ordered_df.d_mult.min())]
            print(h_df)
            ax.plot(h_df['w_mult'], h_df['acc5'], marker[i], c='green', markersize=10, label = 'heuristique '+mem_label[i])
        # plot reference model
        ref_df = results_df[(results_df.backbone==b) & (results_df.w_mult==1.0) & (results_df.d_mult==1.0)]
        ax.plot(ref_df['w_mult'], ref_df['acc5'], 's', c='red', markersize=8, label = 'config initiale')
        ax.legend(loc='center left', bbox_to_anchor=(0.60, 0.3), fontsize=18) #loc="lower right", loc='center left', bbox_to_anchor=(1, 0.5)
        font1 = {'family':'serif','color':'blue','size':18}
        font2 = {'family':'serif','color':'black','size':18}
        #ax.set_title("lucir acc@5 = f(width) for fixed depth multipliers \n"+b+" architectures, e"+str(epochs_init)+"/"+str(epochs_incr), fontdict = font1, loc='center')
        #ax.set_title(b, fontdict=font1)
        ax.set_xlabel("coefficient de largeur", fontdict = font2)
        ax.set_ylabel("précision top 5 en %", fontdict = font2)
        plt.xticks(fontsize=16) 
        plt.yticks(fontsize=16) 
        plt.savefig(os.path.join(save_dir, prefix + 'acc_width_'+b+'.png'))
        
def my_plot(b, mem, marker, mem_label, input_df):
    ### warning : filter out points exceedinig the budget ###
    results_df = input_df[(input_df.n_params<=mem[-1])|((input_df.w_mult==1.0) & (input_df.d_mult==1.0))]
    print(len(results_df))
    ########
    fig, ax = plt.subplots()
    #fig.set_size_inches(7.5, 5.0, forward=True)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0+0.01, box.width, box.height*1.1]) # occupy a bit more space (crop upper margin for title)
    D=list(set(results_df[(results_df.backbone==b)]["d_mult"]))
    colors = matplotlib.cm.get_cmap('cool', len(D))(range(len(D)))[::-1]    
    color_id=0
    # curves without markers; with n_params as annotations 
    for d in sorted(list(set(results_df[(results_df.backbone==b)]["d_mult"]))):        
        ordered_df = results_df[(results_df.backbone==b) & (results_df.d_mult==d)].sort_values('w_mult', axis=0, ascending=True, inplace=False) 
        ax.plot(ordered_df['w_mult'], ordered_df['acc5'], '-', c=colors[color_id], label = 'depth '+str(d), linewidth=4)        
        for k, n in enumerate(ordered_df.n_params):
            ax.annotate('  '+str(round(n/1e6, 1))+'e6', (ordered_df['w_mult'].tolist()[k], ordered_df['acc5'].tolist()[k]), fontsize=14)  
            #'  {:.1e}'.format(txt)        
        color_id+=1
    # plot points with a marker code according to each budget
    for i in range(len(marker)):
        ordered_df = results_df[(results_df.backbone==b) & (results_df.n_params>mem[i]) & (results_df.n_params<=mem[i+1])].sort_values('w_mult', axis=0, ascending=True, inplace=False) 
        print("There are {} points under mem {}".format(len(ordered_df), mem[i+1]))
        ax.plot(ordered_df['w_mult'], ordered_df['acc5'], marker[i], c='black', markersize=8, label = 'budget '+mem_label[i])  
        # heuristic
        h_df = ordered_df[(ordered_df.w_mult==ordered_df.w_mult.max()) & (ordered_df.d_mult==ordered_df.d_mult.min())]
        print(h_df)
        ax.plot(h_df['w_mult'], h_df['acc5'], marker[i], c='mediumseagreen', markersize=12, label = 'heuristique '+mem_label[i])
    # plot reference model
    ref_df = results_df[(results_df.backbone==b) & (results_df.w_mult==1.0) & (results_df.d_mult==1.0)]
    ax.plot(ref_df['w_mult'], ref_df['acc5'], 's', c='r', markersize=8, label = 'config initiale')
    ax.legend(loc='center left', bbox_to_anchor=(0.53, 0.35), fontsize=17) #loc="lower right", loc='center left', bbox_to_anchor=(1, 0.5)
    font1 = {'family':'serif','color':'blue','size':18}
    font2 = {'family':'serif','color':'black','size':18}
    #ax.set_title("lucir acc@5 = f(width) for fixed depth multipliers \n"+b+" architectures, e"+str(epochs_init)+"/"+str(epochs_incr), fontdict = font1, loc='center')
    #ax.set_title(b, fontdict=font1)
    ax.set_xlabel("coefficient de largeur", fontdict = font2)
    ax.set_ylabel("précision top 5 en %", fontdict = font2)
    plt.xticks(fontsize=16) 
    plt.yticks(fontsize=16) 
    plt.savefig(os.path.join(save_dir, prefix + 'acc_width_'+b+'.png'))
    return None

def my_plot2(b, mem, marker, mem_label, input_df):
    ### warning : filter out points exceedinig the budget ###
    results_df = input_df[(input_df.n_params<=mem[-1])|((input_df.w_mult==1.0) & (input_df.d_mult==1.0))]
    print(len(results_df))
    ########
    fig, ax = plt.subplots()
    #fig.set_size_inches(7.5, 5.0, forward=True)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0+0.01, box.width, box.height*1.1]) # occupy a bit more space (crop upper margin for title)
    D=list(set(results_df[(results_df.backbone==b)]["d_mult"]))
    colors = matplotlib.cm.get_cmap('cool', len(D))(range(len(D)))[::-1]    
    color_id=0
    # curves without markers; with n_params as annotations 
    for d in sorted(list(set(results_df[(results_df.backbone==b)]["d_mult"]))):        
        ordered_df = results_df[(results_df.backbone==b) & (results_df.d_mult==d)].sort_values('w_mult', axis=0, ascending=True, inplace=False) 
        # standard dev as enveloppe
        ax.fill_between(ordered_df['w_mult'], 
            ordered_df['acc5']-ordered_df['acc5_std'], ordered_df['acc5']+ordered_df['acc5_std'],
            color = colors[color_id], alpha = 0.2) # adjust_lightness(colors[color_id])        
        # averaged values
        ax.plot(ordered_df['w_mult'], ordered_df['acc5'], '-', c=colors[color_id], label = 'depth '+str(d), linewidth=4)        
        for k, n in enumerate(ordered_df.n_params):
            ax.annotate('  '+str(round(n/1e6, 1))+'e6', (ordered_df['w_mult'].tolist()[k], ordered_df['acc5'].tolist()[k]), fontsize=14)  
            #'  {:.1e}'.format(txt)        
        color_id+=1
    # plot points with a marker code according to each budget
    for i in range(len(marker)):
        ordered_df = results_df[(results_df.backbone==b) & (results_df.n_params>mem[i]) & (results_df.n_params<=mem[i+1])].sort_values('w_mult', axis=0, ascending=True, inplace=False) 
        print("There are {} points under mem {}".format(len(ordered_df), mem[i+1]))
        ax.plot(ordered_df['w_mult'], ordered_df['acc5'], marker[i], c='black', markersize=8, label = 'budget '+mem_label[i])  
        # heuristic
        h_df = ordered_df[(ordered_df.w_mult==ordered_df.w_mult.max()) & (ordered_df.d_mult==ordered_df.d_mult.min())]
        print(h_df)
        ax.plot(h_df['w_mult'], h_df['acc5'], marker[i], c='mediumseagreen', markersize=12, label = 'heuristique '+mem_label[i])
    # plot reference model
    ref_df = results_df[(results_df.backbone==b) & (results_df.w_mult==1.0) & (results_df.d_mult==1.0)]
    ax.plot(ref_df['w_mult'], ref_df['acc5'], 's', c='r', markersize=8, label = 'config initiale')
    ax.legend(loc='center left', bbox_to_anchor=(0.53, 0.35), fontsize=17) #loc="lower right", loc='center left', bbox_to_anchor=(1, 0.5)
    font1 = {'family':'serif','color':'blue','size':18}
    font2 = {'family':'serif','color':'black','size':18}
    #ax.set_title("lucir acc@5 = f(width) for fixed depth multipliers \n"+b+" architectures, e"+str(epochs_init)+"/"+str(epochs_incr), fontdict = font1, loc='center')
    #ax.set_title(b, fontdict=font1)
    ax.set_xlabel("coefficient de largeur", fontdict = font2)
    ax.set_ylabel("précision top 5 en %", fontdict = font2)
    plt.xticks(fontsize=16) 
    plt.yticks(fontsize=16) 
    plt.savefig(os.path.join(save_dir, "filled_"+prefix + 'acc_width_'+b+'.png'))
    return None

# plt.fill_between(x, y-error, y+error)

for b in set(results_df.backbone): 
    print("\n"+b)
    if b =="resnetBasic":
        mem = [0, 1.5e6, 3e6, 6e6 ] #[0, 1.5e6, 3e6 ] # memory budget  
        marker = ['*', "^", "o"] # associated marker '+',
        mem_label = [str(round(n_params/1e6, 1))+'e6' for n_params in mem][1:]#'{:.1e}'.format(n_params)
    elif b == "mobilenet":
        mem = [0, 1.5e6, 3e6, 6e6  ] # [0, 1.5e6, 3e6 ] # memory budget  
        marker = ['*', "^", "o"] # associated marker '+',
        mem_label = [str(round(n_params/1e6, 1))+'e6' for n_params in mem][1:]#'{:.1e}'.format(n_params)
    elif b == "shufflenet" :
        mem = [0, 1.5e6, 3e6, 6e6 ] # [0, 1.5e6, 3e6 ] # memory budget  
        marker = ['*', "^", "o"] # associated marker '+',
        mem_label = [str(round(n_params/1e6, 1))+'e6' for n_params in mem][1:]#'{:.1e}'.format(n_params)
    else :
        print("Wrong backbone")
    my_plot2(b, mem=mem, marker=marker, mem_label=mem_label, input_df=results_df)

