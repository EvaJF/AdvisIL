import matplotlib
import pandas as pd
import os
import matplotlib.pyplot as plt 

curr_dir = os.path.realpath(os.path.dirname(__file__)) # ./AdvisIL/scaling
root_dir = os.path.join(curr_dir, "logs")
dataset = "imagenet_random0" #"inat"
ntot_states = 10
ntot_classes = 100
epochs_init = 70
epochs_incr = 70
algo = "lucir"

input_csv = os.path.join(root_dir, "results_scaling_lucir_imagenetRd0.csv") # AdvisIL/scaling/logs/results_scaling_lucir_inat.csv
prefix = "test_random0_" 
results_df = pd.read_csv(input_csv)
print(results_df.columns)

# cool, turbo colormaps

        
def my_plot(b, mem, marker, mem_label, input_df):
    ### warning : filter out points exceedinig the budget ###
    results_df = input_df[(input_df.n_params<=mem[-1])|((input_df.w_mult==1.0) & (input_df.d_mult==1.0))]
    print(len(results_df))
    ########
    fig, ax = plt.subplots()
    fig.set_size_inches(7.5, 7.5, forward=True)
    ##box = ax.get_position()
    ##ax.set_position([box.x0, box.y0+0.01, box.width, box.height*1.1]) # occupy a bit more space (crop upper margin for title)
    # Shrink current axis's height by 10% on the bottom
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.22,
                    box.width, box.height * 0.9])
    D=list(set(results_df[(results_df.backbone==b)]["d_mult"]))
    colors = matplotlib.cm.get_cmap('cool', len(D))(range(len(D)))[::-1]    
    color_id=0
    # curves without markers; with n_params as annotations 
    for d in sorted(list(set(results_df[(results_df.backbone==b)]["d_mult"]))):        
        ordered_df = results_df[(results_df.backbone==b) & (results_df.d_mult==d)].sort_values('w_mult', axis=0, ascending=True, inplace=False) 
        ax.plot(ordered_df['w_mult'], ordered_df['acc1'], '-', c=colors[color_id], label = 'd = '+str(d), linewidth=4)        
        for k, n in enumerate(ordered_df.n_params):
            ax.annotate('  '+str(round(n/1e6, 1))+'M', (ordered_df['w_mult'].tolist()[k], ordered_df['acc1'].tolist()[k]), fontsize=14)  
            #'  {:.1e}'.format(txt)        
        color_id+=1
    # plot points with a marker code according to each budget
    for i in range(len(marker)):
        ordered_df = results_df[(results_df.backbone==b) & (results_df.n_params>mem[i]) & (results_df.n_params<=mem[i+1])].sort_values('w_mult', axis=0, ascending=True, inplace=False) 
        print("There are {} points under mem {}".format(len(ordered_df), mem[i+1]))
        print(mem_label[i])
        ax.plot(ordered_df['w_mult'], ordered_df['acc1'], marker[i], c='black', markersize=8, label = 'mem. '+str(round(float(mem_label[i])/1e6,1))+'M')  
        # heuristic
        h_df = ordered_df[(ordered_df.w_mult==ordered_df.w_mult.max()) & (ordered_df.d_mult==ordered_df.d_mult.min())]
        print(h_df)
        ax.plot(h_df['w_mult'], h_df['acc1'], marker[i], c='mediumseagreen', markersize=12, label = 'heuristic '+str(round(float(mem_label[i])/1e6, 1))+'M')
    # plot reference model
    ref_df = results_df[(results_df.backbone==b) & (results_df.w_mult==1.0) & (results_df.d_mult==1.0)]
    ax.plot(ref_df['w_mult'], ref_df['acc1'], 's', c='r', markersize=8, label = 'initial archi.')
    #ax.legend(loc='center left', bbox_to_anchor=(0.53, 0.35), fontsize=17) #loc="lower right", loc='center left', bbox_to_anchor=(1, 0.5)
    # Put a legend below current axis
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                fancybox=True, shadow=False, ncol=3, fontsize=12)

    font1 = {'family':'serif','color':'blue','size':18}
    font2 = {'family':'serif','color':'black','size':18}
    #ax.set_title("lucir acc@5 = f(width) for fixed depth multipliers \n"+b+" architectures, e"+str(epochs_init)+"/"+str(epochs_incr), fontdict = font1, loc='center')
    #ax.set_title(b, fontdict=font1)
    ax.set_xlabel("Width multiplier w", fontdict = font2)
    ax.set_ylabel("Average Incr. Acc. (%)", fontdict = font2)
    plt.xticks(fontsize=16) 
    plt.yticks(fontsize=16) 
    plt.savefig(os.path.join(root_dir, prefix + 'acc1_width_'+b+'.png'))
    print("saved under :", os.path.join(root_dir, prefix + 'acc1_width_'+b+'.png'))
    return None

for b in set(results_df.backbone): 
    print("\n"+b)
    if b =="resnetBasic":
        mem = [0, 2.5e6, 5e6 ] #[0, 1.5e6, 3e6 ] # memory budget  
        marker = ['*', "^"] # associated marker '+',
        mem_label = [str(round(n_params/1e6, 1))+'e6' for n_params in mem][1:]#'{:.1e}'.format(n_params)
    elif b == "mobilenet":
        mem = [0, 2.5e6, 5e6 ] # [0, 1.5e6, 3e6 ] # memory budget  
        marker = ['*', "^"] # associated marker '+',

        #mem = [0, 2.5e6, 5e6 ] # memory budget  
        #marker = ['*', "^"] # associated marker '+',
        mem_label = [str(round(n_params/1e6, 1))+'e6' for n_params in mem][1:]#'{:.1e}'.format(n_params)
    elif b == "shufflenet" :
        mem = [0, 2.5e6, 5e6 ] # [0, 1.5e6, 3e6 ] # memory budget  
        marker = ['*', "^"] # associated marker '+',

        #mem = [0, 5e6 ] # memory budget  
        #marker = ["^"] # associated marker '+',
        mem_label = [str(round(n_params/1e6, 1))+'e6' for n_params in mem][1:]#'{:.1e}'.format(n_params)
    else :
        print("Wrong backbone")
    my_plot(b, mem=mem, marker=marker, mem_label=mem_label, input_df=results_df)

