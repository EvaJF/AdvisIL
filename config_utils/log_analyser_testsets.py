import pandas as pd
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sys


##############
### Config ###
output_dir = "/home/efeillet/Code/AdvisIL/testsets"

##################
### Input data ###
df1 = pd.read_csv("/home/efeillet/Code/AdvisIL/testsets/advisIL_test_avg_logs.csv")
## Add results from rd2 dataset  /!\/!\/!\/!\/!\/!\/!\/!\ 
df2 = pd.read_csv("/home/efeillet/Code/AdvisIL/advisIL_avg_logs_ref.csv")
df2 = df2[(df2.dataset == "imagenet_random_2")]
assert(list(df2.columns) == list(df1.columns))
input_df = pd.concat([df1, df2])
assert(len(input_df)==len(df1)+len(df2))
assert(list(input_df.columns) == list(df1.columns))
print("Input Length   : ", len(input_df))


##############################################################
### Dataset by dataset 

data_dic0 = {}
for b in input_df["model_name"].unique() : 
    for m in input_df["method"].unique() : 
        data_dic0[(m, b)] = 0
print(data_dic0)

for dataset in input_df.dataset.unique() : 
    df = input_df[input_df.dataset==dataset]
    # create null counter dic --> heatmap in format x = method, y = model_name
    data_dic = {}
    for b in df["model_name"].unique() : 
        for m in df["method"].unique() : 
            data_dic[(m, b)] = 0
    print(data_dic)
    # other format : x = budget, y = scenario
    # TODO
    data_dic2 = {}
    for b in df["budget"].unique() : 
        for n in df["num_states"].unique() : 
            data_dic2[(n, b)] = [] # will receive couples of backbones and methods
    print(data_dic2)
    # for each {scenario, num_states, budget} print the best {method, backbone} on average (average computed over all reference datasets)
    for num_states in df["num_states"].unique(): # proxy for scenario, replace with B & P
        for budget in df["budget"].unique():
            temp_df = df[(df["num_states"]==num_states) & (df["budget"]==budget)]
            print("\nBest accuracies for num_states {} and budget {}".format(num_states, budget))
            print(">>> Acc@1 : ", temp_df["acc@1"].max())
            print(temp_df[temp_df["acc@1"]==temp_df["acc@1"].max()][["method", "backbone", "model_name", "acc@1"]])
            # feed counter dic
            method = temp_df[temp_df["acc@1"]==temp_df["acc@1"].max()]["method"].tolist()[0]
            model_name = temp_df[temp_df["acc@1"]==temp_df["acc@1"].max()]["model_name"].tolist()[0]
            data_dic[(method, model_name)] += 1
            data_dic0[(method, model_name)] += 1
            data_dic2[(num_states, budget)].append((method, model_name))

    ## Plot heatmap in format x = method / y = model_name matrix for this scenario
    x_labels = sorted(df["method"].unique())
    y_labels = sorted(df["model_name"].unique())
    print(x_labels)
    print(y_labels)    
    data = np.array([[data_dic[(x,y)] for x in x_labels] for y in y_labels])
    print(data.shape)
    fig, ax = plt.subplots()
    im = ax.imshow(data)
    # Show all ticks and label them with the respective list entries
    ax.set_xticks([k for k in range(len(x_labels))])
    ax.set_xticklabels(x_labels)
    ax.set_yticks([k for k in range(len(y_labels))])
    ax.set_yticklabels(y_labels)
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    # Loop over data dimensions and create text annotations.
    for i in range(len(y_labels)):
        for j in range(len(x_labels)):
            text = ax.text(j, i, data[i, j], ha="center", va="center", color="w")
    ax.set_title(dataset+"\n# selections as best {method, model_name} over\n all scenarii+budget; criterion : top1 acc")
    fig.tight_layout()
    out_png = os.path.join(output_dir, dataset+"_heatmap_scenario+budget.png")
    plt.savefig(out_png)
    print("Saved fig under "+out_png)

### All datasets together (instead of mean acc)
## Plot heatmap in format method / model_name matrix for this scenario
x_labels = sorted(df["method"].unique())
y_labels = sorted(df["model_name"].unique())
print(x_labels)
print(y_labels)    
data = np.array([[data_dic0[(x,y)] for x in x_labels] for y in y_labels])
print(data.shape)
fig, ax = plt.subplots()
im = ax.imshow(data)
# Show all ticks and label them with the respective list entries
ax.set_xticks([k for k in range(len(x_labels))])
ax.set_xticklabels(x_labels)
ax.set_yticks([k for k in range(len(y_labels))])
ax.set_yticklabels(y_labels)
# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
# Loop over data dimensions and create text annotations.
for i in range(len(y_labels)):
    for j in range(len(x_labels)):
        text = ax.text(j, i, data[i, j], ha="center", va="center", color="w")
ax.set_title("All 4 test datasets"+"\n# selections as best {method, model_name} over\n all scenarii+budget; criterion : top1 acc")
fig.tight_layout()
out_png = os.path.join(output_dir, "data4_heatmap_scenario+budget.png")
plt.savefig(out_png)
print("Saved fig under "+out_png)

## Plot heatmap in format x = budget / y = scenario matrix for this scenario
# à faire à la main avec un tableur, je ne sais pas comment faire avec matplotlib

#sys.exit()
##############################################################
### Compute average accuracies over all reference datasets ### 

### Compute averages and standard deviations for each combination of hp ###
cols = ["method", "backbone", "budget", "model_name", "width_mult", "depth_mult", "scenario", "B", "P", "num_states", ]
df_group = input_df.groupby(cols)
print("Length : ",len(df_group))
mean = df_group.mean()
stddev = df_group.std()

avg_df = pd.DataFrame()
print("Mean df : ", mean)
print(mean.columns)
print(mean.index)

for j in range(len(cols)) : 
    #print(cols[j])
    new_col_j = [mean.index[i][j] for i in range(len(mean))]
    #print(new_col_j)
    avg_df[cols[j]] = new_col_j

for col in ['acc@1', 'acc@5']:
    avg_df[col] = mean[col].tolist()

avg_df["acc1_std"] = stddev["acc@1"].tolist()
avg_df["acc5_std"] = stddev["acc@5"].tolist()

print(avg_df)
#avg_df.to_csv(os.path.join(output_dir, "reference_avg_logs.csv"))

###############################################
### Basic frequency Analysis + Viz ### TEXT ###

##### Case 1 #####
# for each {scenario, num_states} print the best {method, backbone, model_name} on average (average computed over all reference datasets)
# NB : model name is a proxy for budget

# create null counter dic
data_dic = {}
for b in avg_df["model_name"].unique() : 
    for m in avg_df["method"].unique() : 
        data_dic[(m, b)] = 0
print(data_dic)

# compute best config for each scenario
for num_states in avg_df["num_states"].unique(): # warning : à changer par double filtre sur B et P plutôt    
    temp_df = avg_df[avg_df["num_states"]==num_states]
    print("\nBest top1 accuracy for num_states {} on average over all reference datatsets".format(num_states))
    print(">>> Acc@1 : ", temp_df["acc@1"].max())
    print(temp_df[temp_df["acc@1"]==temp_df["acc@1"].max()][["method", "backbone", "model_name", "budget", "acc@1"]])
    # feed counter dic
    method = temp_df[temp_df["acc@1"]==temp_df["acc@1"].max()]["method"].tolist()[0]
    model_name = temp_df[temp_df["acc@1"]==temp_df["acc@1"].max()]["model_name"].tolist()[0]
    data_dic[(method, model_name)] += 1

## Plot heatmap in format method / model_name matrix for this scenario
x_labels = sorted(avg_df["method"].unique())
y_labels = sorted(avg_df["model_name"].unique())
print(x_labels)
print(y_labels)    
data = np.array([[data_dic[(x,y)] for x in x_labels] for y in y_labels])
print(data.shape)
fig, ax = plt.subplots()
im = ax.imshow(data)
# Show all ticks and label them with the respective list entries
ax.set_xticks([k for k in range(len(x_labels))])
ax.set_xticklabels(x_labels)
ax.set_yticks([k for k in range(len(y_labels))])
ax.set_yticklabels(y_labels)
#ax.set_xticks(np.arange(len(x_labels)), labels=x_labels)
#ax.set_yticks(np.arange(len(y_labels)), labels=y_labels)
# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
# Loop over data dimensions and create text annotations.
for i in range(len(y_labels)):
    for j in range(len(x_labels)):
        text = ax.text(j, i, data[i, j], ha="center", va="center", color="w")
ax.set_title("# selections as best {method, model_name} over all scenarii \ntop1 acc averaged over 4 test datasets")
fig.tight_layout()
out_png = os.path.join(output_dir, "heatmap_scenario.png")
plt.savefig(out_png)
print("Saved fig under "+out_png)


##### Case 2 #####

# create null counter dic
data_dic = {}
for b in avg_df["model_name"].unique() : 
    for m in avg_df["method"].unique() : 
        data_dic[(m, b)] = 0
print(data_dic)

# for each {scenario, num_states, budget} print the best {method, backbone} on average (average computed over all reference datasets)
for num_states in avg_df["num_states"].unique(): # proxy for scenario, replace with B & P
    for budget in avg_df["budget"].unique():
        temp_df = avg_df[(avg_df["num_states"]==num_states) & (avg_df["budget"]==budget)]
        print("\nBest accuracies for num_states {} and budget {}".format(num_states, budget))
        print(">>> Acc@1 : ", temp_df["acc@1"].max())
        print(temp_df[temp_df["acc@1"]==temp_df["acc@1"].max()][["method", "backbone", "model_name", "acc@1"]])
        # feed counter dic
        method = temp_df[temp_df["acc@1"]==temp_df["acc@1"].max()]["method"].tolist()[0]
        model_name = temp_df[temp_df["acc@1"]==temp_df["acc@1"].max()]["model_name"].tolist()[0]
        data_dic[(method, model_name)] += 1

## Plot heatmap in format method / model_name matrix for this scenario
x_labels = sorted(avg_df["method"].unique())
y_labels = sorted(avg_df["model_name"].unique())
print(x_labels)
print(y_labels)    
data = np.array([[data_dic[(x,y)] for x in x_labels] for y in y_labels])
print(data.shape)
fig, ax = plt.subplots()
im = ax.imshow(data)
# Show all ticks and label them with the respective list entries
ax.set_xticks([k for k in range(len(x_labels))])
ax.set_xticklabels(x_labels)
ax.set_yticks([k for k in range(len(y_labels))])
ax.set_yticklabels(y_labels)
# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
# Loop over data dimensions and create text annotations.
for i in range(len(y_labels)):
    for j in range(len(x_labels)):
        text = ax.text(j, i, data[i, j], ha="center", va="center", color="w")
ax.set_title("# selections as best {method, model_name} over all scenarii+budget \ntop1 acc averaged over 4 test datasets")
fig.tight_layout()
out_png = os.path.join(output_dir, "heatmap_scenario+budget.png")
plt.savefig(out_png)
print("Saved fig under "+out_png)

## TODO VIZ for each scenario+budget, add heatmap for method / model_name matrix



##### Case 3 #####
# for each {method, scenario, num_states, budget} print the best {backbone, model, acc@1} on average (average computed over all reference datasets)
for num_states in avg_df["num_states"].unique():
    for budget in avg_df["budget"].unique():
        for method in avg_df["method"].unique():
            temp_df = avg_df[(avg_df["num_states"]==num_states) & (avg_df["budget"]==budget) & (avg_df["method"]==method)]
            print("\nBest accuracies for num_states {} and budget {} and method {}".format(num_states, budget, method))
            print(">>> Acc@1 : ", temp_df["acc@1"].max())
            print(temp_df[temp_df["acc@1"]==temp_df["acc@1"].max()][["backbone", "model_name", "acc@1"]])



### ORACLE vs reco_v0
# entrée utilisateur : scénario, budget, son dataset (pour l'instant, les datasets de ref)
# sortie souhaitée : méthode + backbone
# oracle :  la combinaison méthode + backbone donnant le meilleur score parmi tous les runs effectués pour ce scénario, avec ce budget et ce dataset
# dummy : choix par défaut en prenant la combinaison méthode + backbone la meilleure en moyenne sur tous les datasets de référence, sans distinction de budget ou de scénario
# dummy plus : affine selon le budget
# todo : dummy ++ propose la meilleure combi pour le budgt et le scénario d'entrée, sur la base de acc@1 moyenné sur les 5 datasets d'entrée

# enumerate possible user inputs with reference points
input_list = [] # "scenario", "budget", "dataset"
for num_states in input_df["num_states"].unique() : 
    for budget in input_df["budget"].unique() : 
        for dataset in input_df["dataset"].unique() : 
            input_list.append([num_states, budget, dataset])

# for a given user input, retrieve the best configuration from the reference points
def get_oracle(user_input, df = input_df): 
    """
    Input : 
    --------
    user_input : [num_states, budget, dataset]
    df : parsed logs


    Output : 
    --------
    best_config : the config you would obtain by trying all configurations 
        and picking the one with the best top 1 accuracy.
    """
    num_states, budget, dataset = user_input[0], user_input[1], user_input[2]
    temp_df = df[(df["num_states"]==num_states) & (df["budget"]==budget) & (df["dataset"]==dataset)]
    print("\nBest accuracy for num_states {} and budget {} and dataset {}".format(num_states, budget, dataset))
    print(">>> Acc@1 : ", temp_df["acc@1"].max())
    best = temp_df[temp_df["acc@1"]==temp_df["acc@1"].max()][["method", "backbone", "model_name", "acc@1"]]
    print(best)
    best_config = {"backbone" : best["backbone"].tolist()[0], "method" : best["method"].tolist()[0],  
                    "model_name" : best["model_name"].tolist()[0], "acc@1":best["acc@1"].tolist()[0]}
    return best_config

def get_dummy(user_input, df = input_df):
    """
    Input : 
    --------
    user_input : [num_states, budget, dataset]
    df : parsed logs

    Output : 
    --------
    dummy_reco : by default, backbone = mobilenet, method = spbm because best on average 
    """    
    best_backbone = "resnetBasic"
    best_method = "slda"
    temp_df = df[(df["num_states"]==num_states) & (df["budget"]==budget) & (df["dataset"]==dataset)  & (df["backbone"]==best_backbone)  & (df["method"]==best_method)]
    dummy_reco = {"backbone" : temp_df["backbone"].tolist()[0], "method" : temp_df["method"].tolist()[0],  
                    "model_name" : temp_df["model_name"].tolist()[0], "acc@1":temp_df["acc@1"].tolist()[0]}
    return dummy_reco

def get_dummy_plus(user_input, df = input_df):
    """
    Input : 
    --------
    user_input : [num_states, budget, dataset]
    df : parsed logs


    Output : 
    --------
    dummy_reco : by default, backbone = mobilenet, method = spbm because best on average 
    """    
    best_method = "slda"
    if budget == 1.5e6 :
        best_backbone = "resnetBasic"
    elif budget == 3e6 : 
        best_backbone = "shufflenet"
    else : 
        best_backbone = "mobilenet"
    
    temp_df = df[(df["num_states"]==num_states) & (df["budget"]==budget) & (df["dataset"]==dataset)  & (df["backbone"]==best_backbone)  & (df["method"]==best_method)]
    dummy_reco = {"backbone" : temp_df["backbone"].tolist()[0], "method" : temp_df["method"].tolist()[0],  
                    "model_name" : temp_df["model_name"].tolist()[0], "acc@1":temp_df["acc@1"].tolist()[0]}

    return dummy_reco

def reco_v0(user_input, df = avg_df) :
    """
    Input : 
    --------
    user_input : [num_states, budget, dataset]
    df : parsed logs


    Output : 
    --------
    best_config : the config you obtain by picking the best avg config over the reference runs 
        grouped by number of states (scenario) and budget.    
    """
    num_states, budget = user_input[0], user_input[1]
    # input : {num_states, budget, dataset} --> find best {method, backbone} in terms of mean acc@1 over all datasets
    # therefore, use the dataset containing accuracies averaged over all datasets
    temp_df = df[(df["num_states"]==num_states) & (df["budget"]==budget)]
    print("\nBest accuracy for num_states {} and budget {} / averaged over reference datasets".format(num_states, budget))
    print(">>> Acc@1 : ", temp_df["acc@1"].max())
    best = temp_df[temp_df["acc@1"]==temp_df["acc@1"].max()][["method", "backbone", "model_name", "acc@1"]]
    print(best)
    best_config = {"backbone" : best["backbone"].tolist()[0], "method" : best["method"].tolist()[0],  
                    "model_name" : best["model_name"].tolist()[0], "acc@1":best["acc@1"].tolist()[0]}
    return best_config    

def get_acc1(info_dic, df = avg_df) : 
    item = df[(df["num_states"]==info_dic["num_states"]) & (df["budget"]==info_dic["budget"]) & (df["dataset"]==info_dic["dataset"]) & (df["backbone"]==info_dic["backbone"]) & (df["method"]==info_dic["method"])]
    #print(item)
    assert(len(item)==1)
    acc1 = item["acc@1"].tolist()[0]
    return acc1

def get_noreco(user_input, df = input_df) : 
    """
    Input
    --------
    user_input : [num_states, budget, dataset]
    df : parsed logs

    Output
    --------
    dic with keys mean_acc, stddev_acc, max_acc, min_acc computed from the 3 backbones x 3 methods possible for a given user input
    """
    num_states, budget, dataset = user_input[0], user_input[1], user_input[2]
    temp_df = df[(df["num_states"]==num_states) & (df["budget"]==budget) & (df["dataset"]==dataset)]
    #assert(len(temp_df))==9
    mean_acc = temp_df["acc@1"].mean()
    std_acc = temp_df["acc@1"].std()
    min_acc = temp_df["acc@1"].min()
    max_acc = temp_df["acc@1"].max()
    mean_delta = max_acc - mean_acc 
    max_delta = max_acc - min_acc
    dic_acc = {    
    "mean_acc" :     mean_acc,
    "std_acc" :     std_acc,
    "min_acc" :     min_acc,
    "max_acc" :     max_acc,
    "mean_delta" :     mean_delta,
    "max_delta" :     max_delta,}
    return dic_acc

delta_dummy = []
delta_dummy_plus = []
mean_delta = []
max_delta = []
for user_input in input_list : 
    try : 
        print("\n-----------------------------------\nUser input : ", user_input)
        num_states, budget, dataset = user_input[0], user_input[1], user_input[2]
        oracle = get_oracle(user_input)
        dummy = get_dummy(user_input)
        dummy_plus = get_dummy_plus(user_input)
        no_reco = get_noreco(user_input)
        print("\nOracle     : ", oracle["acc@1"], "NB : best achievable accuracy in our reference points for this num_states & budget & database")
        print("\nDummy      : ", dummy["acc@1"], "NB : accuracy when choosing mobilenet+SPBM by default")
        print("\nDummy_plus : ", dummy_plus["acc@1"], "NB : accuracy when choosing either shufflenet (1.5M) or mobilenet (3M, 6M) + SPBM")
        d1 = oracle["acc@1"]-dummy["acc@1"]
        d2 = oracle["acc@1"]-dummy_plus["acc@1"]
        print("\nDelta acc@1 Oracle-Dummy      : ", d1)
        print("\nDelta acc@1 Oracle-Dummy_plus : ", d2)
        delta_dummy.append(d1)
        delta_dummy_plus.append(d2)
        mean_delta.append(no_reco["mean_delta"])
        max_delta.append(no_reco["max_delta"])
    except :
        print("Missing record")
        
print(len(delta_dummy))
print("\n\nAverages over all user inputs")
# oracls vs dummy recos
print("Mean Delta acc@1 Oracle_Best-Dummy      : ", np.array(delta_dummy).mean())
print("Mean Delta acc@1 Oracle_Best-Dummy_plus : ", np.array(delta_dummy_plus).mean())
# compute the mean gap between best acc and other accs for a given user input
print("Mean mean_discrepancy between Best acc and Any acc :", np.array(mean_delta).mean())
print("Mean max_discrepancy between Best acc and Any acc  :", np.array(max_delta).mean())
print("NB : Best and Any are computed input by input")



