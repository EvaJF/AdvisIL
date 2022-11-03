import os, sys
import pandas as pd
from configparser import ConfigParser
from pprint import PrettyPrinter 
import re
from model_utils import count_params, model_builder

### Configs ###

# loading configuration file
cp = ConfigParser()
cp.read(sys.argv[1])
cp = cp[os.path.basename(__file__)]
# reading parameters
root_dir = cp["root_dir"]
save_dir = cp["save_dir"]
prefix = cp["prefix"]
ntot_classes_ref = int(cp["ntot_classes_ref"])
epochs_init_ref = int(cp["epochs_init_ref"])
epochs_incr_ref = int(cp["epochs_incr_ref"])

### Initialize lists to feed a DataFrame ###

# from parsing
dataset, ntot_states, backbone, w_mult, d_mult = [], [], [], [], []
acc1, acc5 = [], []
algo = []
# to be computed from parsed info
n_params = []
# to be defined from configs
ntot_classes, epochs_init, epochs_incr = [], [], []

### Parse log files and feed the lists ###

# log files only
log_files = [file for file in os.listdir(root_dir) if file.endswith(".log") and file.startswith("log")]
for file in log_files : 
    file_path = os.path.join(root_dir, file)
    with open(file_path, "r") as f :
        data = f.read()
        # parsing
        print("\n\n"+"-"*50, "\nParsing file : ", file_path)
        logs = data.split("EXPE") # list of logs
        for log in logs[1:] : # individual log files 
            print("\n>>> New Log <<<")  
            config = log.split(' ')[2].split('.cf')[0]
            #print("Info : ", info)
            info = config.split('/')
            scenario, network = info[7].split('_'), info[8].split('_')
            #print(scenario)
            #print(network)
            dataset_i = scenario[1]
            ntot_states_i = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", scenario[2])[0]
            backbone_i = network[0]
            w_mult_i = float(re.findall("\d+\.\d+", network[1])[0])
            d_mult_i = float(re.findall("\d+\.\d+", network[2])[0])
            algo_i = info[-1]
            print(dataset_i, ntot_states_i, backbone_i, w_mult_i, d_mult_i, algo_i)           
            try :         
                results = log.split('>>> results LUCIR_CNN | ')[1].split('\n')[0] # raw string with accuracies
                #print("\nTest 1", results)
                results = re.findall("\d+\.\d+", results) 
                #print("Results : ", results)
                acc1_i, acc5_i = float(results[0]), float(results[1]) # individual accuracy at 1 / at 5
                #print("\nTest 2",log.split(' ')[2].split('.cf')[0])
                print(acc1_i, acc5_i)
                print("Successful parsing")
                # feed the lists
                dataset.append(dataset_i)
                ntot_states.append(ntot_states_i)
                backbone.append(backbone_i)
                w_mult.append(w_mult_i)
                d_mult.append(d_mult_i) 
                algo.append(algo_i)
                acc1.append(acc1_i)
                acc5.append(acc5_i)
            except : 
                print("!!! Missing info or unrecognized patter. Check log file manually.")
                print(config)

# sanity checks on list lengths
print(len(dataset), len(ntot_states), len(backbone), len(w_mult), len(d_mult), len(algo) )          
print(len(acc1), len(acc5))
ntot_classes, epochs_init, epochs_incr = [ntot_classes_ref]*len(acc1), [epochs_init_ref]*len(acc1), [epochs_incr_ref]*len(acc1)
print(len(ntot_classes))

# compute n_params
for (ntot_classes_i, backbone_i, w_mult_i, d_mult_i) in zip(ntot_classes, backbone, w_mult, d_mult) :
    n_params_i = count_params(model_builder(ntot_classes_i, backbone_i, w_mult_i, d_mult_i))
    n_params.append(n_params_i)
print(len(n_params))

### Define the DataFrame ###
df = pd.DataFrame({
    "dataset" : dataset,
    "ntot_states" : ntot_states,
    "ntot_classes" : ntot_classes,
    "epochs_init" : epochs_init,
    "epochs_incr" : epochs_incr,
    "algo" : algo,
    "backbone" : backbone,
    "w_mult" : w_mult,
    "d_mult" : d_mult,
    "n_params" : n_params,
    "acc1" : acc1,
    "acc5" : acc5})

### Compute averages and standard deviations for each combination of hp ###

df_group = df.groupby(["backbone", "w_mult", "d_mult"])
print("Length of grouped df : ", len(df_group))
mean = df_group.mean()
stddev = df_group.std()

new_df = pd.DataFrame()
print(mean)
print(mean.columns)
print(mean.index)
new_backbone = [mean.index[i][0] for i in range(len(mean))]
new_w_mult = [mean.index[i][1] for i in range(len(mean))]
new_d_mult = [mean.index[i][2] for i in range(len(mean))]
new_df["backbone"] = new_backbone
new_df["w_mult"] = new_w_mult
new_df["d_mult"] = new_d_mult
for col in ['n_params', 'ntot_classes', 'epochs_init', 'epochs_incr', 'acc1', 'acc5']:
    new_df[col] = mean[col].tolist()
new_df["acc1_std"] = stddev.acc1.tolist()
new_df["acc5_std"] = stddev.acc5.tolist()

print(new_df)

### Save parsed results in a CSV file ###
path_raw_csv = os.path.join(save_dir, prefix+"raw_results_scaling.csv")
df.to_csv(path_raw_csv)
print("Saved raw results under %s" %path_raw_csv)
path_avg_csv = os.path.join(save_dir, prefix+"avg_results_scaling.csv")
new_df.to_csv(path_avg_csv)
print("Saved averaged results under %s" %path_avg_csv)




