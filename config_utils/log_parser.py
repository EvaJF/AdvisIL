import os, sys
import pandas as pd
import numpy as np

root_dir = "/home/efeillet/Code/AdvisIL" # for saving result df //home/efeillet/Code/AdvisIL_results/Documents/code/AdvisIL/

# LUCIR, SPB & SIW
log_dirs = ['/home/efeillet/Code/AdvisIL_results/logs_fauna', '/home/efeillet/Code/AdvisIL_results/logs_flora', '/home/efeillet/Code/AdvisIL_results/logs_food', 
            '/home/efeillet/Code/AdvisIL_results/logs_random0', '/home/efeillet/Code/AdvisIL_results/logs_random1','/home/efeillet/Code/AdvisIL_results/logs_random2']
log_files = []
for dir in log_dirs : 
    log_files+=[os.path.join(dir, filename) for filename in os.listdir(dir) if filename.endswith('.log')] #'/home/data/efeillet/expe/AdvisIL/imagenet_fauna_lucir/imagenet_fauna_lucir.o.log'
log_files.sort()
print(log_files)

# DEESIL
deesil_dirs = ['/home/efeillet/Code/AdvisIL_results/logs_deesil/logs_deesil'] # '/home/efeillet/Code/AdvisIL_results/logs_deesil/advisil2/logs', '/home/efeillet/Code/AdvisIL_results/logs_deesil/advisil_part2/logs'
log_files_deesil = []
for dir in deesil_dirs : 
    log_files_deesil += [os.path.join(dir, filename) for filename in os.listdir(dir) if filename.endswith('.o.log')]

# Deep SLDA
slda_dirs = ['/home/efeillet/Code/AdvisIL_results/logs_slda']
log_files_slda = []
for dir in slda_dirs : 
    log_files_slda += [os.path.join(dir, filename) for filename in os.listdir(dir) if filename.endswith('.o.log')]

### Utility functions for parsing ###

def which_scenario(string):
    ## my runs
    if 's4' in string or 's10' in string or 's20'in string :
        scenario = "equi"
        if "s4" in string : 
            B, P = 25, 25
            num_states = 4
        elif "s10" in string :
            B, P = 10, 10
            num_states = 10
        elif "s20" in string : 
            B, P = 5, 5
            num_states = 20
    elif 's5' in string or 's7' in string or 's11' in string : 
        scenario = "semi"
        if "s5" in string : 
            B, P = 40, 15
            num_states = 5
        elif "s7" in string : 
            B, P = 40, 10
            num_states = 7
        elif "s11" in string : 
            B, P = 40, 6
            num_states = 11
    ## DeeSIL Adrian
    elif '_4.o' in string or '_10.o' in string or '_20.'in string :
        scenario = "equi"
        if "_4" in string : 
            B, P = 25, 25
            num_states = 4
        elif "_10" in string :
            B, P = 10, 10
            num_states = 10
        elif "_20" in string : 
            B, P = 5, 5
            num_states = 20
    elif '_5.o' in string or '_7.o' in string or '_11.o' in string : 
        scenario = "semi"
        if "_5" in string : 
            B, P = 40, 15
            num_states = 5
        elif "_7" in string : 
            B, P = 40, 10
            num_states = 7
        elif "_11" in string : 
            B, P = 40, 6
            num_states = 11
    else : 
        raise("Not a scenario descriptor.")
    return (scenario, B, P, num_states)
    
def header_parser(header):
    """
    Example string : /home/data/efeillet/expe/AdvisIL/imagenet_fauna/imagenet_fauna_s4_k0/resnetBasic_w0.5_d0.5/LUCIR/configs/lucir.cf
    consists in the path to a config file, contains every piece of information needed to characterize an experiment
    Pattern : <root_dir>/<dataset>/<dataset>_s<S>_k0/<model_name>/<method>/configs/<method>.cf
    """
    header = header.split('/')
    method = header[-3]
    model_name = header[-4]
    backbone = model_name.split('_')[0]
    width_mult = float(model_name.split('_')[1][1:])
    depth_mult = float(model_name.split('_')[2][1:])
    scenario, B, P, num_states = which_scenario(header[-5])
    dataset = header[-6]
    expe_dic = {  
    "method" : method.lower(),
    "model_name" : model_name,
    "backbone" : backbone,
    "width_mult" : width_mult,
    "depth_mult" : depth_mult,
    "scenario" : scenario,
    "B" : B,
    "P" : P,
    "num_states" : num_states,
    "dataset" : dataset,}
    return expe_dic

def result_parser(string, method):
    """
    Example string : 
    ###########################################################
    TOP-1 detailed Results
    LUCIR - CNN = [73.066, 49.666, 34.088, 26.849]
    ###########################################################
    TOP-5 detailed Results
    LUCIR - CNN = [92.933, 75.299, 59.711, 49.083]
    ###########################################################
    mean inc accuracy
    >>> results LUCIR_CNN | acc@1 = 36.87 	 acc@5 = 61.36 
    """
    if method == "lucir" or method == "spbm" or method == "siw" : 
        acc1 = float(string.split()[-4])
        acc5 = float(string.split()[-1])
    elif method =="deesil" : 
        acc1 = float(string.split()[-4][:-1])
        acc5 = float(string.split()[-1])
    elif method =="slda":
        acc1 = float(string.split()[4])
        acc5 = float(string.split()[9])
    else : 
        try : 
            acc1 = float(string.split()[-4])
            acc5 = float(string.split()[-1])
        except : 
            acc1 = np.nan
            acc5 = np.nan
    return acc1, acc5

def budget_finder(model_name):
    if model_name == "mobilenet_w1.0_d0.2" or model_name == "shufflenet_w1.6_d0.1" or model_name == "resnetBasic_w0.5_d0.5":
        budget = 1.5e6
    elif model_name == "mobilenet_w1.4_d0.2" or model_name == "shufflenet_w2.6_d0.1" or model_name == "resnetBasic_w0.6_d0.5":
        budget = 3e6
    elif model_name == "mobilenet_w2.2_d0.2" or model_name == "shufflenet_w3.0_d0.1" or model_name == "resnetBasic_w1.0_d0.5":
        budget = 6e6
    else : 
        budget = "unknown"
        raise("Wrong model_name provided : ", model_name)
    return budget

def parser_deesil(expe):
    """
    expe : list of lines from 1 individual experiment, DeeSIL log only
    """
    dataset = expe[0].split(' ')[-1]
    #print(dataset)
    model_name = [line for line in expe if line.startswith("Creating model")][0].split(' ')[-2]
    #print(model_name)
    backbone = model_name.split('_')[0]
    width_mult = float(model_name.split('_')[1][1:])
    depth_mult = float(model_name.split('_')[2][1:]) 
    #print(backbone, width_mult, model_name) 
    filename = file.split('/')[-1]
    scenario, B, P, num_states = which_scenario(filename)      
    expe_dic = {  
    "method" : "deesil",
    "model_name" : model_name,
    "backbone" : backbone,
    "width_mult" : width_mult,
    "depth_mult" : depth_mult,
    "scenario" : scenario,
    "B" : B,
    "P" : P,
    "num_states" : num_states,
    "dataset" : dataset,}
    # results : get mean incremental accuracy
    res = [line for line in expe if line.startswith(">>> results")]
    if len(res) == 1 : 
        acc1, acc5 = result_parser(res[0], expe_dic["method"])
        #print(acc1, acc5)
    else : 
        acc1, acc5 = np.nan, np.nan
        #print("Parsing failure on accuracy")
    expe_dic["acc1"] = acc1
    expe_dic["acc5"] = acc5
    expe_dic["budget"] = budget_finder(expe_dic["model_name"])
    return expe_dic

def parser_slda(expe):
    header = expe[0].split('/')
    dataset = header[-4]
    #print(dataset)
    model_name = header[-2]
    #print(model_name)
    backbone = model_name.split('_')[0]
    width_mult = float(model_name.split('_')[1][1:])
    depth_mult = float(model_name.split('_')[2][1:]) 
    #print(backbone, width_mult, model_name) 
    filename = file.split('/')[-1]
    scenario, B, P, num_states = which_scenario(filename)      
    expe_dic = {  
    "method" : "slda",
    "model_name" : model_name,
    "backbone" : backbone,
    "width_mult" : width_mult,
    "depth_mult" : depth_mult,
    "scenario" : scenario,
    "B" : B,
    "P" : P,
    "num_states" : num_states,
    "dataset" : dataset,}
    # results : get mean incremental accuracy
    res1 = [line for line in expe if line.startswith("Mean TOP1 accuracy =")]
    res5 = [line for line in expe if line.startswith("Mean TOP5 accuracy =")]
    if len(res1) == 1 : 
        acc1, acc5 = result_parser(res1[0]+' '+res5[0], expe_dic["method"])
        #print(acc1, acc5)
    else : 
        acc1, acc5 = np.nan, np.nan
        print("Parsing failure on accuracy")
    expe_dic["acc1"] = acc1
    expe_dic["acc5"] = acc5
    expe_dic["budget"] = budget_finder(expe_dic["model_name"])            
    #print(expe_dic)
    return expe_dic

# initialize result DataFrame
columns = ["method", "dataset", "model_name", "backbone", "width_mult", "depth_mult", "scenario", "B", "P", "num_states", "budget", "acc@1", "acc@5"]
rows = []
n_results = 0

# read logs DeepSLDA

slda_start = 0
slda_end = 0

for file in log_files_slda : 
    print("\n\n>>>>>>>>>>> Parsing DeedSLDA file {} <<<<<<<<<<<<<<<\n".format(file))
    with open(file) as f:
        # sanity check : have all experiments run successfully ?
        lines = f.readlines()
        headers = [line for line in lines if line.startswith("Features:")]
        results1 = [line for line in lines if line.startswith("Mean TOP1 accuracy =")]
        results5 = [line for line in lines if line.startswith("Mean TOP5 accuracy =")]
        print("Number of experiments : {}".format(len(headers)))
        print("Number of results     : {}".format(len(results1)))
        slda_start += len(headers)
        slda_end += len(results1)
        n_results += len(results1)
    with open(file) as f:
        # actual parsing : split log files into segments corresponding to each experiment
        # in case some expeiments did not run correctly
        expe_list = f.read().split("Features:")
        print(len(expe_list))
        for expe in expe_list[1:] : 
            expe = expe.split('\n')
            print(expe[:5])
            expe_dic = parser_slda(expe)
            # add row with structure
            # ["method", "dataset", "model_name", "backbone", "width_mult", "depth_mult", "scenario", "B", "P", "num_states", "budget", "acc@1", "acc@5"]
            rows.append([expe_dic["method"] , expe_dic["dataset"], expe_dic["model_name"], expe_dic["backbone"], expe_dic["width_mult"], expe_dic["depth_mult"], 
                expe_dic["scenario"], expe_dic["B"], expe_dic["P"], expe_dic["num_states"], expe_dic["budget"] , expe_dic["acc1"], expe_dic["acc5"]])             

print("DeepSLDA : ", slda_start, slda_end)


# read logs DEESIL
deesil_start = 0
deesil_end = 0

for file in log_files_deesil : 
    print("\n\n>>>>>>>>>>> Parsing DeeSIL file {} <<<<<<<<<<<<<<<\n".format(file))
    with open(file) as f:
        # sanity check : have all experiments run successfully ?
        lines = f.readlines()
        headers = [line for line in lines if line.startswith("normalization dataset name")]
        results = [line for line in lines if line.startswith(">>> results top1")]
        print("Number of experiments : {}".format(len(headers)))
        print("Number of results     : {}".format(len(results)))
        deesil_start += len(headers)
        deesil_end += len(results)
        n_results += len(results)
    with open(file) as f:
        # actual parsing : split log files into segments corresponding to each experiment
        # in case some expeiments did not run correctly
        expe_list = f.read().split("normalization dataset name")
        #print(len(expe_list))
        for expe in expe_list[1:] : 
            expe = expe.split('\n')
            #print(expe[:5])
            expe_dic = parser_deesil(expe)
            # add row with structure
            # ["method", "dataset", "model_name", "backbone", "width_mult", "depth_mult", "scenario", "B", "P", "num_states", "budget", "acc@1", "acc@5"]
            rows.append([expe_dic["method"] , expe_dic["dataset"], expe_dic["model_name"], expe_dic["backbone"], expe_dic["width_mult"], expe_dic["depth_mult"], 
                expe_dic["scenario"], expe_dic["B"], expe_dic["P"], expe_dic["num_states"], expe_dic["budget"] , expe_dic["acc1"], expe_dic["acc5"]])             

print("DeeSIL : ", deesil_start, deesil_end)

# read logs LUCIR + SPB + SIW
for file in log_files: 
    print("\n\n>>>>>>>>>>> Parsing file {} <<<<<<<<<<<<<<<\n".format(file))
    with open(file) as f:
        # sanity check : have all experiments run successfully ?
        lines = f.readlines()
        headers = [line for line in lines if line.startswith("EXPE /home/")]
        results = [line for line in lines if line.startswith(">>> results")]
        print("Number of experiments : {}".format(len(headers)))
        print("Number of results     : {}".format(len(results)))
        n_results += len(results)
    with open(file) as f:
        # actual parsing : split log files into segments corresponding to each experiment
        # in case some expeiments did not run correctly
        expe_list = f.read().split("EXPE")
        #print(len(expe_list))
        for expe in expe_list[1:] : 
            expe = expe.split('\n')
            # header : path to config file
            #print(expe[:3])
            header = expe[0].split(' ')[-1]
            #print(header)
            expe_dic = header_parser(header)
            # results : get mean incremental accuracy
            res = [line for line in expe if line.startswith(">>> results")]
            if len(res) == 1 : 
                acc1, acc5 = result_parser(res[0], expe_dic["method"])
                #print(acc1, acc5)
            else : 
                acc1, acc5 = np.nan, np.nan
                #print("Parsing failure on accuracy")
            expe_dic["acc1"] = acc1
            expe_dic["acc5"] = acc5
            expe_dic["budget"] = budget_finder(expe_dic["model_name"])
            # add row with structure
            # ["method", "dataset", "model_name", "backbone", "width_mult", "depth_mult", "scenario", "B", "P", "num_states", "budget", "acc@1", "acc@5"]
            if "tofilter" in file and expe_dic["backbone"] != "resnetBasic" : 
                print("DISCARD sample ", file, expe_dic)
                # discard results from first SPB run with mobilenets and shufflenets
            else : # ok
                rows.append([expe_dic["method"] , expe_dic["dataset"], expe_dic["model_name"], expe_dic["backbone"], expe_dic["width_mult"], expe_dic["depth_mult"], 
                    expe_dic["scenario"], expe_dic["B"], expe_dic["P"], expe_dic["num_states"], expe_dic["budget"] , expe_dic["acc1"], expe_dic["acc5"]])             



# fill result table 
result_df = pd.DataFrame(rows, columns = columns)
print("Raw length : ",  len(result_df))

# filter out nan values
result_df = result_df[result_df["acc@1"].notnull()]
print("After filtering out nan acc : ", len(result_df))
result_df.to_csv(os.path.join(root_dir, "raw_logs_ref.csv"))
# check for doublons 
check_df = result_df.groupby(["method", "dataset", "model_name", "scenario", "num_states"]).count()
print("Doublons : ", len(check_df[check_df["acc@1"]>1]))
print(check_df[check_df["acc@1"]>1])


# average results when multiple runs (SIW) 
### Compute averages and standard deviations for each combination of hp ###
cols = ["method", "dataset", "model_name", "backbone", "width_mult", "depth_mult", "scenario", "B", "P", "num_states", "budget"]
df_group = result_df.groupby(cols)
print("Number of distinct experiments: ",len(df_group))
mean = df_group.mean()
stddev = df_group.std()
avg_df = pd.DataFrame()
print("Computing mean dataframe")
print("mean", mean)
print("stddev", stddev)
for j in range(len(cols)) : 
    #print(cols[j])
    new_col_j = [mean.index[i][j] for i in range(len(mean))]
    #print(new_col_j)
    avg_df[cols[j]] = new_col_j
for col in ['acc@1', 'acc@5']:
    avg_df[col] = mean[col].tolist()
avg_df["acc1_std"] = stddev["acc@1"].tolist()
avg_df["acc5_std"] = stddev["acc@5"].tolist()
#print(avg_df)
avg_df.to_csv(os.path.join(root_dir, "advisIL_avg_logs_ref.csv"))
print("Length of averaged dataset :", len(avg_df))

### sanity checks ###
print("Averaged results, grouped by method and dataset")
print(avg_df.groupby(["method", "dataset"]).count()) # "model_name", "scenario", "num_states"
print("Averaged results, grouped by individual experiment")
check_df = avg_df.groupby(["method", "dataset", "model_name", "scenario", "num_states"]).count()
check_df.to_csv(os.path.join(root_dir, "check_logs_ref.csv"))
print(check_df)



        


