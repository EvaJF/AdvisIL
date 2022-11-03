import yaml
import pprint
from configparser import ConfigParser
import sys
import os
#from model_utils import model_namer -- circumvent torch imports

def model_namer(backbone, width_mult, depth_mult):
    model_name=backbone+'_w'+str(width_mult)+'_d'+str(depth_mult)
    return model_name

# loading configuration file
cp = ConfigParser()
cp.read(sys.argv[1])
cp = cp[os.path.basename(__file__)]
# reading parameters
yaml_folder_path = cp["yaml_folder_path"]
yaml_file_path = cp['yaml_file_path']
datasets_mean_std_file_path = cp['datasets_mean_std_file_path']
dataset_files_dir = cp['dataset_files_dir']
output_dir = cp['output_dir']
partition_gpu = cp['partition_gpu']
print(partition_gpu)

u_IL_method = "LUCIR"

# safely load yaml config file
def read_yaml(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)

def expe_writer(expe_dic, output_dir =  output_dir):
    """
    ### Example of expe_dic structure ###
    expe0:
    backbone : resnetBasic # base architecture
    width_mult : 1.0 # width multiplier
    depth_mult : 1.0 # depth multiplier
    normalization_dataset_name : inat # as written in the reference normalization file
    datasets_mean_std_file_path : "/home/users/efeillet/images_list_files/datasets_mean_std.txt"
    train_file_path : /home/users/efeillet/images_list_files/train100/inat/train.lst # text file with first line = root_path, folder for dataset, trian100 means no val set
    test_file_path : /home/users/efeillet/images_list_files/train100/inat/test.lst  
    B : 10 # number of classes in the first, non-incremental state
    P : 10 # number of classes per incremental state
    last_batch_number : 3
    IL_method : LUCIR  
    first_epochs : 2 # scratch / first batch
    subsequent_epochs :  2 # incremental states
    """

    ######### 1. CONFIG ##########

    ### Parameters ###
    # get base parameters
    backbone = expe_dic["backbone"]
    print(backbone)
    width_mult = expe_dic["width_mult"]
    depth_mult = expe_dic["depth_mult"]
    normalization_dataset_name = expe_dic["normalization_dataset_name"]
    datasets_mean_std_file_path = expe_dic["datasets_mean_std_file_path"]
    B = expe_dic["B"]
    P = expe_dic["P"] # number of classes per incremental batch
    last_batch_number = expe_dic["last_batch_number"]     
    train_file_path = expe_dic["train_file_path"] # /home/users/efeillet/images_list_files/train100/mininat/train.lst
    test_file_path = expe_dic["test_file_path"] # /home/users/efeillet/images_list_files/train100/mininat/test.lst
    first_epochs = expe_dic["first_epochs"]
    subsequent_epochs = expe_dic["subsequent_epochs"]
    IL_method = expe_dic["IL_method"]
    # define derivated parameters
    model_name = model_namer(backbone, width_mult, depth_mult)
    assert(B % P == 0)
    assert(B >= P)
    num_classes = B+P*(last_batch_number-1)
    S = int((num_classes - B) / P) + 1
    print('S = '+str(S))
    memory_size = 0
    ckp_prefix = '{}_s{}_k{}'.format(normalization_dataset_name, S, memory_size)
    
    ## Write configs ###
    lucir = """[main.py]
############################### ARCHITECTURE
backbone = {backbone}
width_mult = {width_mult}
depth_mult = {depth_mult}
############################## DATA
normalization_dataset_name = {normalization_dataset_name}
train_batch_size       = 128
test_batch_size        = 50
eval_batch_size        = 128
base_lr                = 0.1
lr_strat               = 40, 60
lr_factor              = 0.1
custom_weight_decay    = 0.0001
custom_momentum        = 0.9
train_file_path = {train_file_path}
test_file_path = {test_file_path} 
############################### EXPE
num_classes = {num_classes}
num_workers = 8
B = {B}
P = {P}
last_batch_number = {last_batch_number} 
memory_size = 0
output_dir = {output_dir}
first_epochs = {first_epochs} 
subsequent_epochs = {subsequent_epochs} 
############################### LUCIR
imprint_weights = True
mimic_score = True
less_forget = True
mr_loss = True
####################
cb_finetune = False
ft_epochs = 20
ft_base_lr = 0.001
ft_lr_strat = 10
ft_flag = 2
################### do not change
datasets_mean_std_file_path = {datasets_mean_std_file_path}
nb_runs = 1
rs_ratio = 0.0
resume = False
T = 2
beta=0.25
random_seed=-1
dist = 0.5
K = 2
lw_mr = 1.0
lw_ms = 1.0
lamda = 10.0
adapt_lamda = True
exists_val = False""".format(backbone=backbone, width_mult=width_mult, depth_mult=depth_mult,
    first_epochs=first_epochs, subsequent_epochs = subsequent_epochs,  
    num_classes = num_classes, normalization_dataset_name=normalization_dataset_name, 
    train_file_path=train_file_path, test_file_path=test_file_path,
    B=B, P=P, last_batch_number=last_batch_number,  output_dir= output_dir,
    datasets_mean_std_file_path=datasets_mean_std_file_path)

    ### Save config files ###
    # check is subdirs exist else create subdirs
    config_path = os.path.join(output_dir, 
                            normalization_dataset_name, 
                            ckp_prefix, 
                            model_name, 
                            IL_method,
                            "configs")
    for directory in [output_dir, os.path.join(output_dir, normalization_dataset_name),
        os.path.join(output_dir,normalization_dataset_name,ckp_prefix), 
        os.path.join(output_dir, normalization_dataset_name, ckp_prefix, model_name),
        os.path.join(output_dir, normalization_dataset_name, ckp_prefix, model_name, IL_method),
        config_path]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print("\n\nCreated folder %s" % directory)
    # open and write files
    for (text, filename) in [(lucir, "lucir.cf")]: 
        print("\nCreating new file with path %s" %os.path.join(config_path, filename))
        text_file = open(os.path.join(config_path, filename), "w")   
        #write string to file
        n = text_file.write(text)
        #close file
        text_file.close()
        print("Successfully wrote %s in %s" %(filename, config_path), '\n')


    ######### 2. LAUNCHER ##########

    # define derivated parameters    
    logs_path = os.path.join(output_dir, 
                               normalization_dataset_name, 
                               ckp_prefix, 
                               model_name, 
                               IL_method,
                               "logs")
    if not os.path.exists(logs_path):
        os.makedirs(logs_path)
        print("\n\nCreated folder %s for storing logs" %logs_path)
    error_path = os.path.join(logs_path, "error.log")
    log_path = os.path.join(logs_path, "log.log")
    config_path = os.path.join(output_dir, 
                               normalization_dataset_name, 
                               ckp_prefix, 
                               model_name, 
                               IL_method,
                               "configs")

    ### Write launcher file ###  
    launcher = """#!/usr/bin/env bash
#SBATCH --error={error_path}  
#SBATCH --output={log_path}  
#SBATCH --job-name=lucir
#SBATCH --partition={partition}
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
source /home/users/efeillet/miniconda3/bin/activate
conda activate py37
nvidia-smi 
srun --nodes=1 --ntasks=1 --gres=gpu:1 python /home/users/efeillet/incremental-scaler/LUCIR/codes/main.py {config_path}/lucir.cf 
""".format(error_path=error_path, log_path=log_path, partition=partition_gpu, config_path=config_path) 
     
    ### Save launcher file ###
    # open and write file
    expe_dir = os.path.join(output_dir, 
                               normalization_dataset_name, 
                               ckp_prefix, 
                               model_name, 
                               IL_method)
    print("\nCreating new file with path %s" %os.path.join(expe_dir,  "launcher.sh"))
    text_file = open(os.path.join(expe_dir,  "launcher.sh"), "w")   
    #write string to file
    n = text_file.write(launcher)
    #close file
    text_file.close()
    print("Successfully wrote launcher file\n")      
    return None


### Single laucher ###

def single_launcher(expe_list, u_dataset, u_prefix, partition = partition_gpu, output_dir =  output_dir):
    """
    Single launcher file will be stored at output_dir/unique/u_launcher.sh
    With logs at output_dir/unique/log/error.log and log.log
    """
    # root output dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print("\n\nCreated folder %s" % output_dir)
    # dir for unified launcher file
    u_dir = os.path.join(output_dir, u_dataset, u_prefix, "unique", u_IL_method)
    if not os.path.exists(u_dir):
        os.makedirs(u_dir)
        print("\n\nCreated folder %s" % u_dir)
    # logs subdir
    log_dir = os.path.join(u_dir, "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        print("\n\nCreated folder %s" % log_dir)  
    single_error_path = os.path.join(log_dir, "error.log")  
    single_log_path = os.path.join(log_dir, "log.log") 
    # beginning of unified launcher file 
    single_launcher ="""#!/usr/bin/env bash
#SBATCH --error={single_error_path}  
#SBATCH --output={single_log_path}  
#SBATCH --job-name=lucir
#SBATCH --partition={partition}
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
source /home/users/efeillet/miniconda3/bin/activate
conda activate py37
nvidia-smi 
""".format(single_error_path=single_error_path, single_log_path=single_log_path, 
    partition=partition)
    
    for expe in expe_list : 
        expe_dic = yaml_content[expe]
        print("\n\n >>>>> EXPE <<<<<")
        
        ######### 1. INDIVIDUAL CONFIG FILE ##########
        ### Parameters ###
        # get base parameters
        backbone = expe_dic["backbone"]
        print(backbone)
        width_mult = expe_dic["width_mult"]
        depth_mult = expe_dic["depth_mult"]
        normalization_dataset_name = expe_dic["normalization_dataset_name"]
        datasets_mean_std_file_path = expe_dic["datasets_mean_std_file_path"]
        B = expe_dic["B"]
        P = expe_dic["P"] # number of classes per incremental batch
        last_batch_number = expe_dic["last_batch_number"]     
        train_file_path = expe_dic["train_file_path"] # /home/users/efeillet/images_list_files/train100/mininat/train.lst
        test_file_path = expe_dic["test_file_path"] # /home/users/efeillet/images_list_files/train100/mininat/test.lst
        first_epochs = expe_dic["first_epochs"]
        subsequent_epochs = expe_dic["subsequent_epochs"]
        IL_method = expe_dic["IL_method"]
        # define derivated parameters
        model_name = model_namer(backbone, width_mult, depth_mult)
        assert(B % P == 0)
        assert(B >= P)
        num_classes = B+P*(last_batch_number-1)
        S = int((num_classes - B) / P) + 1
        print('S = '+str(S))
        memory_size = 0
        ckp_prefix = '{}_s{}_k{}'.format(normalization_dataset_name, S, memory_size)
        
        ## Write configs ###
        lucir = """[main.py]
############################### ARCHITECTURE
backbone = {backbone}
width_mult = {width_mult}
depth_mult = {depth_mult}
############################## DATA
normalization_dataset_name = {normalization_dataset_name}
train_batch_size       = 128
test_batch_size        = 50
eval_batch_size        = 128
base_lr                = 0.1
lr_strat               = 40, 60
lr_factor              = 0.1
custom_weight_decay    = 0.0001
custom_momentum        = 0.9
train_file_path = {train_file_path}
test_file_path = {test_file_path} 
############################### EXPE
num_classes = {num_classes}
num_workers = 8
B = {B}
P = {P}
last_batch_number = {last_batch_number} 
memory_size = 0
output_dir = {output_dir}
first_epochs = {first_epochs} 
subsequent_epochs = {subsequent_epochs} 
############################### LUCIR
imprint_weights = True
mimic_score = True
less_forget = True
mr_loss = True
####################
cb_finetune = False
ft_epochs = 20
ft_base_lr = 0.001
ft_lr_strat = 10
ft_flag = 2
################### do not change
datasets_mean_std_file_path = {datasets_mean_std_file_path}
nb_runs = 1
rs_ratio = 0.0
resume = False
T = 2
beta=0.25
random_seed=-1
dist = 0.5
K = 2
lw_mr = 1.0
lw_ms = 1.0
lamda = 10.0
adapt_lamda = True
exists_val = False""".format(backbone=backbone, width_mult=width_mult, depth_mult=depth_mult,
        first_epochs=first_epochs, subsequent_epochs = subsequent_epochs,   
        num_classes = num_classes, normalization_dataset_name=normalization_dataset_name, 
        train_file_path=train_file_path, test_file_path=test_file_path,
        B=B, P=P, last_batch_number=last_batch_number,  output_dir= output_dir,
        datasets_mean_std_file_path=datasets_mean_std_file_path)

        ### Save config files ###
        # create subdir
        config_path = os.path.join(output_dir, 
                                normalization_dataset_name, 
                                ckp_prefix, 
                                model_name, 
                                IL_method,
                                "configs")
        if not os.path.exists(config_path):
            os.makedirs(config_path)
            print("\n\nCreated folder %s" %config_path)
        # open and write files
        for (text, filename) in [(lucir, "lucir.cf")]: 
            print("\nCreating new file with path %s" %os.path.join(config_path, filename))
            text_file = open(os.path.join(config_path, filename), "w")   
            #write string to file
            n = text_file.write(text)
            #close file
            text_file.close()
            print("Successfully wrote %s in %s" %(filename, config_path), '\n')


        ######### 2. INDIVIDUAL LAUNCHER FILE ##########

        # define derivated parameters    
        logs_path = os.path.join(output_dir, 
                                normalization_dataset_name, 
                                ckp_prefix, 
                                model_name, 
                                IL_method,
                                "logs")
        if not os.path.exists(logs_path):
            os.makedirs(logs_path)
            print("\n\nCreated folder %s for storing logs" %logs_path)
        error_path = os.path.join(logs_path, "error.log")
        log_path = os.path.join(logs_path, "log.log")
        config_path = os.path.join(output_dir, 
                                normalization_dataset_name, 
                                ckp_prefix, 
                                model_name, 
                                IL_method,
                                "configs")

        ### Write launcher file ###  
        launcher = """#!/usr/bin/env bash
#SBATCH --error={error_path}  
#SBATCH --output={log_path}  
#SBATCH --job-name=lucir
#SBATCH --partition={partition}
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
source /home/users/efeillet/miniconda3/bin/activate
conda activate py37
nvidia-smi 
srun --nodes=1 --ntasks=1 --gres=gpu:1 python /home/users/efeillet/incremental-scaler/LUCIR/codes/main.py {config_path}/lucir.cf 
""".format(error_path=error_path, log_path=log_path, partition=partition_gpu, config_path=config_path) 
        
        ### Save launcher file ###
        # open and write file
        expe_dir = os.path.join(output_dir, 
                                normalization_dataset_name, 
                                ckp_prefix, 
                                model_name, 
                                IL_method)
        print("\nCreating new file with path %s" %os.path.join(expe_dir,  "launcher.sh"))
        text_file = open(os.path.join(expe_dir,  "launcher.sh"), "w")   
        #write string to file
        n = text_file.write(launcher)
        #close file
        text_file.close()
        print("Successfully wrote launcher file\n")   
        
        ######### 3. SINGLE UNIFIED LAUNCHER FILE ##########
        
        ### Write trial specific line ###
        # append echo
        single_launcher+="""echo EXPE /home/users/efeillet/incremental-scaler/LUCIR/codes/main.py {config_path}/lucir.cf\n""".format(
            config_path=config_path) 
        # append srun
        single_launcher+="""srun --nodes=1 --ntasks=1 --gres=gpu:1 python /home/users/efeillet/incremental-scaler/LUCIR/codes/main.py {config_path}/lucir.cf\n""".format(
            config_path=config_path)
    
    #print(single_launcher)
    ######## FINALLY ########    
    ### Save unified launcher file ###
    # open and write file
    u_path = os.path.join(u_dir,  "u_launcher_"+IL_method+".sh")
    print("\nCreating new file with path %s" %u_path)
    text_file = open(u_path, "w")   
    #write string to file
    n = text_file.write(single_launcher)
    #close file
    text_file.close()
    print("Successfully wrote unifed launcher file\n") 
    return None
    


### EXECUTE ###

if yaml_folder_path != "None": # a folder containing multiple yaml files is provided
    for filename in [filename for filename in os.listdir(yaml_folder_path) if filename.endswith(".yaml")]:
        print("\n Yaml file : ", filename)
        u_prefix = filename.split(".")[0] # ex : imagenet_random_0_siw_equi_s10
        u_dataset = '_'.join(u_prefix.split('_')[:-3])
        print(u_dataset, u_prefix)
        yaml_content = read_yaml(os.path.join(yaml_folder_path, filename))
        single_launcher(yaml_content, u_dataset, u_prefix)
elif yaml_file_path != "None" : # only one specific yaml file to process
    u_dataset = cp["u_dataset"]
    u_prefix = cp["u_prefix"]
    yaml_content = read_yaml(yaml_file_path)
    single_launcher(yaml_content, u_dataset, u_prefix)
#   for expe in yaml_content : 
#    expe_dic = yaml_content[expe]
#    print("\n\n >>>>> EXPE <<<<<")
#    pprint.pprint(expe_dic)
#    expe_writer(expe_dic)
else : 
    print("Wrong path to yaml file. Please check config file.")

# remind user that yaml files must be specific to a combination of dataset, number of states and repartition method: 
#"Unique dataset : {u_dataset}, unique prefix : {u_prefix}".format(u_dataset=u_dataset, u_prefix=u_prefix))
#   print("Caution ! you are assuming all experiments follow the same prefix ! ex : inat_s10_k0.")
#    print("If this is not the case, please use individual launcher files instead.")
# Else Use individual launchers instead of a unified launcher")


