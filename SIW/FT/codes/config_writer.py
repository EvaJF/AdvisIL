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
save_root_dir = cp['save_root_dir']
models_save_dir = cp['save_root_dir']
fc_params_destination_path = cp['save_root_dir']
partition_gpu = cp['partition_gpu']
print(partition_gpu)

u_IL_method = "SIW"

# safely load yaml config file
def read_yaml(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)

def config_writer(expe_dic, save_root_path = save_root_dir):
    """
    ### Example of expe_dic structure ###
    (expe0:)
        {dataset_v : mininat30 # for config folder / dataset name + number of classes used in total 
        backbone : resnetBasic # base architecture
        width_mult : 1.0 # width multiplier
        depth_mult : 1.0 # depth multiplier
        normalization_dataset_name : mininat # as written in the reference normalization file
        last_batch_number : 3 # also S, index
        num_batches : 3 # total number of batches to come
        P : 10 # number of classes per batch
        train_file_path : /your/path/to/AdvisIL/images_list_files/train100/mininat/train.lst # text file with first line = root_path, folder for dataset, trian100 means no val set
        val_file_path : /your/path/to/AdvisIL/images_list_files/train100/mininat/test.lst # caution, val is in fact test here
        algo_name : no_mem_ft # here memoryless fine_tuning
        IL_method : SIW # combined with standardization of initial weights as in Belouadah et al. 
        }
    """
    ### Parameters ###
    # get parameters
    backbone = expe_dic["backbone"]
    width_mult = expe_dic["width_mult"]
    depth_mult = expe_dic["depth_mult"]
    normalization_dataset_name = expe_dic["normalization_dataset_name"]
    last_batch_number = expe_dic["last_batch_number"] # also S
    num_batches = expe_dic["num_batches"]
    P = expe_dic["P"] # number of classes per incremental batch
    train_file_path = expe_dic["train_file_path"] # /your/path/to/AdvisIL/images_list_files/train100/mininat/train.lst
    val_file_path = expe_dic["val_file_path"] # /your/path/to/AdvisIL/images_list_files/train100/mininat/test.lst
    algo_name = expe_dic["algo_name"] #no_mem_ft
    #IL_method = expe_dic["IL_method"] # inFT_siw
    first_epochs = expe_dic["first_epochs"] # number of epochs for training in scratch
    subsequent_epochs = expe_dic["subsequent_epochs"] # number of epochs for training in the incremental states
    try : 
        B = expe_dic["B"] # semi
    except : 
        B = P # equi
    # define derivated parameters
    max_class= B-1 # max index of a class in the first batch / scratch
    model_name = model_namer(backbone, width_mult, depth_mult)
    batch1_model_name = '_'.join([normalization_dataset_name, 's'+str(num_batches), model_name, "batch1"])+".pt"
    first_batch_model_load_path = os.path.join(models_save_dir, normalization_dataset_name, 
        normalization_dataset_name+'_s'+str(num_batches), model_name, "scratch", batch1_model_name)
    # example of first_model_load_path /your/path/to/AdvisIL/expe/results/mininat/mininat_s3/resnetBasic_w1.0_d1.0/scratch/mininat_s3_resnetBasic_w1.0_d1.0_batch1.pt
    inc_models_load_path_prefix = os.path.join(models_save_dir, normalization_dataset_name, 
        normalization_dataset_name+'_s'+str(num_batches), model_name, algo_name+"_models", 
        '_'.join([algo_name, normalization_dataset_name, 's'+str(num_batches), model_name, "batch"]))
    # example  of inc_models_load_path_prefix /your/path/to/AdvisIL/expe/results/mininat/mininat_s3/resnetBasic_w1.0_d1.0/nomemft_models/nomemft_mininat_s3_resnetBasic_w1.0_d1.0_batch
    ft_feat_scores_path = os.path.join(models_save_dir, normalization_dataset_name, 
        normalization_dataset_name+'_s'+str(num_batches), model_name, algo_name+"_models", "weight_bias")
    # example of /your/path/to/AdvisIL/expe/results/mininat/mininat_s3/resnetBasic_w1.0_d1.0/nomemft_models/weight_bias/
    
    ### Write configs ###
    scratch =  """[scratch.py]
normalization_dataset_name = {normalization_dataset_name}
datasets_mean_std_file_path = {datasets_mean_std_file_path}
#######
num_workers = 1
gpu = 0
####### by default keep resnet18 as backbone network
backbone = {backbone}
width_mult = {width_mult}
depth_mult = {depth_mult}
########
old_batch_size= 128
new_batch_size= 32
val_batch_size= 128
#######
starting_epoch=0
num_epochs={first_epochs}
#######
lr_decay=0.1
lr=0.1
momentum=0.9
weight_decay=0.0005
patience=60
#######
train_file_path = {train_file_path}
val_file_path = {val_file_path}
max_class = {max_class}
#######
model_load_path = None
num_batches = {num_batches}
models_save_dir = {models_save_dir}
saving_intermediate_models = False
    """.format(normalization_dataset_name=normalization_dataset_name, datasets_mean_std_file_path=datasets_mean_std_file_path,
    backbone = backbone, width_mult = width_mult, depth_mult = depth_mult, 
    first_epochs=first_epochs,
    train_file_path = train_file_path, val_file_path = val_file_path, 
    max_class = max_class, num_batches = num_batches, models_save_dir = models_save_dir
    )

    no_mem_ft = """[no_mem_ft.py]
datasets_mean_std_file_path = {datasets_mean_std_file_path}
num_workers = 1
gpu = 0
##############################
backbone = {backbone}
width_mult = {width_mult}
depth_mult = {depth_mult}
########
old_batch_size=128
new_batch_size=32
test_batch_size=128
exemplars_batch_size = 128
starting_epoch=0
num_epochs = {subsequent_epochs}
lr_decay=0.1
lr=0.1
momentum=0.9
weight_decay=0.0005
patience = 15
normalization_dataset_name = {normalization_dataset_name}
############
first_batch_number = 2
last_batch_number = {last_batch_number}
B = {B}
P = {P}
##############################
algo_name = {algo_name}
num_batches = {num_batches}
dataset_files_dir = {dataset_files_dir}
first_batch_model_load_path =  {first_batch_model_load_path}
############################## DO NOT CHANGE
saving_intermediate_models = False
models_save_dir = {models_save_dir}
    """.format(datasets_mean_std_file_path=datasets_mean_std_file_path, 
    backbone = backbone, width_mult = width_mult, depth_mult = depth_mult, 
    subsequent_epochs=subsequent_epochs,
    normalization_dataset_name=normalization_dataset_name, last_batch_number=last_batch_number,
    P=P, B = B, algo_name=algo_name, num_batches=num_batches, 
    dataset_files_dir=dataset_files_dir, first_batch_model_load_path=first_batch_model_load_path,
    models_save_dir=models_save_dir)

    features_extraction = """[features_extraction.py]
num_workers = 1
batch_size = 128
gpu = 0
normalization_dataset_name = {normalization_dataset_name}
B = {B}
P = {P}
S = {last_batch_number}
datasets_mean_std_file_path = {datasets_mean_std_file_path}
########
backbone = {backbone}
width_mult = {width_mult}
depth_mult = {depth_mult}
########
first_batch_model_load_path = {first_batch_model_load_path}
inc_models_load_path_prefix =  {inc_models_load_path_prefix}
images_list_dir = {images_list_dir}
fc_params_destination_path = {fc_params_destination_path}
""".format(normalization_dataset_name = normalization_dataset_name, P = P, B = B, last_batch_number=last_batch_number,
    datasets_mean_std_file_path=datasets_mean_std_file_path, backbone=backbone, width_mult=width_mult, depth_mult=depth_mult,
    first_batch_model_load_path=first_batch_model_load_path, inc_models_load_path_prefix=inc_models_load_path_prefix, 
    images_list_dir=dataset_files_dir, fc_params_destination_path=fc_params_destination_path
    )

    inFT_siw = """[inFT_siw.py]
images_list_files_path = {dataset_files_dir}
ft_feat_scores_path = {ft_feat_scores_path}
ft_weights_dir = {ft_feat_scores_path}
K = 0
B = {B}
P = {P}
S = {num_batches}
dataset = {normalization_dataset_name}
batch_size = 256
    """.format(dataset_files_dir=dataset_files_dir,
    ft_feat_scores_path=ft_feat_scores_path, 
    P=P, B=B, num_batches=num_batches, normalization_dataset_name=normalization_dataset_name)

    ### Save config files ###
    # create subdir
    if not os.path.exists(save_root_path):
        os.makedirs(save_root_path)
        print("\n\nCreated folder %s" %save_root_path)
    save_root_path = os.path.join(save_root_path, normalization_dataset_name, 
        normalization_dataset_name+'_s'+str(num_batches), model_name, u_IL_method, "configs")
    if not os.path.exists(save_root_path):
        os.makedirs(save_root_path)
        print("\n\nCreated folder %s" %save_root_path)
    # open and write files
    for (text, filename) in [(scratch, "scratch.cf"), (no_mem_ft, "no_mem_ft.cf"), 
    (features_extraction, "features_extraction.cf"), (inFT_siw, 'inFT_siw.cf')]: 
        print("\nCreating new file with path %s" %os.path.join(save_root_path, filename))
        text_file = open(os.path.join(save_root_path, filename), "w")   
        #write string to file
        n = text_file.write(text)
        #close file
        text_file.close()
        print("Successfully wrote %s in %s" %(filename, save_root_path), '\n')

    return None

### launcher files ###

def launcher_writer(expe_dic, save_root_path = save_root_dir):
    
    ### Parameters ###
    # get parameters
    backbone = expe_dic["backbone"]
    width_mult = expe_dic["width_mult"]
    depth_mult = expe_dic["depth_mult"]
    normalization_dataset_name = expe_dic["normalization_dataset_name"]
    num_batches = expe_dic["num_batches"]
    
    # define derivated parameters
    model_name = model_namer(backbone, width_mult, depth_mult)
    logs_folder = os.path.join(save_root_path, normalization_dataset_name, 
        normalization_dataset_name+'_s'+str(num_batches), model_name, "logs")
    if not os.path.exists(logs_folder):
        os.makedirs(logs_folder)
        print("\n\nCreated folder %s for storing logs" %logs_folder)
    error_path = os.path.join(logs_folder, "error.log")
    log_path = os.path.join(logs_folder, "log.log")
    config_path = os.path.join(save_root_path, normalization_dataset_name, 
    normalization_dataset_name+'_s'+str(num_batches), model_name, u_IL_method, "configs")

    ### Write launcher file ###  
    launcher = """#!/usr/bin/env bash
#SBATCH --error={error_path}  
#SBATCH --output={log_path}  
#SBATCH --job-name=inFTsiw
#SBATCH --partition={partition}
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
source .../miniconda3/bin/activate
conda activate py37
nvidia-smi 
srun --nodes=1 --ntasks=1 --gres=gpu:1 python /your/path/to/SIW/FT/codes/scratch.py {config_path}/scratch.cf 
srun --nodes=1 --ntasks=1 --gres=gpu:1 python /your/path/to/SIW/FT/codes/no_mem_ft.py {config_path}/no_mem_ft.cf 
srun --nodes=1 --ntasks=1 --gres=gpu:1 python /your/path/to/SIW/FT/codes/features_extraction.py {config_path}/features_extraction.cf 
srun --nodes=1 --ntasks=1 --gres=gpu:1 python /your/path/to/SIW/FT/codes/inFT_siw.py {config_path}/inFT_siw.cf & wait
""".format(error_path=error_path, log_path=log_path, partition=partition_gpu, config_path=config_path) 
     
    ### Save launcher file ###
    # create subdir
    if not os.path.exists(save_root_path):
        os.makedirs(save_root_path)
        print("\n\nCreated folder %s" %save_root_path)
    save_root_path = os.path.join(save_root_path, normalization_dataset_name, 
        normalization_dataset_name+'_s'+str(num_batches), model_name, u_IL_method, "configs")
    if not os.path.exists(save_root_path):
        os.makedirs(save_root_path)
        print("\n\nCreated folder %s" %save_root_path)
    # open and write file
    filename = "launcher.sh"
    print("\nCreating new file with path %s" %os.path.join(save_root_path, filename))
    text_file = open(os.path.join(save_root_path, filename), "w")   
    #write string to file
    n = text_file.write(launcher)
    #close file
    text_file.close()
    print("Successfully wrote %s in %s" %(filename, save_root_path), '\n')      
    return None


def single_launcher(expe_list, u_dataset, u_prefix, partition = partition_gpu, output_dir = save_root_dir):
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
#SBATCH --job-name=siw
#SBATCH --partition={partition}
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
source .../miniconda3/bin/activate
conda activate py37
nvidia-smi 
""".format(single_error_path=single_error_path, single_log_path=single_log_path, 
    partition=partition)
    
    for expe in expe_list : 
        expe_dic = yaml_content[expe]
        print("\n\n >>>>> EXPE <<<<<")
        
        ######### 1. INDIVIDUAL CONFIG FILE ##########
        ### Parameters ###
        ### get base parameters       
        backbone = expe_dic["backbone"]
        width_mult = expe_dic["width_mult"]
        depth_mult = expe_dic["depth_mult"]
        normalization_dataset_name = expe_dic["normalization_dataset_name"]
        last_batch_number = expe_dic["last_batch_number"] # also S
        num_batches = expe_dic["num_batches"]
        B = expe_dic["B"] # number of classes in the first batch, semi or equi
        P = expe_dic["P"] # number of classes per incremental batch
        train_file_path = expe_dic["train_file_path"] # .../images_list_files/train100/mininat/train.lst
        val_file_path = expe_dic["val_file_path"] # .../images_list_files/train100/mininat/test.lst
        algo_name = expe_dic["algo_name"] #no_mem_ft
        IL_method = "SIW"
        first_epochs = expe_dic["first_epochs"] # number of epochs for training in scratch
        subsequent_epochs = expe_dic["subsequent_epochs"] # number of epochs for training in the incremental states
        ### define derivated parameters
        max_class= B-1 # max index of a class in the first batch / scratch
        model_name = model_namer(backbone, width_mult, depth_mult)
        batch1_model_name = '_'.join([normalization_dataset_name, 's'+str(num_batches), model_name, "batch1"])+".pt"
        memory_size = 0
        num_classes = B+P*(last_batch_number-1)
        S = int((num_classes - B) / P) + 1 # should be the same as last_batch_number
        print('S = '+str(S))
        ckp_prefix = '{}_s{}_k{}'.format(normalization_dataset_name, S, memory_size)
        first_batch_model_load_path = os.path.join(models_save_dir, normalization_dataset_name, 
            ckp_prefix, model_name, "scratch", batch1_model_name)
        inc_models_load_path_prefix = os.path.join(models_save_dir, normalization_dataset_name, 
            ckp_prefix, model_name, algo_name+"_models", 
            '_'.join([algo_name, normalization_dataset_name, 's'+str(num_batches), model_name, "batch"]))
        ft_feat_scores_path = os.path.join(models_save_dir, normalization_dataset_name, 
            ckp_prefix, model_name, algo_name+"_models", "weight_bias")

        ### Write configs ###
        scratch =  """[scratch.py]
normalization_dataset_name = {normalization_dataset_name}
datasets_mean_std_file_path = {datasets_mean_std_file_path}
#######
num_workers = 1
gpu = 0
####### by default keep resnet18 as backbone network
backbone = {backbone}
width_mult = {width_mult}
depth_mult = {depth_mult}
########
old_batch_size= 128
new_batch_size= 32
val_batch_size= 128
#######
starting_epoch=0
num_epochs={first_epochs}
#######
lr_decay=0.1
lr=0.1
momentum=0.9
weight_decay=0.0005
patience=60
#######
train_file_path = {train_file_path}
val_file_path = {val_file_path}
max_class = {max_class}
#######
model_load_path = None
num_batches = {num_batches}
models_save_dir = {models_save_dir}
saving_intermediate_models = False""".format(normalization_dataset_name=normalization_dataset_name, datasets_mean_std_file_path=datasets_mean_std_file_path,
        backbone = backbone, width_mult = width_mult, depth_mult = depth_mult, 
        first_epochs=first_epochs,
        train_file_path = train_file_path, val_file_path = val_file_path, 
        max_class = max_class, num_batches = num_batches, models_save_dir = models_save_dir
        )

        no_mem_ft = """[no_mem_ft.py]
datasets_mean_std_file_path = {datasets_mean_std_file_path}
num_workers = 1
gpu = 0
##############################
backbone = {backbone}
width_mult = {width_mult}
depth_mult = {depth_mult}
########
old_batch_size=128
new_batch_size=32
test_batch_size=128
exemplars_batch_size = 128
starting_epoch=0
num_epochs = {subsequent_epochs}
lr_decay=0.1
lr=0.1
momentum=0.9
weight_decay=0.0005
patience = 15
normalization_dataset_name = {normalization_dataset_name}
############
first_batch_number = 2
last_batch_number = {last_batch_number}
B = {B}
P = {P}
##############################
algo_name = {algo_name}
num_batches = {num_batches}
dataset_files_dir = {dataset_files_dir}
first_batch_model_load_path =  {first_batch_model_load_path}
############################## DO NOT CHANGE
saving_intermediate_models = False
models_save_dir = {models_save_dir}""".format(datasets_mean_std_file_path=datasets_mean_std_file_path, 
        backbone = backbone, width_mult = width_mult, depth_mult = depth_mult, 
        subsequent_epochs=subsequent_epochs,
        normalization_dataset_name=normalization_dataset_name, last_batch_number=last_batch_number,
        P=P, B=B, algo_name=algo_name, num_batches=num_batches, 
        dataset_files_dir=dataset_files_dir, first_batch_model_load_path=first_batch_model_load_path,
        models_save_dir=models_save_dir)

        features_extraction = """[features_extraction.py]
num_workers = 1
batch_size = 128
gpu = 0
normalization_dataset_name = {normalization_dataset_name}
B = {B}
P = {P}
S = {last_batch_number}
datasets_mean_std_file_path = {datasets_mean_std_file_path}
########
backbone = {backbone}
width_mult = {width_mult}
depth_mult = {depth_mult}
########
first_batch_model_load_path = {first_batch_model_load_path}
inc_models_load_path_prefix =  {inc_models_load_path_prefix}
images_list_dir = {images_list_dir}
fc_params_destination_path = {fc_params_destination_path}""".format(normalization_dataset_name = normalization_dataset_name, 
        P = P, B= B, last_batch_number=last_batch_number,
        datasets_mean_std_file_path=datasets_mean_std_file_path, backbone=backbone, width_mult=width_mult, depth_mult=depth_mult,
        first_batch_model_load_path=first_batch_model_load_path, inc_models_load_path_prefix=inc_models_load_path_prefix, 
        images_list_dir=dataset_files_dir, fc_params_destination_path=fc_params_destination_path
        )

        inFT_siw = """[inFT_siw.py]
images_list_files_path = {dataset_files_dir}
ft_feat_scores_path = {ft_feat_scores_path}
ft_weights_dir = {ft_feat_scores_path}
K = 0
B = {B}
P = {P}
S = {num_batches}
dataset = {normalization_dataset_name}
batch_size = 256""".format(dataset_files_dir=dataset_files_dir,
        ft_feat_scores_path=ft_feat_scores_path, 
        P=P, B=B,  num_batches=num_batches, normalization_dataset_name=normalization_dataset_name)
        ### Save config files ###
        # create subdir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print("\n\nCreated folder %s" %output_dir)
        save_root_path = os.path.join(output_dir, normalization_dataset_name, 
            ckp_prefix, model_name, u_IL_method, "configs")
        if not os.path.exists(save_root_path):
            os.makedirs(save_root_path)
            print("\n\nCreated folder %s" %save_root_path)
        # open and write files
        for (text, filename) in [(scratch, "scratch.cf"), (no_mem_ft, "no_mem_ft.cf"), 
        (features_extraction, "features_extraction.cf"), (inFT_siw, 'inFT_siw.cf')]: 
            print("\nCreating new file with path %s" %os.path.join(save_root_path, filename))
            text_file = open(os.path.join(save_root_path, filename), "w")   
            #write string to file
            n = text_file.write(text)
            #close file
            text_file.close()
            print("Successfully wrote %s in %s" %(filename, save_root_path), '\n')
                
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
        ### Write individual launcher file ###  
        launcher = """#!/usr/bin/env bash
#SBATCH --error={error_path}  
#SBATCH --output={log_path}  
#SBATCH --job-name=siw
#SBATCH --partition={partition}
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
source .../miniconda3/bin/activate
conda activate py37
nvidia-smi 
srun --nodes=1 --ntasks=1 --gres=gpu:1 python /your/path/to/AdvisIL/SIW/FT/codes/scratch.py {config_path}/scratch.cf 
srun --nodes=1 --ntasks=1 --gres=gpu:1 python /your/path/to/AdvisIL/SIW/FT/codes/no_mem_ft.py {config_path}/no_mem_ft.cf 
srun --nodes=1 --ntasks=1 --gres=gpu:1 python /your/path/to/AdvisIL/SIW/FT/codes/features_extraction.py {config_path}/features_extraction.cf 
srun --nodes=1 --ntasks=1 --gres=gpu:1 python /your/path/to/AdvisIL/SIW/FT/codes/inFT_siw.py {config_path}/inFT_siw.cf & wait
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
        single_launcher+="""echo EXPE .../AdvisIL/SIW/codes/main.py {config_path}/siw.cf\n""".format(
            config_path=config_path) 
        # append srun
        single_launcher+="""srun --nodes=1 --ntasks=1 --gres=gpu:1 python /your/path/to/AdvisIL/SIW/FT/codes/scratch.py {config_path}/scratch.cf 
srun --nodes=1 --ntasks=1 --gres=gpu:1 python /your/path/to/AdvisIL/SIW/FT/codes/no_mem_ft.py {config_path}/no_mem_ft.cf 
srun --nodes=1 --ntasks=1 --gres=gpu:1 python /your/path/to/AdvisIL/SIW/FT/codes/features_extraction.py {config_path}/features_extraction.cf 
srun --nodes=1 --ntasks=1 --gres=gpu:1 python /your/path/to/AdvisIL/SIW/FT/codes/inFT_siw.py {config_path}/inFT_siw.cf
""".format(config_path=config_path)
    
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
else : 
    print("Wrong path to yaml file. Please check config file.")

# remind user that yaml files must be specific to a combination of dataset, number of states and repartition method: 
#"Unique dataset : {u_dataset}, unique prefix : {u_prefix}".format(u_dataset=u_dataset, u_prefix=u_prefix))
#   print("Caution ! you are assuming all experiments follow the same prefix ! ex : inat_s10_k0.")
#    print("If this is not the case, please use individual launcher files instead.")
# Else Use individual launchers instead of a unified launcher")


#for expe in yaml_content : 
#    expe_dic = yaml_content[expe]
#    print("\n\n >>>>> EXPE <<<<<")
#    pprint.pprint(expe_dic)
#    config_writer(expe_dic)
#    launcher_writer(expe_dic)