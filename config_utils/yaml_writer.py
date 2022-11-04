import os

# where to store yaml files
destination_path = "/home/data/efeillet/expe/AdvisIL/yaml_files"
# scenarii : (prefix, total number of states, number of classes int he first batch, number of classes per incremental state).
state_list = [("equi_s4", 4, 25, 25),  ("equi_s10", 10, 10, 10),  ("equi_s20", 20, 5, 5), 
              ("semi_s7", 7, 40, 10),  ("semi_s11", 11, 40, 6),  ("semi_s5", 5, 40, 15)] # ("semi_s3", 3, 50, 25),  ("semi_s6", 6, 50, 10),  ("semi_s11", 11, 50, 5)
# dataset list : random subsets and thematic subsets from ImageNet, each contains 100 classes with 400 images (train+test)
dataset_list = ["imagenet_random_0", "imagenet_random_1", "imagenet_random_2", 
                "imagenet_fauna", "imagenet_flora", "imagenet_food",]

# Network hyperparameters
net_hp_dic = {
    "resnet1" : {"backbone":"resnetBasic", "width_mult":0.5, "depth_mult":0.5, "lr":0.1},
    "resnet2" : {"backbone":"resnetBasic", "width_mult":0.6, "depth_mult":0.5, "lr":0.05},
    "resnet3" : {"backbone":"resnetBasic", "width_mult":1.0, "depth_mult":0.5, "lr":0.01},
    "mobilenet1" : {"backbone":"shufflenet", "width_mult":1.6, "depth_mult":0.1, "lr":0.1},
    "mobilenet2" : {"backbone":"shufflenet", "width_mult":2.6, "depth_mult":0.1, "lr":0.05},
    "mobilenet3" : {"backbone":"shufflenet", "width_mult":3.0, "depth_mult":0.1, "lr":0.01},
    "shufflenet1" : {"backbone":"mobilenet", "width_mult":1.0, "depth_mult":0.2, "lr":0.1},
    "shufflenet2" : {"backbone":"mobilenet", "width_mult":1.4, "depth_mult":0.2, "lr":0.05},
    "shufflenet3" : {"backbone":"mobilenet", "width_mult":2.2, "depth_mult":0.2, "lr":0.01},
}

### LUCIR ###
for dataset in dataset_list: 
    ### scenario ###
    for (prefix, last_batch_number, B, P) in state_list : 
        config_dic = {"dataset" : dataset, "last_batch_number":last_batch_number, "B":B, "P":P}
        i=0
        text=""
        ### networks ###
        for net in net_hp_dic.keys():
            config_dic_i = {**config_dic, **net_hp_dic[net]} # join dict
            config_dic_i["i"]=i
            print(config_dic_i)
            text += """expe{i}:
  backbone : {backbone} # base architecture
  width_mult : {width_mult} # width multiplier
  depth_mult : {depth_mult} # depth multiplier
  normalization_dataset_name : {dataset} # as written in the reference normalization file
  datasets_mean_std_file_path : "/home/users/efeillet/images_list_files/datasets_mean_std.txt"
  train_file_path : /home/users/efeillet/images_list_files/train100/{dataset}/train.lst # text file with first line = root_path, folder for dataset, trian100 means no val set
  test_file_path : /home/users/efeillet/images_list_files/train100/{dataset}/test.lst  
  B : {B} # number of classes in the first, non-incremental state
  P : {P} # number of classes per incremental state
  last_batch_number : {last_batch_number}
  IL_method : LUCIR  
  lr : {lr}
  first_epochs : 70 # scratch / first batch 
  subsequent_epochs : 50 # incr \n\n""".format(**config_dic_i)
            i+=1

        ## SAVE FILE
        file_path = os.path.join(destination_path, "lucir", dataset + "_lucir_"+prefix+".yaml")
        print("Creating new file with path %s" %file_path)
        text_file = open(file_path, "w")   
        #write string to file
        n = text_file.write(text)
        #close file
        text_file.close()
        print("Successful saving.\n")

### SPB-M ###
for dataset in dataset_list: 
    ### scenario ###
    for (prefix, last_batch_number, B, P) in state_list : 
        config_dic = {"dataset" : dataset, "last_batch_number":last_batch_number, "B":B, "P":P}
        i=0
        text=""
        ### networks ###
        for net in net_hp_dic.keys():
            config_dic_i = {**config_dic, **net_hp_dic[net]} # join dict
            config_dic_i["i"]=i
            print(config_dic_i)
            text += """expe{i}:
  backbone : {backbone} # base architecture
  width_mult : {width_mult} # width multiplier
  depth_mult : {depth_mult} # depth multiplier
  normalization_dataset_name : {dataset} # as written in the reference normalization file
  datasets_mean_std_file_path : "/home/users/efeillet/images_list_files/datasets_mean_std.txt"
  train_file_path : /home/users/efeillet/images_list_files/train100/{dataset}/train.lst # text file with first line = root_path, folder for dataset, trian100 means no val set
  test_file_path : /home/users/efeillet/images_list_files/train100/{dataset}/test.lst  
  B : {B} # number of classes in the first, non-incremental state
  P : {P} # number of classes per incremental state
  last_batch_number : {last_batch_number}
  IL_method : SPBM  
  lr : {lr}
  first_epochs : 70 # scratch / first batch 
  subsequent_epochs : 50 # incr \n\n""".format(**config_dic_i)
            i+=1

        ## SAVE FILE
        file_path = os.path.join(destination_path, "spbm", dataset + "_spbm_"+prefix+".yaml")
        print("Creating new file with path %s" %file_path)
        text_file = open(file_path, "w")   
        #write string to file
        n = text_file.write(text)
        #close file
        text_file.close()
        print("Successful saving.\n")

### SIW ###
for dataset in dataset_list: 
    ### scenario ###
    for (prefix, last_batch_number, B, P) in state_list : 
        config_dic = {"dataset" : dataset, "last_batch_number":last_batch_number, "B":B, "P":P}
        i=0
        text=""
        ### networks ###
        for net in net_hp_dic.keys():
            config_dic_i = {**config_dic, **net_hp_dic[net]} # join dict
            config_dic_i["i"]=i
            print(config_dic_i)
            text += """expe{i}:
  backbone : {backbone} # base architecture
  width_mult : {width_mult} # width multiplier
  depth_mult : {depth_mult} # depth multiplier
  normalization_dataset_name : {dataset} # as written in the reference normalization file
  datasets_mean_std_file_path : "/home/users/efeillet/images_list_files/datasets_mean_std.txt"
  train_file_path : /home/users/efeillet/images_list_files/train100/{dataset}/train.lst # text file with first line = root_path, folder for dataset, trian100 means no val set
  val_file_path : /home/users/efeillet/images_list_files/train100/{dataset}/test.lst  
  B : {B} # number of classes in the first, non-incremental state
  P : {P} # number of classes per incremental state
  last_batch_number : {last_batch_number}
  num_batches : {last_batch_number}
  IL_method : inFT_siw   
  algo_name : nomemft
  lr : {lr}
  first_epochs : 70 # scratch / first batch 
  subsequent_epochs : 50 # incremental states\n\n""".format(**config_dic_i)
            i+=1

        ## SAVE FILE
        file_path = os.path.join(destination_path, "siw", dataset + "_siw_"+prefix+".yaml")
        print("Creating new file with path %s" %file_path)
        text_file = open(file_path, "w")   
        #write string to file
        n = text_file.write(text)
        #close file
        text_file.close()
        print("Successful saving.\n")