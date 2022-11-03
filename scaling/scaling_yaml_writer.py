import os

destination_path = "/home/data/efeillet/expe/scaling/yaml_files"
#state_list = [("equi_s4", 4, 25, 25), ("equi_s10", 10, 10, 10), ("equi_s20", 20, 5, 5), ("semi_s3", 3, 50, 25), ("semi_s6", 6, 50, 10), ("semi_s11", 11, 50, 5)]
state_list = [("equi_s10", 10, 10, 10)]

for dataset in ["imagenet_random_0",  "imagenet_fauna"]: # "imagenet_flora", "imagenet_food", "imagenet_random_1", "imagenet_random_2",
    for (_, last_batch_number, B, P) in state_list : 
        config_dic = {"dataset" : dataset, "last_batch_number":last_batch_number, "B":B, "P":P}
        i=0
        text=""
        ### RESNET ###
        backbone = "resnetBasic"
        w_list = [0.1, 0.4, 0.6, 1.0]
        d_list = [0.5, 1.0]
        for w in w_list:
            for d in d_list : 
                config_dic["i"]=i
                config_dic["backbone"]=backbone
                config_dic["w"]=w
                config_dic["d"]=d
                text += """expe{i}:
  backbone : {backbone} # base architecture
  width_mult : {w} # width multiplier
  depth_mult : {d} # depth multiplier
  normalization_dataset_name : {dataset} # as written in the reference normalization file
  datasets_mean_std_file_path : "/home/users/efeillet/images_list_files/datasets_mean_std.txt"
  train_file_path : /home/users/efeillet/images_list_files/train100/{dataset}/train.lst # text file with first line = root_path, folder for dataset, trian100 means no val set
  test_file_path : /home/users/efeillet/images_list_files/train100/{dataset}/test.lst  
  B : {B} # number of classes in the first, non-incremental state
  P : {P} # number of classes per incremental state
  last_batch_number : {last_batch_number}
  IL_method : LUCIR  
  first_epochs : 70 # scratch / first batch 
  subsequent_epochs : 70 # incr \n\n""".format(**config_dic)
                i+=1
        ### SHUFFLENET ###
        backbone = "shufflenet"
        w_list = [0.1, 0.4, 0.6, 1.0, 1.6, 2.0, 2.6, 3.0]
        d_list = [0.1, 0.2, 0.4, 1.0]
        for w in w_list:
            for d in d_list : 
                config_dic["i"]=i
                config_dic["backbone"]=backbone
                config_dic["w"]=w
                config_dic["d"]=d
                text += """expe{i}:
  backbone : {backbone} # base architecture
  width_mult : {w} # width multiplier
  depth_mult : {d} # depth multiplier
  normalization_dataset_name : {dataset} # as written in the reference normalization file
  datasets_mean_std_file_path : "/home/users/efeillet/images_list_files/datasets_mean_std.txt"
  train_file_path : /home/users/efeillet/images_list_files/train100/{dataset}/train.lst # text file with first line = root_path, folder for dataset, trian100 means no val set
  test_file_path : /home/users/efeillet/images_list_files/train100/{dataset}/test.lst  
  B : {B} # number of classes in the first, non-incremental state
  P : {P} # number of classes per incremental state
  last_batch_number : {last_batch_number}
  IL_method : LUCIR  
  first_epochs : 70 # scratch / first batch 
  subsequent_epochs : 70 # incr \n\n""".format(**config_dic)  
                i+=1
        ### MOBILENET ###
        backbone = "mobilenet"
        w_list = [0.2, 0.6, 1.0, 1.4, 1.8, 2.2]
        d_list = [0.2, 0.6, 1.0]
        for w in w_list:
            for d in d_list : 
                config_dic["i"]=i
                config_dic["backbone"]=backbone
                config_dic["w"]=w
                config_dic["d"]=d
                text += """expe{i}:
  backbone : {backbone} # base architecture
  width_mult : {w} # width multiplier
  depth_mult : {d} # depth multiplier
  normalization_dataset_name : {dataset} # as written in the reference normalization file
  datasets_mean_std_file_path : "/home/users/efeillet/images_list_files/datasets_mean_std.txt"
  train_file_path : /home/users/efeillet/images_list_files/train100/{dataset}/train.lst # text file with first line = root_path, folder for dataset, trian100 means no val set
  test_file_path : /home/users/efeillet/images_list_files/train100/{dataset}/test.lst  
  B : {B} # number of classes in the first, non-incremental state
  P : {P} # number of classes per incremental state
  last_batch_number : {last_batch_number}
  IL_method : LUCIR  
  first_epochs : 70 # scratch / first batch 
  subsequent_epochs : 70 # incr \n\n""".format(**config_dic)  
                i+=1
        ## SAVE FILE
        file_path = os.path.join(destination_path, dataset + "_lucir_equi_s10_scaling.yaml")
        print("\nCreating new file with path %s" %file_path)
        text_file = open(file_path, "w")   
        #write string to file
        n = text_file.write(text)
        #close file
        text_file.close()
        print("Successful saving.")


