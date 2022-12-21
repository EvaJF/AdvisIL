import sys
from os.path import isfile, isdir, join
from os import listdir
import numpy as np
from scipy.stats import spearmanr
import itertools
import pprint 
pp = pprint.PrettyPrinter(indent=4)


"""
compares the performance of:
-individual methods - using the same method+backbone combination for all incremental configurations 
-oracle - selecting the best method+backbong combination for each incremental configuration
USAGE EXAMPLE:
python3 advisil_reco_vote.py reco/ref_configs.csv reco/test_configs.csv > advisil_reco_vote.log
"""

ref_log = sys.argv[1]
test_log = sys.argv[2]


def compute_scenarios_L2_similarities(ref_log, test_log):
    """
    Computes L2 distances between scenarios from the reference and the test configurations. Min-max normalization applied to each dimension of the scenarios.
    Scenario : (memory budget in nbr of params, nbr of steps in the incremental process, nbr of initial classes alpha, nbr of classes per incremental step beta) 

    Input:
    ------
    ref_log : csv. parsed results corresponding to reference configurations.
    test_log : csv. parsed results corresponding to test configurations.

    Output:
    -------
    sim_dict : dict. To each test scenario, associates its closest reference scenario according to the L2 distance.
    """
    sim_dict = {}
    #list to store all values of the parameters used to compute the similarity between test and reference scenarios
    k_list = []
    alpha_list = []
    beta_list = []
    size_list = []
    max_list = []
    min_list = []
    ref_dict = {} #dictionary for reference configurations
    f_log = open(ref_log)
    for line in f_log:
        parts = line.rstrip().split(",")
        if isfloat(parts[13]):
            #form an entry for the current budget
            k_list.append(float(parts[9]))
            alpha_list.append(float(parts[10]))
            beta_list.append(float(parts[11]))
            size_list.append(float(parts[12]))
            crt_config = parts[9]+" "+parts[10]+" "+parts[11]+" "+parts[12]
            ref_dict[crt_config] = ""
    test_dict = {} #dictionary for reference configurations
    f_log = open(test_log)
    for line in f_log:
        parts = line.rstrip().split(",")
        if isfloat(parts[13]):
            k_list.append(float(parts[9]))
            alpha_list.append(float(parts[10]))
            beta_list.append(float(parts[11]))
            size_list.append(float(parts[12]))
            #form an entry for the current budget
            crt_config = parts[9]+" "+parts[10]+" "+parts[11]+" "+parts[12]
            if not crt_config in ref_dict:
                test_dict[crt_config] = ""
    #get the max and min value of each parameter
    k_max = np.max(np.asarray(k_list))
    k_min = np.min(np.asarray(k_list))
    alpha_max = np.max(np.asarray(alpha_list))
    alpha_min = np.min(np.asarray(alpha_list))
    beta_max = np.max(np.asarray(beta_list))
    beta_min = np.min(np.asarray(beta_list))
    size_max = np.max(np.asarray(size_list))
    size_min = np.min(np.asarray(size_list))    
    max_list = [k_max,alpha_max,beta_max,size_max]
    min_list = [k_min,alpha_min,beta_min,size_min]
    print("mins :",k_min,alpha_min,beta_min,size_min)
    print("maxes:",k_max,alpha_max,beta_max,size_max)
    #compute the similarities between test and reference scenarios
    for test_cf in test_dict:
        np_test = np.fromstring(test_cf,dtype=float,sep=' ')
        norm_test = []
        for pos in range(0,len(max_list)):
            crt_norm = (max_list[pos]-np_test[pos])/(max_list[pos]-min_list[pos])
            norm_test.append(crt_norm)
        tmp_ref = {} #tmp dict for the reference scenarios
        for ref_cf in ref_dict:
            np_ref = np.fromstring(ref_cf,dtype=float,sep=' ')
            norm_ref = []
            for pos in range(0,len(max_list)):
                crt_norm = (max_list[pos]-np_ref[pos])/(max_list[pos]-min_list[pos])
                norm_ref.append(crt_norm)
            #compute the L2 distance between the two scenarios
            crt_l2 = np.linalg.norm(np.asarray(norm_test)-np.asarray(norm_ref))
            tmp_ref[ref_cf] = crt_l2
        ref_sorted = sorted(tmp_ref, key=tmp_ref.get, reverse=False)
        sim_dict[test_cf] = ref_sorted[0]
    return sim_dict

  
def compute_votes(ref_log, test_log, sim_dict):
    """
    Computes AdvisIL recommendations according to a ranking-based vote.


    Input:
    ------
    ref_log : csv. parsed results corresponding to reference configurations.
    test_log : csv. parsed results corresponding to test configurations.
    sim_dict : dict. To each test scenario, associates its closest reference scenario according to the L2 distance.

    Output:
    -------
    None (refer to the printed logs)
    """
    
    #get the recommended combinations for AdvisIL and the fixed methods
    advisil_dict = {} #dictionary used to recommend algorithms using a simple rank-based voting system
    advisil_backbone_dict = {} #dictionary to store advisil recommendations per backbone
    advisil_algorithm_dict = {} #dictionary to store advisil recommendations per algorithm
    advisil_restricted_dict = {} #dictionary used to test only a subset of methods for advisil

    fixed_ranked = []     
    backbones = ['resnetBasic', 'shufflenet','mobilenet']
    best_algos = ['mobil', 'slda'] # 'spbm',
    restricted_algos = ['deesil', 'spbm', 'lucir', 'siw'] 
    ref_datasets = ["imagenet_fauna", "imagenet_flora", "imagenet_food", "imagenet_random_0", "imagenet_random_1"]
    #initialize reco dict per backbone
    for back in backbones:
        advisil_backbone_dict[back] = {}
    #initialize reco dict for best two algorithms
    for algo in best_algos:
        advisil_algorithm_dict[algo] = {}

    tmp_dict = {} #tmp dictonary which will be used to store the accuracy of the different algo+backbone combination
    comb_dict = {} #dictionary to store all algorithm+backbone combinations
    f_log = open(ref_log)
    for line in f_log:
        parts = line.rstrip().split(",")
        if isfloat(parts[13]):
            #form an entry for the current budget
            crt_config = parts[9]+" "+parts[10]+" "+parts[11]+" "+parts[12]
            crt_dataset = parts[3]
            #create a dictionary entry for the current user config
            if not crt_config in tmp_dict:
                tmp_dict[crt_config] = {}
            #create an entry for each dataset for the current configuration
            if not crt_dataset in tmp_dict[crt_config]:
                tmp_dict[crt_config][crt_dataset] = []
            tmp_dict[crt_config][crt_dataset].append((parts[2],parts[5],parts[13]))
            comb = parts[2]+" "+parts[5]
            if not comb in comb_dict:
                comb_dict[comb] = []
    f_log.close()

    ###create a dictionary which recommends an algorithm+backbone per configuration based on a voting system
    for config in tmp_dict:
        print("\n*** Scenario ***", config)
        tmp_config = {} #dictionary to store the ranks of the methods for each particular configuration
        #print(config)
        for dataset in tmp_dict[config]:
            #print(dataset)
            #print(tmp_dict[config][dataset],"\n")
            #rank the algo+backbone combinations for the config+dataset
            tmp_ranks = {}
            for triplet in tmp_dict[config][dataset]:
                comb = triplet[0]+" "+triplet[1]
                tmp_ranks[comb] = triplet[2]
                #print(" ",comb,triplet[2])
            comb_ranked = sorted(tmp_ranks, key=tmp_ranks.get, reverse=True)            
            for rank in range(0,len(comb_ranked)):
                #print ("  ",comb_ranked[rank],tmp_ranks[comb_ranked[rank]])
                crt_comb = comb_ranked[rank]
                crt_rank = rank+1
                comb_dict[crt_comb].append(crt_rank)
                #print(" ",crt_comb,crt_rank)
                if not crt_comb in tmp_config:
                    tmp_config[crt_comb] = []
                tmp_config[crt_comb].append(crt_rank)
        
        #compute the mean ranks of the combinations for the current configuration
        config_means = {}        
        for comb in tmp_config:
           comb_mean = np.mean(np.asarray(tmp_config[comb])) 
           config_means[comb] = comb_mean
        config_ranked = sorted(config_means, key=config_means.get, reverse=False)
        advisil_dict[config] = config_ranked[0]
            
        #compute advisil recommendations for each backbone
        for back in backbones:
            config_means = {}
            for comb in tmp_config:
                if back in comb:
                    comb_mean = np.mean(np.asarray(tmp_config[comb])) 
                    config_means[comb] = comb_mean
            config_ranked = sorted(config_means, key=config_means.get, reverse=False)
            advisil_backbone_dict[back][config] = config_ranked[0]
        
        #compute advisil recommendations using the best 2 methods (fetril, dslda)
        #print("best algos", best_algos)
        for algo in best_algos:
            print("\nALGO - ", algo)
            config_means = {}
            for comb in tmp_config:
                #print("comb : ", comb)
                comb_parts = comb.split(" ")
                if algo in comb_parts[0]:
                    comb_mean = np.mean(np.asarray(tmp_config[comb])) 
                    config_means[comb] = comb_mean
            config_ranked = sorted(config_means, key=config_means.get, reverse=False)
            print("reco (1)", config, ' --> ', config_ranked[0])
            advisil_algorithm_dict[algo][config] = config_ranked[0]

        #compute advisil recommendations for a subset of algorithms (here removed the top 2 FeTrIL and DeepSLDA from the algorithm list)
        print("\nOther algos - ", restricted_algos)
        config_means = {}
        for comb in tmp_config:
            #print("comb : ", comb)
            comb_parts = comb.split(" ")
            if comb_parts[0] in restricted_algos:
                comb_mean = np.mean(np.asarray(tmp_config[comb])) 
                config_means[comb] = comb_mean
        config_ranked = sorted(config_means, key=config_means.get, reverse=False)
        print("reco (2)", config, ' --> ', config_ranked[0])
        advisil_restricted_dict[config] = config_ranked[0]

    #rank the combination depending on their ranking
    comb_tmp_ranks = {}
    for comb in comb_dict:
        comb_mean = np.mean(np.asarray(comb_dict[comb]))
        comb_tmp_ranks[comb] = comb_mean

    fixed_comb = sorted(comb_tmp_ranks, key=comb_tmp_ranks.get, reverse=False)
    fixed_back_ranked = []
    #compute the performance vs. the best fixed combination of each algorithm
    unique_algo = {}
    unique_back = {}
    print("\n\nPerformance gain vs best fixed algo+backbone")
    for pos in range(0,len(fixed_comb)):
        comb_parts = fixed_comb[pos].split(" ")
        #print(comb_parts[0])
        if not comb_parts[0] in unique_algo:
            unique_algo[comb_parts[0]] = ""
            fixed_ranked.append(fixed_comb[pos])
            print(pos,fixed_comb[pos],comb_tmp_ranks[fixed_comb[pos]])
        if not comb_parts[1] in unique_back:
            unique_back[comb_parts[1]] = ""
            fixed_back_ranked.append(fixed_comb[pos])
    
    print("\nAdvisIL dict")
    pp.pprint(advisil_dict)
    print("\nfixed_ranked")
    pp.pprint(fixed_ranked)
    print("\nfixed_comb") 
    pp.pprint(fixed_comb)

    test_cont = {}  #dict to store the content of test configs
    test_dataset = {} #dict to store the content of test configs at dataset level
    f_log_test = open(test_log)
    for line in f_log_test:
        #print(line)
        parts = line.rstrip().split(",")
        if isfloat(parts[13]):
            #form an entry for the current budget
            crt_config = parts[9]+" "+parts[10]+" "+parts[11]+" "+parts[12]
            crt_dataset = parts[3]
            crt_comb = parts[2]+" "+parts[5]
            crt_accu = float(parts[13])
            #make sure that reference configurations are excluded
            if not crt_config in tmp_dict:
                if not crt_config in test_cont:
                    test_cont[crt_config] = {}
                    test_dataset[crt_config] = {}
                if not crt_comb in test_cont[crt_config]:
                    test_cont[crt_config][crt_comb] = []
                if not crt_dataset in test_dataset[crt_config]:
                    test_dataset[crt_config][crt_dataset] = {}
                
                test_cont[crt_config][crt_comb].append(crt_accu)
                test_dataset[crt_config][crt_dataset][crt_comb] = crt_accu
    f_log_test.close()

    print("\n\nBEST FIXED")
    #compute the accuracy for the top ranked fixed methods
    for fixed in fixed_ranked:
        fixed_accu = []
        for config in test_cont:
            #compute the mean accuracy of the method for the current config over all datasets
            conf_mean_accu = np.mean(np.asarray(test_cont[config][fixed]))
            fixed_accu.append(conf_mean_accu)
        mean_fixed_accu = np.mean(np.asarray(fixed_accu))
        print(fixed,", acc@1 = ",mean_fixed_accu)

    #compute the advisil accuracy by using a voting system for all datasets associated to a scenario
    advisil_accu = []
    for config in test_cont:
        sim_ref = sim_dict[config] #get the similar configuration from the reference catalog
        advisil_comb = advisil_dict[sim_ref]
        print(config,",",advisil_comb)
        conf_mean_accu = np.mean(np.asarray(test_cont[config][advisil_comb]))
        advisil_accu.append(conf_mean_accu)
    mean_advisil_accu = np.mean(advisil_accu)
    print("\nAdvisIL simple vote, acc@1 = ", mean_advisil_accu)

    ### compute the accuracy of the oracle for the test configurations
    oracle_data_accu = []
    for config in test_dataset:
        #get the best accuracy for the configuration (oracle)
        for dset in test_dataset[config]:
            #get the number of datasets
            max_cf_dset = 0
            for comb in test_dataset[config][dset]:
                if test_dataset[config][dset][comb] > max_cf_dset:
                    max_cf_dset = test_dataset[config][dset][comb]
            oracle_data_accu.append(max_cf_dset)
    mean_data_oracle_accu = np.mean(oracle_data_accu)
    print("\nOracle, acc@1 = ",mean_data_oracle_accu, len(oracle_data_accu))

    #compute the average accuracy for each of the six test scenarios

    ################################### PER CLASS REPARTITION (TAB 2) ##############################################
    if 1 == 0:
        scenarios_list = scenarios_list = ["2 2 50", "4 4 25", "20 20 5", "40 5 13", "50 5 11", "50 10 6"]
        print ("\n***detailed results per scenario***")
        top3_methods = ["mobil resnetBasic", "slda shufflenet", "spbm mobilenet"]
        dataset_fixed_accu = {}
        for target_scen in scenarios_list:
            print("\n SCENARIO:",target_scen)
            #compute the accuracy for the top ranked fixed methods
            for fixed in fixed_ranked:
                fixed_accu = []
                for config in test_cont:
                    cf_parts = config.split(" ")
                    crt_scen = cf_parts[0]+" "+cf_parts[1]+" "+cf_parts[2]
                    if crt_scen == target_scen:
                        #compute the mean accuracy of the method for the current config over all datasets
                        conf_mean_accu = np.mean(np.asarray(test_cont[config][fixed]))
                        fixed_accu.append(conf_mean_accu)
                mean_fixed_accu = np.mean(np.asarray(fixed_accu))
                print(fixed,", acc@1 = ",mean_fixed_accu)
                dataset_fixed_accu[fixed] = mean_fixed_accu

            #compute the advisil accuracy by using a voting system for all datasets associated to a scenario
            advisil_accu = []
            for config in test_cont:
                cf_parts = config.split(" ")
                crt_scen = cf_parts[0]+" "+cf_parts[1]+" "+cf_parts[2]
                if crt_scen == target_scen:
                    sim_ref = sim_dict[config] #get the similar configuration from the reference catalog
                    advisil_comb = advisil_dict[sim_ref]
                    print(config,",",advisil_comb)
                    conf_mean_accu = np.mean(np.asarray(test_cont[config][advisil_comb]))
                    advisil_accu.append(conf_mean_accu)
            mean_advisil_accu = np.mean(advisil_accu)
            
            print("\nadvisil, acc@1 = ",mean_advisil_accu)
            for fixed in top3_methods:
                delta_advisil = mean_advisil_accu-dataset_fixed_accu[fixed]
                print("  delta",fixed,delta_advisil)

            #compute the accuracy of the oracle for the test configurations
            oracle_data_accu = []
            for config in test_dataset:
                cf_parts = config.split(" ")
                crt_scen = cf_parts[0]+" "+cf_parts[1]+" "+cf_parts[2]
                #get the best accuracy for the configuration (oracle)
                if crt_scen == target_scen:
                    for dset in test_dataset[config]:
                        #get the number of datasets
                        max_cf_dset = 0
                        for comb in test_dataset[config][dset]:
                            if test_dataset[config][dset][comb] > max_cf_dset:
                                max_cf_dset = test_dataset[config][dset][comb]
                        oracle_data_accu.append(max_cf_dset)
            mean_data_oracle_accu = np.mean(oracle_data_accu)
            delta_oracle = mean_advisil_accu-mean_data_oracle_accu
            print("\noracle, acc@1 = ",mean_data_oracle_accu, len(oracle_data_accu),", delta:",delta_oracle)   
    
    ################################### PER MEMORY BUDGET (TAB 2) ###############################################
    if 1 == 0:
        memory_list = ["1500000.0", "3000000.0", "6000000.0"]
        top3_methods = ["mobil resnetBasic", "slda shufflenet", "spbm mobilenet"]
        print ("\n***detailed results per memory budget***")
        dataset_fixed_accu = {}
        for target_mem in memory_list:
            print("\n MEMORY:",target_mem)
            #compute the accuracy for the top ranked fixed methods
            for fixed in fixed_ranked:
                fixed_accu = []
                for config in test_cont:
                    cf_parts = config.split(" ")
                    crt_mem = cf_parts[3]
                    if crt_mem == target_mem:
                        print(config)
                        #compute the mean accuracy of the method for the current config over all datasets
                        conf_mean_accu = np.mean(np.asarray(test_cont[config][fixed]))
                        fixed_accu.append(conf_mean_accu)
                mean_fixed_accu = np.mean(np.asarray(fixed_accu))
                print(fixed,", acc@1 = ",mean_fixed_accu)
                dataset_fixed_accu[fixed] = mean_fixed_accu

            #compute the advisil accuracy by using a voting system for all datasets associated to a scenario
            advisil_accu = []
            for config in test_cont:
                cf_parts = config.split(" ")
                crt_mem = cf_parts[3]
                if crt_mem == target_mem:
                    sim_ref = sim_dict[config] #get the similar configuration from the reference catalog
                    advisil_comb = advisil_dict[sim_ref]
                    print(config,",",advisil_comb)
                    conf_mean_accu = np.mean(np.asarray(test_cont[config][advisil_comb]))
                    advisil_accu.append(conf_mean_accu)
            mean_advisil_accu = np.mean(advisil_accu)
            
            print("\nadvisil, acc@1 = ",mean_advisil_accu)
            for fixed in top3_methods:
                delta_advisil = mean_advisil_accu-dataset_fixed_accu[fixed]
                print("  delta",fixed,delta_advisil)

            #compute the accuracy of the oracle for the test configurations
            oracle_data_accu = []
            for config in test_dataset:
                cf_parts = config.split(" ")
                crt_mem = cf_parts[3]
                #get the best accuracy for the configuration (oracle)
                if crt_mem == target_mem:
                    for dset in test_dataset[config]:
                        #get the number of datasets
                        max_cf_dset = 0
                        for comb in test_dataset[config][dset]:
                            if test_dataset[config][dset][comb] > max_cf_dset:
                                max_cf_dset = test_dataset[config][dset][comb]
                        oracle_data_accu.append(max_cf_dset)
            mean_data_oracle_accu = np.mean(oracle_data_accu)
            delta_oracle = mean_advisil_accu-mean_data_oracle_accu
            print("\noracle, acc@1 = ",mean_data_oracle_accu, len(oracle_data_accu),", delta:",delta_oracle)          

    ################################### PER DATASET (TAB 2) ###############################################
    if 1 == 0:
        print ("\n***detailed results per dataset***")
        datasets_list = ["imagenet_random_2", "inat", "food101","google_landmarks"]
        top3_methods = ["mobil resnetBasic", "slda shufflenet", "spbm mobilenet"]
        dataset_fixed_accu = {}
        for target_dset in datasets_list:
            print("\n DATASET:",target_dset)
            #compute the accuracy for the top ranked fixed methods
            for fixed in fixed_ranked:
                fixed_accu = []
                for config in test_cont:
                    conf_mean_accu = test_dataset[config][target_dset][fixed]
                    fixed_accu.append(conf_mean_accu)
                mean_fixed_accu = np.mean(np.asarray(fixed_accu))
                print(fixed,", acc@1 = ",mean_fixed_accu)
                dataset_fixed_accu[fixed] = mean_fixed_accu

            #compute the advisil accuracy by using a voting system for all datasets associated to a scenario
            advisil_accu = []
            for config in test_dataset:
                sim_ref = sim_dict[config] #get the similar configuration from the reference catalog
                advisil_comb = advisil_dict[sim_ref]
                print(config,",",advisil_comb)
                conf_mean_accu = test_dataset[config][target_dset][advisil_comb]
                advisil_accu.append(conf_mean_accu)
            mean_advisil_accu = np.mean(advisil_accu)
            
            print("\nadvisil, acc@1 = ",mean_advisil_accu)
            for fixed in top3_methods:
                delta_advisil = mean_advisil_accu-dataset_fixed_accu[fixed]
                print("  delta",fixed,delta_advisil)

            #compute the accuracy of the oracle for the test configurations
            oracle_data_accu = []
            for config in test_dataset:
                max_cf_dset = 0
                for comb in test_dataset[config][target_dset]:
                    if test_dataset[config][target_dset][comb] > max_cf_dset:
                        max_cf_dset = test_dataset[config][target_dset][comb]
                oracle_data_accu.append(max_cf_dset)
            mean_data_oracle_accu = np.mean(oracle_data_accu)
            delta_oracle = mean_advisil_accu-mean_data_oracle_accu
            print("\noracle, acc@1 = ",mean_data_oracle_accu, len(oracle_data_accu),", delta:",delta_oracle)          

    ################################### BLOCK TO FILL IN THE DETAILED (TAB 3) ###############################################
    if 1 == 0:
        memory_list = ["1500000.0", "3000000.0", "6000000.0"]
        scenarios_list = ["2 2 50", "4 4 25", "20 20 5", "40 5 13", "50 5 11", "50 10 6"]
        top3_methods = ["spbm mobilenet", "slda shufflenet", "mobil resnetBasic"]
        for mem in memory_list:
            print("MEMORY",mem)
            for fixed in top3_methods:
                meth_row = fixed
                fixed_accu = []
                for scen in scenarios_list:
                    config = scen+" "+mem                    
                    conf_mean_accu = np.mean(np.asarray(test_cont[config][fixed]))
                    fixed_accu.append(conf_mean_accu)
                    meth_row = meth_row+" & "+"{:.2f}".format(conf_mean_accu)
                mean_fixed_accu = np.mean(np.asarray(fixed_accu))
                meth_row = meth_row +" & "+"{:.2f}".format(mean_fixed_accu)+" \\\\"
                print(meth_row) 

        for mem in memory_list:
            advisil_accu = []
            adv_row = "advisil"
            for scen in scenarios_list:
                config = scen+" "+mem 
                sim_ref = sim_dict[config] #get the similar configuration from the reference catalog
                advisil_comb = advisil_dict[sim_ref]
                print(config,",",advisil_comb)
                conf_mean_accu = np.mean(np.asarray(test_cont[config][advisil_comb]))
                adv_row = adv_row+" & "+"{:.2f}".format(conf_mean_accu)
                advisil_accu.append(conf_mean_accu)
            mean_advisil_accu = np.mean(advisil_accu)
            adv_row = adv_row +" & "+"{:.2f}".format(mean_advisil_accu)+" \\\\"
            print(adv_row)

    ################################### ADVISIL WITH FIXED BACKBONE (ablation) #######
    if 1 == 0:
        # print accuracy of best fixed method per backbone
        print("\nBEST BACKBONE FIXED")
        #compute the accuracy for the top ranked fixed methods
        for fixed in fixed_back_ranked:
            fixed_accu = []
            for config in test_cont:
                #compute the mean accuracy of the method for the current config over all datasets
                conf_mean_accu = np.mean(np.asarray(test_cont[config][fixed]))
                fixed_accu.append(conf_mean_accu)
            mean_fixed_accu = np.mean(np.asarray(fixed_accu))
            print(fixed,", acc@1 = ",mean_fixed_accu)

        #compute the advisil accuracy by using a voting system for all datasets associated to a scenario
        for back in backbones:
            advisil_accu = []
            for config in test_cont:
                sim_ref = sim_dict[config] #get the similar configuration from the reference catalog
                advisil_comb = advisil_backbone_dict[back][sim_ref]
                #print(config,",",advisil_comb)
                conf_mean_accu = np.mean(np.asarray(test_cont[config][advisil_comb]))
                advisil_accu.append(conf_mean_accu)
            mean_advisil_accu = np.mean(advisil_accu)
            print("fixed backbone",back," advisil, acc@1 = ",mean_advisil_accu)

    ################################### ADVISIL WITH FIXED BEST ALGORITHMS (ablation) #########
    if 1 == 0:
        print("\nBEST ALGORITHM FIXED")
        #compute the advisil accuracy by using a voting system for all datasets associated to a scenario
        for algo in best_algos:
            advisil_accu = []
            for config in test_cont:
                sim_ref = sim_dict[config] #get the similar configuration from the reference catalog
                advisil_comb = advisil_algorithm_dict[algo][sim_ref]
                #print(config,",",advisil_comb)
                conf_mean_accu = np.mean(np.asarray(test_cont[config][advisil_comb]))
                advisil_accu.append(conf_mean_accu)
            mean_advisil_accu = np.mean(advisil_accu)
            print("fixed algo",algo," advisil, acc@1 = ",mean_advisil_accu)

   

def compute_votes_dataset_ablation(ref_log,test_log,sim_dict):
    #get the recommended combinations for advisil and the fixed methods
    advisil_dict = {} #dictionary used to recommend algorithms using a simple rank-based voting system
    advisil_backbone_dict = {} #dictionary to store advisil recommendations per backbone
    advisil_algorithm_dict = {} #dictionary to store advisil recommendations per algorithm

    backbones = ['resnetBasic', 'shufflenet','mobilenet']
    best_algos = ['mobil', 'slda', 'spbm']
    ref_datasets = ["imagenet_fauna", "imagenet_flora", "imagenet_food", "imagenet_random_0", "imagenet_random_1"]
    advisil_means = [] #store the advisil performance for different size of the set of reference datasets
    #compute advisil performance with a variable number of datasets
    for num_dset in range(1, 6):
        print("number of subsets:",num_dset)
        num_advisil = [] #list to store the performance of advisil for different sizes of the reference set
        #get the unique subsets of the current size
        num_subsets = findsubsets(ref_datasets,num_dset)
        for subset in num_subsets:
            #initialize the advisil dict per backbone
            for back in backbones:
                advisil_backbone_dict[back] = {}
            #initialize the advisil dict for best two algorithms
            for algo in best_algos:
                advisil_algorithm_dict[algo] = {}

            tmp_dict = {} #tmp dictonary which will be used to store the accuracy of the different algo+backbone combination
            comb_dict = {} #dictionary to store all algorithm+backbone combinations
            f_log = open(ref_log)
            for line in f_log:
                parts = line.rstrip().split(",")
                if isfloat(parts[13]):
                    #form an entry for the current budget
                    crt_config = parts[9]+" "+parts[10]+" "+parts[11]+" "+parts[12]
                    crt_dataset = parts[3]
                    if crt_dataset in subset:
                        #create a dictionary entry for the current user config
                        if not crt_config in tmp_dict:
                            tmp_dict[crt_config] = {}
                        #create an entry for each dataset for the current configuration
                        if not crt_dataset in tmp_dict[crt_config]:
                            tmp_dict[crt_config][crt_dataset] = []
                        tmp_dict[crt_config][crt_dataset].append((parts[2],parts[5],parts[13]))
                        comb = parts[2]+" "+parts[5]
                        if not comb in comb_dict:
                            comb_dict[comb] = []
            f_log.close()

            #create a dictionary which recommends an algorithm+backbone per configuration based a voting system
            for config in tmp_dict:
                #print("*** CONFIG ***", config)
                tmp_config = {} #dictionary to store the ranks of the methods for each particular configuration
                #print(config)
                for dataset in tmp_dict[config]:
                    #print(dataset)
                    #print(tmp_dict[config][dataset],"\n")
                    #rank the algo+backbone combinations for the config+dataset
                    tmp_ranks = {}
                    for triplet in tmp_dict[config][dataset]:
                        comb = triplet[0]+" "+triplet[1]
                        tmp_ranks[comb] = triplet[2]
                        #print(" ",comb,triplet[2])
                    comb_ranked = sorted(tmp_ranks, key=tmp_ranks.get, reverse=True)            
                    for rank in range(0,len(comb_ranked)):
                        #print ("  ",comb_ranked[rank],tmp_ranks[comb_ranked[rank]])
                        crt_comb = comb_ranked[rank]
                        crt_rank = rank+1
                        comb_dict[crt_comb].append(crt_rank)
                        #print(" ",crt_comb,crt_rank)
                        if not crt_comb in tmp_config:
                            tmp_config[crt_comb] = []
                        tmp_config[crt_comb].append(crt_rank)
                #compute the mean ranks of the combinations for the current configuration
                config_means = {}        
                for comb in tmp_config:
                    comb_mean = np.mean(np.asarray(tmp_config[comb])) 
                    config_means[comb] = comb_mean
                    config_ranked = sorted(config_means, key=config_means.get, reverse=False)
                    advisil_dict[config] = config_ranked[0]


            test_cont = {}  #dict to store the content of test configs
            test_dataset = {} #dict to store the content of test configs at dataset level
            f_log_test = open(test_log)
            for line in f_log_test:
                parts = line.rstrip().split(",")
                if isfloat(parts[13]):
                    #form an entry for the current budget
                    crt_config = parts[9]+" "+parts[10]+" "+parts[11]+" "+parts[12]
                    crt_dataset = parts[3]
                    crt_comb = parts[2]+" "+parts[5]
                    crt_accu = float(parts[13])
                    #make sure that reference configurations are excluded
                    if not crt_config in tmp_dict:
                        if not crt_config in test_cont:
                            test_cont[crt_config] = {}
                            test_dataset[crt_config] = {}
                        if not crt_comb in test_cont[crt_config]:
                            test_cont[crt_config][crt_comb] = []
                        if not crt_dataset in test_dataset[crt_config]:
                            test_dataset[crt_config][crt_dataset] = {}
                        
                        test_cont[crt_config][crt_comb].append(crt_accu)
                        test_dataset[crt_config][crt_dataset][crt_comb] = crt_accu
            f_log_test.close()

            #compute the advisil accuracy by using a voting system for all datasets associated to a scenario
            advisil_accu = []
            for config in test_cont:
                sim_ref = sim_dict[config] #get the similar configuration from the reference catalog
                advisil_comb = advisil_dict[sim_ref]
                #print(config,",",advisil_comb)
                conf_mean_accu = np.mean(np.asarray(test_cont[config][advisil_comb]))
                advisil_accu.append(conf_mean_accu)
            mean_advisil_accu = np.mean(advisil_accu)
            #print("\nadvisil simple vote, acc@1 = ",mean_advisil_accu)
            num_advisil.append(mean_advisil_accu)
        num_mean_advisil = np.mean(num_advisil)
        print("ref datasets, acc@1:",num_dset,num_mean_advisil)
        advisil_means.append((num_dset,num_mean_advisil))
    
    #compute the delta between current advisil and full advisil
    for el in range(0,len(advisil_means)):
        crt_delta = advisil_means[el][1] - advisil_means[-1][1]
        print("ref datasets, crt delta: ",advisil_means[el][0],crt_delta)

def findsubsets(s, n):
    return list(itertools.combinations(s, n))

def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False

if __name__ == '__main__':   
    #get the similarities between scenarios
    print("Precomputing L2 distances between reference and test scenarios...")
    sim_dict = compute_scenarios_L2_similarities(ref_log, test_log)
    pp.pprint(sim_dict)
    print("\nComputing recommendations according to a rank-based vote")
    compute_votes(ref_log,test_log, sim_dict)
    print("\nAblation study --> number of reference datasets")
    compute_votes_dataset_ablation(ref_log,test_log,sim_dict)