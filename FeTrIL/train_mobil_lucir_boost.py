import random
import sys
import os
from os.path import isfile, join
from sklearn.svm import LinearSVC
from sklearn.datasets import load_svmlight_file, dump_svmlight_file
from sklearn.preprocessing import normalize
import numpy as np
import pandas as pd
from numpy.linalg import norm
from multiprocessing import Pool, cpu_count, Array

from numpy.linalg import norm


def compute_dist_np(feat1, feat2):
    diff = feat1 - feat2
    feat_dist = np.dot(diff,diff)
    return feat_dist

def normalize_l2(v):
    '''
    L2-normalization of numpy arrays
    '''
    norm = np.linalg.norm(v)
    if norm == 0: 
        return v
    return v / norm

"""
python /home/users/apopescu/MobIL/train_mobil_lucir.py 1.0 0.0001 100 food101 50 5 /home/data/didi/ST/features/ /home/data/didi/ST/svms/


"""


#provide the following arguments to facilitate paralellization
regul = sys.argv[1] #value of the regularization parameter for the SVMs
toler = sys.argv[2] #value of the tolerance parameter for the SVMs
nb_classes = int(sys.argv[3]) #number of classes in TOTAL
dataset = sys.argv[4]
first_batch_size = int(sys.argv[5])
il_states = int(sys.argv[6]) #initialement on donnerait 3, ici on donne 4
feat_root = sys.argv[7]
svms_root = sys.argv[8]
state_id = int(sys.argv[9])
ratio = -1
s = il_states-1
incr_batch_size = (nb_classes-first_batch_size)//s

def normalize_train_features(il_dir,norm_type,state_id,state_size):
    feats_libsvm = []
    min_pos = 0
    max_pos = first_batch_size + (state_id) * state_size
    feat_dir = il_dir
    crt_pos = 0
    class_list = list(range(nb_classes))
    f_list = class_list
    for crt_id in f_list:
        if crt_pos >= min_pos and crt_pos < max_pos:
            class_feats = os.path.join(feat_dir,str(crt_id))
            f_feats = open(class_feats)
            for crt_feat in f_feats:
                crt_feat = crt_feat.rstrip()
                np_feat = np.fromstring(crt_feat,dtype=float,sep=' ')
                if norm_type == "l2":
                    np_feat_l2 = normalize_l2(np_feat)
                    libsvm_feat = str(crt_id)
                    for cdim in range(0,np_feat_l2.shape[0]):
                        libsvm_feat = libsvm_feat+" "+str(np_feat_l2[cdim]) 
                    feats_libsvm.append(libsvm_feat)
                else:
                    print("exiting - unknown normalization type:",norm_type)
                    sys.exit(0)
            f_feats.close()
        crt_pos = crt_pos+1
    
    return feats_libsvm
    
    
""" MAIN """
if __name__ == '__main__':
    norm_type = "l2"
    #il_dir = f"{feat_root}/mobil/{dataset}/b{first_batch_size}/s{il_states}/train/batch{state_id}"
    il_dir = os.path.join(feat_root,"mobil","b"+str(first_batch_size),"s"+str(il_states),"train","batch"+str(state_id))
    #svms_dir = f"{svms_root}/mobil/{dataset}/b{first_batch_size}/s{il_states}/batch{state_id}"
    svms_dir = os.path.join(svms_root,"mobil","b"+str(first_batch_size),"s"+str(il_states),"batch"+str(state_id))
    min_pos = 0
    max_pos = first_batch_size + (state_id) * incr_batch_size
    state_size = first_batch_size
    if state_id>0:
        state_size = incr_batch_size
    to_check = any([not os.path.exists(os.path.join(svms_dir,str(crt_id)+".model")) for crt_id in range(min_pos,max_pos)])
    if not to_check:
        print("SVM already created for range",[os.path.join(svms_dir,str(crt_id)+".model") for crt_id in [min_pos,max_pos-1]])
    else:
        norm_feats = normalize_train_features(il_dir,norm_type,state_id,state_size)
        df = pd.DataFrame([elt.split() for elt in norm_feats], columns=['classe']+['feat'+str(i+1) for i in range(-1+len(norm_feats[0].split(' ')))])
        y_true = df['classe'].to_numpy()
        X = df.drop(columns=['classe']).to_numpy(dtype=float)
        os.makedirs(svms_dir, exist_ok=True)

        def calc_thrd(crt_id):
            crt_id = str(crt_id)
            print("training: ",crt_id, "; STATE",state_id)
            crt_id_svm_path = os.path.join(svms_dir,crt_id+".model")
            if (not os.path.exists(crt_id_svm_path)):
                y = np.empty(y_true.shape, dtype = str)
                y[y_true==crt_id]='+1'
                y[y_true!=crt_id]='-1'
                if ratio>0:
                    idx_to_take = np.random.choice(len(y[y_true!=crt_id]), size=min(len(y[y_true!=crt_id]),int(ratio)*len(y[y_true==crt_id])), replace=False)
                    X_crt = np.concatenate((X[y_true==crt_id],X[y_true!=crt_id][idx_to_take]))
                    y_crt = np.concatenate((y[y_true==crt_id],y[y_true!=crt_id][idx_to_take]))
                    clf = LinearSVC(penalty='l2', dual=False, tol=float(toler), C=float(regul), multi_class='ovr', fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, random_state=123)
                    clf.fit(X_crt,y_crt)
                else:
                    clf = LinearSVC(penalty='l2', dual=False, tol=float(toler), C=float(regul), multi_class='ovr', fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, random_state=123)
                    clf.fit(X,y)
                #print(f'{crt_id} is trained :)')
                svm_weights = clf.coef_
                svm_bias = clf.intercept_
                out_weights = ""
                for it in range(0, svm_weights.size):
                    out_weights = out_weights+" "+str(svm_weights.item(it))
                out_weights = out_weights.lstrip()
                out_bias = str(svm_bias.item(0))
                with open(crt_id_svm_path,"w") as f_svm:
                    f_svm.write(out_weights+"\n") 
                    f_svm.write(out_bias+"\n")
            else:
                print("SVM already created for:",crt_id_svm_path)
        range_tro_koul = list(range(min_pos,max_pos))
        random.shuffle(range_tro_koul)
        #with Pool(72//int(il_states)) as p:
        with Pool(10) as p:
            p.map(calc_thrd, range_tro_koul)


    
