import sys
import os
from os import listdir
from os.path import isfile, join
from sklearn.datasets import load_svmlight_file, dump_svmlight_file
import numpy as np
from numpy.linalg import norm
from sklearn.preprocessing import Normalizer
from multiprocessing import Pool


""" list of arguments for the script """
nb_classes = int(sys.argv[1]) #number of classes in TOTAL
dataset = sys.argv[2]
first_batch_size = int(sys.argv[3])
il_states = int(sys.argv[4]) #initialement on donnerait 3, ici on donne 4
feat_root = sys.argv[5]
svms_root = sys.argv[6]
pred_root = sys.argv[7]
s= il_states-1 #(pour Eva :) )

crt_b = str(first_batch_size)

test_feats_path = os.path.join(feat_root,"b"+crt_b,"test/")

#svms_dir = f"{svms_root}/mobil/{dataset}/b{first_batch_size}/s{il_states}/"
svms_dir = os.path.join(svms_root,"mobil","b"+str(first_batch_size),"s"+str(il_states))
#pred_path = f"{pred_root}/mobil/{dataset}/b{first_batch_size}/s{il_states}/"
pred_path = os.path.join(pred_root,"mobil","b"+str(first_batch_size),"s"+str(il_states))
model_dir = svms_dir
os.makedirs(pred_path, exist_ok=True)
S = s


def compute_feature(i):
   corresponding_batch = (i-first_batch_size)//((nb_classes-first_batch_size)//S)+1
   if i<first_batch_size:
      corresponding_batch=0
   test_feats = os.path.join(test_feats_path,str(i))
   print(test_feats)
   for batchs in range(corresponding_batch, S+1):
      os.makedirs(os.path.join(pred_path,"batch"+str(batchs)),exist_ok=True)
      pred_file = os.path.join(pred_path,"batch"+str(batchs),str(i))
      if not os.path.exists(pred_file): # TODO 
         with open(pred_file, "w") as f_pred:
            syns = []
            f_list_syn = list(range(((nb_classes-first_batch_size)//S)*(batchs)+first_batch_size))
            #print('class',f_list_syn[-1],'batch', batchs)
            for syn in f_list_syn:
               syn = str(syn)
               syns.append(syn)
            print("synsets:",len(syns))
            weights_list = []  
            biases_list = []
            for syn in range(0,len(syns)):
               line_cnt = 0 # counter to get the weights and bias lines
               target_model = os.path.join(model_dir,"batch"+str(batchs),str(syn)+".model")
               f_model = open(target_model)
               for line in f_model:
                  line = line.rstrip()
                  if line_cnt == 0:
                     parts = line.split(" ")
                     parts_float = [] # tmp list to store the weights
                     for pp in parts:
                        parts_float.append(float(pp))
                     weights_list.append(parts_float)
                  elif line_cnt == 1:
                     biases_list.append(float(line))
                  line_cnt = line_cnt + 1
               f_model.close()
            print("list sizes - weights:",len(weights_list),"; biases:",len(biases_list))
            f_test_feat = open(test_feats, 'r')
            for vline in f_test_feat.readlines():
               vparts = vline.split(" ")
               crt_feat = [[float(vp) for vp in vparts]]
               crt_feat = Normalizer().fit_transform(crt_feat)[0]
               pred_dict = []
               for cls_cnt in range(0, len(weights_list)):
                  cls_score = np.dot(crt_feat, weights_list[cls_cnt]) + biases_list[cls_cnt]
                  pred_dict.append(-cls_score)
               pred_line = ""
               predictions_idx = sorted(range(len(pred_dict)), key=lambda k: -pred_dict[k])
               for idx in predictions_idx:
                  pred_line = pred_line+" "+str(idx)+":"+str(pred_dict[idx]) 
               pred_line = pred_line.lstrip()
               f_pred.write(pred_line+"\n")
            f_test_feat.close()
      else:
         print("exists predictions file:",pred_file)
with Pool() as p:
   p.map(compute_feature, range(nb_classes))
   #p.map(compute_feature, range(1))
