import os
import sys
import csv
import numpy as np
from multiprocessing import Pool


nb_classes = int(sys.argv[1]) #number of classes in TOTAL
dataset = sys.argv[2]
first_batch_size = int(sys.argv[3])
il_states = int(sys.argv[4])  #initialement on donnerait 3, ici on donne 4
pred_root= sys.argv[5]
s= il_states-1 #(pour Eva :) )

"""
root_path_pred = f"{pred_root}/mobil/{dataset}/b{first_batch_size}/s{il_states}/"
if "food1000" in root_path_pred:
    root_path_pred = f"{pred_root}/mobil/{dataset}/b{first_batch_size}/s{il_states}/"
elif "ils500" in root_path_pred:
    root_path_pred = f"{pred_root}/mobil/{dataset}/b{first_batch_size}/s{il_states}/"
"""

root_path_pred = os.path.join(pred_root,"mobil","b"+str(first_batch_size),"s"+str(il_states))

batch_size = (nb_classes-first_batch_size)//s

batches = range(s+1)
#batches = [1,2]
resultats = {}
def flatten(t):
    return [item for sublist in t for item in sublist]
def compute_score(nb_batch):
    path_pred = os.path.join(root_path_pred,"batch"+str(nb_batch))
    y_pred = []
    score_top5 = []
    y_true = []
    for c in range(first_batch_size+batch_size*nb_batch):
        with open(os.path.join(path_pred,str(c)), newline='') as f:
            reader = csv.reader(f, delimiter=' ')
            to_append_top5 = [[int(elt[i].split(":")[0]) for i in range(5)] for elt in list(reader)]
            to_append = [elt[0] for elt in to_append_top5]
            y_pred.append(to_append)
            score_top5.append([c in to_append_top5[i] for i in range(len(to_append))])
            y_true.append([c for _ in to_append])
    y_pred = np.asarray(flatten(y_pred))
    y_pred_top5 = flatten(score_top5)
    y_true = np.asarray(flatten(y_true))
    #print('batch',nb_batch,', y_pred=',y_pred)
    #print('batch',nb_batch,', y_pred_top5=',y_pred_top5)
    #print('batch',nb_batch,', y_true=',y_true)
    return((nb_batch,[np.mean(y_pred == y_true),np.mean(y_pred_top5)]))

def detailled_score(nb_batch):
    path_pred = os.path.join(root_path_pred,"batch"+str(nb_batch))
    res = []
    for c in range(first_batch_size+batch_size*nb_batch):
        y_pred = []
        score_top5 = []
        y_true = []
        with open(os.path.join(path_pred,str(c)), newline='') as f:
            reader = csv.reader(f, delimiter=' ')
            to_append_top5 = [[int(elt[i].split(":")[0]) for i in range(5)] for elt in list(reader)]
            to_append = [elt[0] for elt in to_append_top5]
            y_pred.append(to_append)
            score_top5.append([c in to_append_top5[i] for i in range(len(to_append))])
            y_true.append([c for _ in to_append])
        y_pred = np.asarray(flatten(y_pred))
        y_pred_top5 = flatten(score_top5)
        y_true = np.asarray(flatten(y_true))
        #print('batch',nb_batch,', y_pred=',y_pred)
        #print('batch',nb_batch,', y_pred_top5=',y_pred_top5)
        #print('batch',nb_batch,', y_true=',y_true)
        prdic = np.mean(y_pred == y_true)
        res.append(prdic)

    return([np.mean(res[:first_batch_size])]+[np.mean(res[i:i+batch_size]) for i in range(first_batch_size,len(res),batch_size)])

with Pool() as p:
    resultats = dict(p.map(compute_score, batches))
#print(resultats)
#for i in range(s+1):
#    scori = detailled_score(i)
#    print(f'score_{i}=',scori, f', std_{i}=',np.std(scori))
    
top1=[]
top5=[]
all_top1 = '['
for batch_number in batches:
    all_top1 = all_top1+str(f'{resultats[batch_number][0]:.3f}')+","
    print(f'batch {batch_number}, top1 = {resultats[batch_number][0]:.3f}, top5 = {resultats[batch_number][1]:.3f}')
    top1.append(resultats[batch_number][0])
    top5.append(resultats[batch_number][1])
print('=================================================')
print('===================  TOTAL  =====================')
all_top1 = all_top1.rstrip(',')
all_top1 = all_top1+']'
print(all_top1)
print([round(100*elt,2) for elt in top1])
print(f'top1 sans etat init = {sum(top1[1:])/len(top1[1:]):.3f} top5 = {sum(top5[1:])/len(top5[1:]):.3f}')
print(f'top1 avec etat init = {sum(top1)/len(top1):.3f} top5 = {sum(top5)/len(top5):.3f}')
print('>>> results MobIL | acc@1 = {el1:.3f} \t acc@5 = {el2:.3f} '.format(el1 = sum(top1[1:])/len(top1[1:]), el2 = sum(top5[1:])/len(top5[1:])))