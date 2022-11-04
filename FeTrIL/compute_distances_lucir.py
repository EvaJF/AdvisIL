from genericpath import exists
import os
import csv
import numpy as np
from sklearn.metrics import pairwise_distances
import sys
from multiprocessing import Pool

"""
python /home/users/apopescu/MobIL/compute_distances_lucir.py 100 food101 50 5 /home/data/didi/ST/features/
"""

nb_classes = int(sys.argv[1]) #number of classes in TOTAL
dataset = sys.argv[2]
first_batch_size = int(sys.argv[3])
il_states = int(sys.argv[4]) #initialement on donnerait 3, ici on donne 4
feat_root= sys.argv[5]
s = il_states-1

#hard-coding for food-1000 and ilsvrc-500 pretrained features


"""
if "food1000" in feat_root:
    train_data_path = f"{feat_root}/{dataset}/b1000/train/"
elif "ils500" in feat_root:
    train_data_path = f"{feat_root}/{dataset}/b500/train/"
elif "imn1000" in feat_root:
    train_data_path = f"{feat_root}/{dataset}/b1000/train/"
elif "imn2000" in feat_root:
    train_data_path = f"{feat_root}/{dataset}/b2000/train/"
elif "imn4000" in feat_root:
    train_data_path = f"{feat_root}/{dataset}/b4000/train/"
elif "imn6000" in feat_root:
    train_data_path = f"{feat_root}/{dataset}/b6000/train/"
"""

crt_b = str(first_batch_size)

train_data_path = os.path.join(feat_root,"b"+crt_b,"train/")
print(train_data_path)
#train_data_path_lucir = f"{feat_root}/mobil/{dataset}/b{first_batch_size}/s{il_states}/train/"
train_data_path_lucir = os.path.join(feat_root,"mobil","b"+str(first_batch_size),"s"+str(il_states),"train/")
print("we will save here -->", train_data_path_lucir)
batch_size = (nb_classes-first_batch_size)//s



total_liste = []
def compute_batch(curr_state):
    os.makedirs(os.path.join(train_data_path_lucir,"batch"+str(curr_state)), exist_ok=True)
    for data_path in set([train_data_path]):
        chaton = []
        poney = []
        for i in range(first_batch_size+curr_state*batch_size):
            path_to_data = data_path+str(i)
            to_np = []
            f_data = open(path_to_data)
            for dline in f_data:
                dline = dline.rstrip()
                to_np.append(np.fromstring(dline, sep=' ',dtype=float))
            f_data.close()
            
            with open(path_to_data, newline='') as f:
                reader = csv.reader(f, delimiter=' ')
                data = np.array(list(reader), dtype=float)
            chaton.append(data)
            poney += [i for _ in range(len(data))]
        if data_path == train_data_path:
            X_train = np.concatenate(chaton)
            y_train = np.array(poney)
            
    X = X_train
    y = y_train
    

    means = [np.mean(X[y==i], axis=0) for i in set(y)]
    if curr_state==0:
        means_curr_state = means[:first_batch_size]
    else:
        means_curr_state = means[first_batch_size+(curr_state-1)*batch_size:first_batch_size+curr_state*batch_size]

    distances = pairwise_distances(means, means_curr_state)

    if curr_state==0:
        c_distance_mini = np.argmin(distances, axis=1)
    else:
        c_distance_mini = np.argmin(distances, axis=1)+first_batch_size+(curr_state-1)*batch_size


    Xp = []
    Yp = []
    for c in set(y):
        if not os.path.exists(os.path.join(train_data_path_lucir,"batch"+str(curr_state),str(c))):
            X_tmp = X[y==c_distance_mini[c]] - np.expand_dims(means[c_distance_mini[c]], 0) + np.expand_dims(means[c], 0)
            print("saving batch",curr_state,"- class",c)
            np.savetxt(os.path.join(train_data_path_lucir,"batch"+str(curr_state),str(c)), X_tmp, fmt='%1.8f')

to_compute = []
for i in range(s+1):
    to_check = any([not os.path.exists(os.path.join(train_data_path_lucir,"batch"+str(i),str(c))) for c in range(first_batch_size+(i)*batch_size)])
    if to_check:
        to_compute.append(i)
print("ToC:",to_compute)
with Pool() as p:
    p.map(compute_batch, to_compute)
