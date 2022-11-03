#!/usr/bin/env python
# coding=utf-8
import torch
import numpy as np
from utils_pytorch import *

def compute_features(tg_feature_model, evalloader, num_samples, num_features, device=None):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tg_feature_model.eval()

    #print("num_samples : ", num_samples)
    #print("num_features : ", num_features)
    #print("features matrix has size ", num_samples, num_features)
    features = np.zeros([num_samples, num_features])
    start_idx = 0
    with torch.no_grad():
        for inputs, targets in evalloader:
            inputs = inputs.to(device)
            #print("inputs size", inputs.shape)
            #print("features slice of dim ", features[start_idx:start_idx+inputs.shape[0], :].shape)
            #print("size of model outputs : np.squeeze(tg_feature_model(inputs).cpu()) ", np.squeeze(tg_feature_model(inputs).cpu()).shape)
            features[start_idx:start_idx+inputs.shape[0], :] = np.squeeze(tg_feature_model(inputs).cpu())
            start_idx = start_idx+inputs.shape[0]
    assert(start_idx==num_samples)
    return features
