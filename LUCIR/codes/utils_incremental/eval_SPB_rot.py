#!/usr/bin/env python
# coding=utf-8
import torch
import torch.nn as nn
from utils_pytorch import *
from Utils import DataUtils
import torch.nn.functional as F
from AverageMeter import *
import math
from torch import linalg as LA
utils = DataUtils()
# Margin Ranking Loss
cur_features = []
ref_features = []
old_scores = []
new_scores = []
def get_ref_features(self, inputs, outputs):
    global ref_features
    ref_features = inputs[0]

def get_cur_features(self, inputs, outputs):
    global cur_features
    cur_features = inputs[0]

def get_old_scores_before_scale(self, inputs, outputs):
    global old_scores
    old_scores = outputs

def get_new_scores_before_scale(self, inputs, outputs):
    global new_scores
    new_scores = outputs

def eval_SPB_rot(first_path_model, tg_model, alltestloader, iteration, P, device=None):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    top = min(5, tg_model.fc.out_features)
    tg_model = torch.load(first_path_model)
    tg_model.eval()
    top1all = AverageMeter()
    top5all = AverageMeter()
    top1old = AverageMeter()
    top5old = AverageMeter()
    top1new = AverageMeter()
    top5new = AverageMeter()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(alltestloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = tg_model(inputs)
            if iteration>0:
                oldsamples = targets<=iteration*P
                newsamples = targets>iteration*P
                newoutputs, newtargets = outputs.data[newsamples], targets[newsamples]
                oldoutputs, oldtargets = outputs.data[oldsamples], targets[oldsamples]
                if oldtargets.size(0)>0:
                    prec1, prec5 = utils.accuracy(oldoutputs, oldtargets, topk=(1, top))
                    top1old.update(prec1.item(), inputs.size(0))
                    top5old.update(prec5.item(), inputs.size(0))
                if newtargets.size(0)>0:
                    prec1, prec5 = utils.accuracy(newoutputs, newtargets, topk=(1, top))
                    top1new.update(prec1.item(), inputs.size(0))
                    top5new.update(prec5.item(), inputs.size(0))
            if targets.size(0)>0:
                prec1, prec5 = utils.accuracy(outputs.data, targets, topk=(1, top))
                top1all.update(prec1.item(), inputs.size(0))
                top5all.update(prec5.item(), inputs.size(0))
    if iteration>0:
        print('state',iteration,'| old-classes Test ({}) |  acc@1 = {:.2f} | acc@{} = {:.2f}'.format(len(alltestloader), top1old.avg, top, top5old.avg))
        print('state',iteration,'| new-classes Test ({}) |  acc@1 = {:.2f} | acc@{} = {:.2f}'.format(len(alltestloader), top1new.avg, top, top5new.avg))
    print('state',iteration,'| all-classes Test ({}) |  acc@1 = {:.2f} | acc@{} = {:.2f}'.format(len(alltestloader), top1all.avg, top, top5all.avg))

    return tg_model


