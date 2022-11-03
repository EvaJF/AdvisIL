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

def train_eval_SPB(epochs, first_path_model, tg_model, ref_model, tg_optimizer, tg_lr_scheduler, \
            trainloader, testloader, \
            iteration, start_iteration, \
            lamda, \
            dist, K, lw_mr, \
            fix_bn=False, weight_per_class=None, device=None):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    top = min(5, tg_model.fc.out_features)
    if iteration > start_iteration:
        ref_model.eval()
        num_old_classes = ref_model.fc.out_features
        num_new_classes = tg_model.fc.out_features - num_old_classes
        handle_ref_features = ref_model.fc.register_forward_hook(get_ref_features)
        handle_cur_features = tg_model.fc.register_forward_hook(get_cur_features)
        handle_old_scores_bs = tg_model.fc.fc1.register_forward_hook(get_old_scores_before_scale)
        handle_new_scores_bs = tg_model.fc.fc2.register_forward_hook(get_new_scores_before_scale)
        tg_feature_model = nn.Sequential(*list(tg_model.children())[:-1])
        ref_feature_model = nn.Sequential(*list(ref_model.children())[:-1])
        for epoch in range(epochs):
            #train
            tg_model.train()
            if iteration > start_iteration:
                tg_feature_model.train()
                ref_feature_model.train()
            if fix_bn:
                for m in tg_model.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.eval()

            tg_lr_scheduler.step()
            for batch_idx, (inputs, targets) in enumerate(trainloader):
                inputs, targets = inputs.to(device), targets.to(device)
                tg_optimizer.zero_grad()
                outputs = tg_model(inputs)
                if iteration == start_iteration:
                    loss = nn.CrossEntropyLoss(weight_per_class)(outputs, targets)
                else:
                    loss1 = torch.mean(LA.norm(F.normalize(tg_feature_model(inputs).squeeze(), dim=1) - F.normalize(ref_feature_model(inputs).squeeze(), dim=1), dim=1))#.pow(2).sum(1).sqrt()
                    loss2 = nn.CrossEntropyLoss(weight_per_class)(outputs, targets)
                    #print(loss1,loss2)
                    loss = num_old_classes/num_new_classes*loss1 + num_new_classes/num_old_classes*loss2
                loss.backward()
                tg_optimizer.step()

            # eval
            top1 = AverageMeter()
            top5 = AverageMeter()
            tg_model.eval()

            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(testloader):
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = tg_model(inputs)
                    prec1, prec5 = utils.accuracy(outputs.data, targets, topk=(1, top))
                    top1.update(prec1.item(), inputs.size(0))
                    top5.update(prec5.item(), inputs.size(0))

            print('{:03}/{:03} | Test ({}) |  acc@1 = {:.2f} | acc@{} = {:.2f}'.format(
                epoch+1, epochs,  len(testloader), top1.avg, top, top5.avg))

        print("Removing register_forward_hook")
        handle_ref_features.remove()
        handle_cur_features.remove()
        handle_old_scores_bs.remove()
        handle_new_scores_bs.remove()
    else:
         # eval
        try:
            tg_model = torch.load(first_path_model)
        except:
            ref_model.eval()
            num_old_classes = ref_model.fc.out_features
            num_new_classes = tg_model.fc.out_features - num_old_classes
            handle_ref_features = ref_model.fc.register_forward_hook(get_ref_features)
            handle_cur_features = tg_model.fc.register_forward_hook(get_cur_features)
            handle_old_scores_bs = tg_model.fc.fc1.register_forward_hook(get_old_scores_before_scale)
            handle_new_scores_bs = tg_model.fc.fc2.register_forward_hook(get_new_scores_before_scale)
            tg_feature_model = nn.Sequential(*list(tg_model.children())[:-1])
            ref_feature_model = nn.Sequential(*list(ref_model.children())[:-1])
            for epoch in range(epochs):
                #train
                tg_model.train()
                if iteration > start_iteration:
                    tg_feature_model.train()
                    ref_feature_model.train()
                if fix_bn:
                    for m in tg_model.modules():
                        if isinstance(m, nn.BatchNorm2d):
                            m.eval()

                tg_lr_scheduler.step()
                for batch_idx, (inputs, targets) in enumerate(trainloader):
                    inputs, targets = inputs.to(device), targets.to(device)
                    tg_optimizer.zero_grad()
                    outputs = tg_model(inputs)
                    loss = nn.CrossEntropyLoss(weight_per_class)(outputs, targets)
                    loss.backward()
                    tg_optimizer.step()

                # eval
                top1 = AverageMeter()
                top5 = AverageMeter()
                tg_model.eval()

                with torch.no_grad():
                    for batch_idx, (inputs, targets) in enumerate(testloader):
                        inputs, targets = inputs.to(device), targets.to(device)
                        outputs = tg_model(inputs)
                        prec1, prec5 = utils.accuracy(outputs.data, targets, topk=(1, top))
                        top1.update(prec1.item(), inputs.size(0))
                        top5.update(prec5.item(), inputs.size(0))

                print('{:03}/{:03} | Test ({}) |  acc@1 = {:.2f} | acc@{} = {:.2f}'.format(
                    epoch+1, epochs,  len(testloader), top1.avg, top, top5.avg))
        top1 = AverageMeter()
        top5 = AverageMeter()
        
        tg_model.eval()

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = tg_model(inputs)
                prec1, prec5 = utils.accuracy(outputs.data, targets, topk=(1, top))
                top1.update(prec1.item(), inputs.size(0))
                top5.update(prec5.item(), inputs.size(0))

        print('first state | Test ({}) |  acc@1 = {:.2f} | acc@{} = {:.2f}'.format(len(testloader), top1.avg, top, top5.avg))

    return tg_model