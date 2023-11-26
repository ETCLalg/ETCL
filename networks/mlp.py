import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict

from copy import deepcopy

from .subnet import SubnetConv2d, SubnetLinear, get_none_masks

class SubnetMLPNet(nn.Module):
    def __init__(self, taskcla, sparsity, n_hidden=100, input_size=784):
        super(SubnetMLPNet, self).__init__()

        self.act=OrderedDict()
        self.fc1 = SubnetLinear(input_size, n_hidden, sparsity=sparsity, bias=False)
        self.fc2 = SubnetLinear(n_hidden, n_hidden, sparsity=sparsity, bias=False)

        self.taskcla = taskcla

        self.multi_head = True

        if self.multi_head:
            self.last = nn.ModuleList()
            for t, n in self.taskcla:
                self.last.append(nn.Linear(n_hidden, n, bias=False))
        else:
            self.last = nn.Linear(n_hidden, taskcla[0][1], bias=False)

        self.relu = nn.ReLU()

        # Constant none_masks
        self.none_masks = {}
        for name, module in self.named_modules():
            if isinstance(module, SubnetLinear) or isinstance(module, SubnetConv2d):
                self.none_masks[name + '.weight'] = None
                self.none_masks[name + '.bias'] = None

    def forward(self, x, task_id, mask, mode="train", size=-1, epoch=-1, end_epoch=0):
        if mask is None:
            mask = self.none_masks

        bsz = deepcopy(x.size(0))
        x=x.reshape(bsz,-1)
        # self.act['Lin1'] = x
        x = self.fc1(x, weight_mask=mask['fc1.weight'], bias_mask=mask['fc1.bias'], mode=mode, epoch=epoch, end_epoch=end_epoch)
        x = self.relu(x)
        # self.act['Lin2'] = x
        x = self.fc2(x, weight_mask=mask['fc2.weight'], bias_mask=mask['fc2.bias'], mode=mode, epoch=epoch, end_epoch=end_epoch)
        x = self.relu(x)
        self.act['last.' + str(task_id)] = x

        if self.multi_head:
            h_keys = ["last.{}.weight".format(task_id), "last.{}.bias".format(task_id)]
            y = self.last[task_id](x)

        else:
            y = self.last(x)

        return y

    def get_masks(self, task_id):
        task_mask = {}
        for name, module in self.named_modules():
            # For the time being we only care about the current task outputhead
            if self.multi_head:
                if 'last' in name:
                    if name != 'last.' + str(task_id):
                        continue
            else:
                None

            if isinstance(module, SubnetLinear) or isinstance(module, SubnetConv2d):
                task_mask[name + '.weight'] = (module.weight_mask.detach().clone() > 0).type(torch.long)

                if getattr(module, 'bias') is not None:
                    task_mask[name + '.bias'] = (module.bias_mask.detach().clone() > 0).type(torch.long)
                else:
                    task_mask[name + '.bias'] = None

        return task_mask
    

    def get_feature_matrix(self, x, task_id, feature_matrix=None):
        task_mask = self.get_masks(task_id)
        threshold = [0.99999, 0.99999, 0.99]

        mat_list = OrderedDict()
        with torch.no_grad():
            self.eval()
            for name, module in self.named_modules():
                # For the time being we only care about the current task outputhead
                if f'last.{task_id}' in name:
                    bsz = x.size(0)
                    activation = x[0:bsz].T.cpu().numpy()
                    mat_list[name + ".weight"] = activation
                if isinstance(module, SubnetLinear):
                    bsz = x.size(0)
                    activation = x[0:bsz].T.cpu().numpy()                    
                    mat_list[name + ".weight"] = activation
                    with torch.no_grad():
                        x = module(x, weight_mask=task_mask[name + '.weight'], bias_mask=task_mask[name + '.bias'], mode="valid")

        
        self.train()
        return self.update_feature_martix(mat_list, feature_matrix, threshold, x.device)

        
    def update_feature_martix(self, mat_list, feature_matrix, threshold, device):
        print ('Threshold: ', threshold) 
        if not feature_matrix:
            feature_matrix = OrderedDict()
            for i, key in enumerate(mat_list.keys()):
                activation = mat_list[key]
                U,S,Vh = np.linalg.svd(activation, full_matrices=False)
                # criteria (Eq-5)
                sval_total = (S**2).sum()
                sval_ratio = (S**2)/sval_total
                r = np.sum(np.cumsum(sval_ratio)<0.9999) #+1  
                feature_matrix[key] = U[:,0:r]
        else:
            for i, key in enumerate(mat_list.keys()):
                if key not in feature_matrix.keys():
                    activation = mat_list[key]
                    U,S,Vh = np.linalg.svd(activation, full_matrices=False)
                    # criteria (Eq-5)
                    sval_total = (S**2).sum()
                    sval_ratio = (S**2)/sval_total
                    r = np.sum(np.cumsum(sval_ratio)<0.9999) #+1  
                    feature_matrix[key] = U[:,0:r]
                else:
                    activation = mat_list[key]
                    U1,S1,Vh1=np.linalg.svd(activation, full_matrices=False)
                    sval_total = (S1**2).sum()
                    # Projected Representation (Eq-8)
                    act_hat = activation - np.dot(np.dot(feature_matrix[key],feature_matrix[key].transpose()),activation)
                    U,S,Vh = np.linalg.svd(act_hat, full_matrices=False)
                    # criteria (Eq-9)
                    sval_hat = (S**2).sum()
                    sval_ratio = (S**2)/sval_total               
                    accumulated_sval = (sval_total-sval_hat)/sval_total
                    
                    r = 0
                    for ii in range (sval_ratio.shape[0]):
                        if accumulated_sval < threshold[i]:
                            accumulated_sval += sval_ratio[ii]
                            r += 1
                        else:
                            break
                    if r == 0:
                        print ('Skip Updating GPM for layer: {}'.format(i+1)) 
                        continue
                    # update GPM
                    Ui=np.hstack((feature_matrix[key],U[:,0:r]))  
                    if Ui.shape[1] > Ui.shape[0] :
                        feature_matrix[key]=Ui[:,0:Ui.shape[0]]
                    else:
                        feature_matrix[key]=Ui
    
        print('-'*40)
        print('Gradient Constraints Summary')
        print('-'*40)
        for key in feature_matrix.keys():
            print ('Layer {} : {}/{}'.format(key,feature_matrix[key].shape[1], feature_matrix[key].shape[0]))
        print('-'*40)
        return feature_matrix
            