import random
import numpy as np
import torch
from networks.subnet import SubnetLinear, SubnetConv2d
from networks.subnet import GetSubnetFaster

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def data_sampler_for_class(x, y, n_sample):
    #对每个类别进行采样n个样本
    x_sample = []
    len(torch.unique(y))
    for i in torch.unique(y):
        #随机采样
        index = np.random.choice(np.where(y == i)[0], n_sample//len(torch.unique(y)))
        x_sample.append(x[index])
    x_sample = torch.cat(x_sample, dim=0)
    return x_sample


def kaiming_init(model):
    for module in model.modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            torch.nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)
        if isinstance(module, torch.nn.BatchNorm2d):
            torch.nn.init.constant_(module.weight, 1)
            torch.nn.init.constant_(module.bias, 0)
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            torch.nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)
                if isinstance(module, SubnetLinear, SubnetConv2d):
                    torch.nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                    if module.bias is not None:
                        torch.nn.init.constant_(module.bias, 0)

def get_random_mask(model):
    init_mask = {}
    for name, module in model.named_modules():
        # For the time being we only care about the current task outputhead
        if 'last' in name:
                continue

        if isinstance(module, SubnetLinear) or isinstance(module, SubnetConv2d):
            
            w_m = module.w_m.detach().clone()

            #w_m随机
            w_m = torch.rand_like(w_m)

            weight_mask = GetSubnetFaster.apply(w_m, module.zeros_weight, module.ones_weight, module.sparsity)
            init_mask[name + '.weight'] = (weight_mask.detach().clone() > 0).type(torch.long)

            if getattr(module, 'bias') is not None:
                init_mask[name + '.bias'] = (weight_mask.detach().clone() > 0).type(torch.long)
            else:
                init_mask[name + '.bias'] = None

def get_init_mask(model):
    init_mask = {}
    from networks.subnet import GetSubnetFaster
    for name, module in model.named_modules():
        # For the time being we only care about the current task outputhead
        if 'last' in name:
                continue


        if isinstance(module, SubnetLinear) or isinstance(module, SubnetConv2d):
            weight_mask = GetSubnetFaster.apply(module.w_m.abs(),
                                                        module.zeros_weight,
                                                        module.ones_weight,
                                                        module.sparsity)
            init_mask[name + '.weight'] = (weight_mask.detach().clone() > 0).type(torch.long)

            if getattr(module, 'bias') is not None:
                init_mask[name + '.bias'] = (weight_mask.detach().clone() > 0).type(torch.long)
            else:
                init_mask[name + '.bias'] = None
    return init_mask

def get_backbone_fc_param(model):
    fc_param = []
    backbone_param = []
    for n, param in model.named_parameters():
        if 'last' in n:
            fc_param.append(param)
        else:
            backbone_param.append(param)
    return backbone_param, fc_param

def get_feature_bases(model, task_id, feature_list, feature_mat, threshold, device):
    for i, key in enumerate(model.act.keys()):
        if key in feature_mat.keys():
            continue
        activation = feature_list[i]
        U,S,Vh = np.linalg.svd(activation, full_matrices=False)
        # criteria (Eq-5)
        sval_total = (S**2).sum()
        sval_ratio = (S**2)/sval_total
        r = np.sum(np.cumsum(sval_ratio)<threshold) #+1  
        B = U[:,0:r]
        Uf=torch.Tensor(np.dot(B,B.transpose())).to(device)
        print('Layer {} - Projection Matrix shape: {}'.format(key,Uf.shape))
        feature_mat[key] = Uf.detach().clone()



def init_model_mask(model):
    for name, module in model.named_modules():
        # For the time being we only care about the current task outputhead
        if 'last' in name:
                continue

        if isinstance(module, SubnetLinear) or isinstance(module, SubnetConv2d):
            module.init_mask_parameters()