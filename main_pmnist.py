# Authorized by Haeyong Kang.

import torch
import torch.optim as optim
import numpy as np
import argparse,time
from copy import deepcopy
from networks.mlp import SubnetMLPNet as MLPNet
from networks.utils import *
from utils import *
from collections import OrderedDict
from scipy.stats import wasserstein_distance

def contrastive_calculate(model, task_id, task_relation):

    sim_tasks, dis_tasks = task_relation
    sims = sim_tasks[task_id]
    dsims = dis_tasks[task_id]
    
    loss = []
    current_fc_weight = model.last[task_id].weight
    sz = current_fc_weight.shape[0]

    for s in sims:
        
        sim_fc_weigth = model.last[s].weight
        cos_sim = torch.nn.functional.cosine_similarity(current_fc_weight.view(sz, -1),sim_fc_weigth.view(sz, -1), dim=1)
        cos_sim = (torch.mean(cos_sim) + 1.0) / 2.0 
        label = torch.ones(1).to(model.last[s].weight.device)

        loss.append(torch.nn.functional.binary_cross_entropy(cos_sim.view(1), label))


    # for s in dsims:
    #     dsim_fc_weigth = model.last[s].weight
    #     cos_sim = torch.nn.functional.cosine_similarity(current_fc_weight.view(sz, -1), dsim_fc_weigth.view(sz, -1))
    #     cos_sim = (torch.mean(cos_sim) + 1.0) / 2.0

    #     label = torch.zeros(1).to(model.last[s].weight.device)

    #     loss.append(torch.nn.functional.binary_cross_entropy(cos_sim.view(1), label))
    #     loss.append(cos_sim)




    if len(loss) == 0:
        return 0

    loss = torch.mean(torch.stack(loss))
    return loss

def update_task_discrimination(task_id, feature_list_ori, feature_list_new, sim_tasks, dsim_tasks, threshold=0.7):

    #计算训练后的下一个任务和原任务的距离
    distance_ori = []
    for t in range(task_id):
        distance_ori.append(
            wasserstein_distance(
                feature_list_ori[task_id].flatten(), feature_list_ori[t].flatten()
            )
        )
    distance_new = []
    for t in range(task_id):
        distance_new.append(
            wasserstein_distance(
                feature_list_new[t][0].flatten(), feature_list_new[t][1].flatten()
            )
        )

    distance_ori_np = np.array(distance_ori)
    distance_new_np = np.array(distance_new)

    dis = np.abs((distance_ori_np - distance_new_np))
    indices = np.where(dis < 0.1)
    factors = 10 ** (np.ceil(-np.log10(dis[indices])) -1)
    dis[indices] *= factors

    print("distance ori:")
    for ii in distance_ori_np:
        print("{:>2.2f}".format(ii), end=' ')
    print()

    print("distance new:")
    for ii in distance_new_np:
        print("{:>2.2f}".format(ii), end=' ')
    print()

    print("D-value between old and new distance:")
    for ii in dis:
        print("{:>2.2f}".format(ii), end=' ')
    print()

    print("The threshold: ", threshold)

    sim_flag_1 = distance_new_np < distance_ori_np
    sim_flag_2 = dis > threshold
    sim_flag = sim_flag_1 * sim_flag_2

    sim_tasks[task_id] = np.where(sim_flag)[0]
    dsim_tasks[task_id] = np.where(sim_flag_1 == False)[0]


    print(f"Task {task_id} have sim tasks: ")
    for i in sim_tasks[task_id]:
        print("{}".format(i, sim_tasks[i]), end=", ")
    print()

    print(f"Task {task_id} have dsim tasks: ")
    for i in dsim_tasks[task_id]:
        print("{}".format(i, dsim_tasks[i]), end=", ")
    print()



    return sim_tasks, dsim_tasks


def train(args, model, device, x,y, optimizer, criterion, task_id_nominal, consolidated_masks, feature_matrix, per_task_masks, task_relation):

    sim_tasks, dis_tasks = task_relation
    sims = sim_tasks[task_id_nominal]
    disms = dis_tasks[task_id_nominal]

    backbone_optimizer, fc_optimizer = optimizer

    model.train()
    r=np.arange(x.size(0))
    np.random.shuffle(r)
    r=torch.LongTensor(r).to(device)
    # Loop batches
    for i in range(0,len(r),args.batch_size_train):
        if ((i + args.batch_size_train) <= len(r)):
            b=r[i:i+args.batch_size_train]
        else:
            b=r[i:]
        b = b.to(x.device)
        data = x[b]
        data, target = data.to(device), y[b].to(device)
        loss = 0

        backbone_optimizer.zero_grad()
        if fc_optimizer is not None:
            fc_optimizer.zero_grad()

        for i in sims:
            mask = per_task_masks[i]
            output = model(data, i, mask=mask, mode="train")
            loss += criterion(output, target)

        loss += contrastive_calculate(model, task_id_nominal, task_relation)

        output = model(data, task_id_nominal, None, mode="train")
        loss += criterion(output, target)
        loss.backward()

        # Continual Subnet no backprop
        if consolidated_masks is not None and consolidated_masks != {}: # Only do this for tasks 1 and beyond
            # if args.use_continual_masks:
            for key in consolidated_masks.keys():
                # Skip if not task head is not for curent task
                if 'last' in key:
                    continue
                # Determine wheter it's an output head or not
                key_split = key.split('.')
                if 'last' in key_split or len(key_split) == 2:
                    if 'last' in key_split:
                        module_attr = key_split[-1]
                        module_name = '.'.join(key_split[:-2])
                    else:
                        module_attr = key_split[1]
                        module_name = key_split[0]

                    # Zero-out gradients
                    if (hasattr(getattr(model, module_name), module_attr)):
                        if (getattr(getattr(model, module_name), module_attr) is not None):
                            getattr(getattr(model, module_name), module_attr).grad[consolidated_masks[key] == 1] = 0
                else:
                    module_attr = key_split[-1]
                    # Zero-out gradients
                    curr_module = getattr(getattr(model, key_split[0])[int(key_split[1])], key_split[2])
                    if hasattr(curr_module, module_attr):
                        if getattr(curr_module, module_attr) is not None:
                            getattr(curr_module, module_attr).grad[consolidated_masks[key] == 1] = 0

            #Orthogonal projection gradient updata for fc layer
            for key in feature_matrix.keys():
                if 'last.' in key and str(task_id_nominal) not in key:
                    if getattr(getattr(model,  'last'), key.split('.')[1]).weight.grad is not None:
                        temp = getattr(getattr(model,  'last'), key.split('.')[1]).weight.grad.clone()
                        temp = temp - torch.mm(temp, feature_matrix[key])
                        getattr(getattr(model,  'last'), key.split('.')[1]).weight.grad = temp

        backbone_optimizer.step()
        if fc_optimizer is not None:
            fc_optimizer.step()


def test(args, model, device, x, y, criterion, task_id_nominal, curr_task_masks=None, mode="test"):
    model.eval()
    total_loss = 0
    total_num = 0
    correct = 0
    r=np.arange(x.size(0))
    np.random.shuffle(r)
    r=torch.LongTensor(r).to(device)
    with torch.no_grad():
        # Loop batches
        for i in range(0,len(r),args.batch_size_test):
            if ((i + args.batch_size_test) <= len(r)):
                b=r[i:i+args.batch_size_test]
            else: b=r[i:]

            data = x[b]
            data, target = data.to(device), y[b].to(device)
            if curr_task_masks:
                output = model(data, task_id_nominal, mask=curr_task_masks, mode=mode)
            else:
                output = model(data, task_id_nominal, mask=None, mode=mode)
            loss = criterion(output, target)
            pred = output.argmax(dim=1, keepdim=True)

            correct    += pred.eq(target.view_as(pred)).sum().item()
            total_loss += loss.data.cpu().numpy().item()*len(b)
            total_num  += len(b)

    acc = 100. * correct / total_num
    final_loss = total_loss / total_num
    return final_loss, acc


def main(args):
    ## Device Setting
    tstart=time.time()
    ## Device Setting
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    set_seed(seed=args.seed)

    
    ## Load PermutedMNIST
    from dataloader import pmnist
    data, taskcla, inputsize = pmnist.get(seed=args.seed,
                                          pc_valid=args.pc_valid)

    

    acc_matrix=np.zeros((10,10))
    sparsity_matrix = []
    sparsity_per_task = {}
    criterion = torch.nn.CrossEntropyLoss()

    model = MLPNet(taskcla, args.sparsity, n_hidden=args.n_hidden).to(device)
    # kaiming_init(model)

    print ('Model parameters ---')
    for k_t, (m, param) in enumerate(model.named_parameters()):
        print (k_t,m,param.shape)
    print ('-'*40)

    task_id = 0
    task_list = []
    per_task_masks, consolidated_masks = {}, {}

    sim_tasks = [[] for _ in range(args.n_tasks)]
    dsim_tasks = [[] for _ in range(args.n_tasks)]

    init_mask = get_init_mask(model)
    feature_mat = OrderedDict() 
    data_buffer = []
    feature_list_ori = []
    feature_list_new = [[] for i in range(args.n_tasks)]
    sim_tasks = [[] for _ in range(args.n_tasks)]
    dsim_tasks = [[] for _ in range(args.n_tasks)]



    for k, ncla in taskcla:

        init_model_mask(model)
        #save n samples for each learned tasks
        data_buffer.append(data[k]['train']['x'][:args.sample_size])

        print("get init dristribution for task {}...".format(k))
        mat = get_representation_matrix(model, device, data_buffer[task_id], task_id, None, size=(-1, 784))[task_id]
        feature_list_ori.append(mat)
        
        #get sim tasks and dis tasks
        if task_id > 0:
            print("get new dristribution for task {}...".format(k))
            for i in range(task_id):
                mat_old_task = get_representation_matrix(model, device, data_buffer[i], i, per_task_masks[i], size=(-1, 784))[i]
                mat_new_task = get_representation_matrix(model, device, data_buffer[task_id], i, per_task_masks[i], size=(-1, 784))[task_id]
                feature_list_new[task_id].append((mat_old_task, mat_new_task))
            print("get sim and dis tasks for task {}...".format(k))
            update_task_discrimination(task_id, feature_list_ori, feature_list_new[task_id], sim_tasks, dsim_tasks, threshold=args.sim_threshold)
        

        print('*'*40)
        print('Task {:2d} ({:s})'.format(k,data[k]['name']))
        print('*'*40)
        xtrain=data[k]['train']['x']
        ytrain=data[k]['train']['y']
        xvalid=data[k]['valid']['x']
        yvalid=data[k]['valid']['y']
        xtest =data[k]['test']['x']
        ytest =data[k]['test']['y']

        task_list.append(k)

        lr = args.lr
        best_loss=np.inf
        print ('-'*40)
        print ('Task ID :{} | Learning Rate : {}'.format(task_id, lr))
        print ('-'*40)

        # Optimizer
        lr = args.lr
        backbone_param, fc_param = get_backbone_fc_param(model)
        if args.optim == "sgd":
            backbone_optimizer = optim.SGD(backbone_param, lr=lr, momentum=args.momentum)
        elif args.optim == "adam":
            backbone_optimizer = optim.Adam(backbone_param, lr=lr)
        else:
            raise Exception("[ERROR] The optimizer " + str(args.optim) + " is not supported!")
        fc_optimizer = optim.SGD(fc_param, lr=lr, momentum=args.momentum)



        for epoch in range(1, args.n_epochs+1):
            # Train
            clock0 = time.time()
            train(args, model, device, xtrain, ytrain, [backbone_optimizer, fc_optimizer], criterion, task_id, consolidated_masks, feature_mat, per_task_masks, (sim_tasks, dsim_tasks))
            clock1 = time.time()
            tr_loss,tr_acc = test(args, model, device, xtrain, ytrain,  criterion, task_id, curr_task_masks=consolidated_masks, mode="valid")
            clock2=time.time()
            print('Epoch {:3d} | Train: loss={:.3f}, acc={:5.1f}% | time={:5.1f}ms | test time={:5.1f}ms'.format(epoch,\
                                                        tr_loss,tr_acc, 1000*(clock1-clock0), (clock2 - clock1)*1000 ), end='')
            # Validate
            valid_loss,valid_acc = test(args, model, device, xvalid, yvalid,  criterion, task_id, curr_task_masks=None, mode="valid")
            print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss, valid_acc),end='')
            # Adapt lr
            if valid_loss<best_loss:
                best_loss=valid_loss
                best_model=get_model(model)
                patience=args.lr_patience
                print(' *',end='')
            else:
                patience-=1
                if patience<=0:
                    lr/=args.lr_factor
                    print(' lr={:.1e}'.format(lr),end='')
                    if lr<args.lr_min:
                        print()
                        break
                    patience=args.lr_patience
                    adjust_learning_rate(backbone_optimizer, epoch, args)
                    if fc_optimizer is not None:
                        adjust_learning_rate(fc_optimizer, epoch, args)
            print()

        # Restore best model
        set_model_(model,best_model)
        per_task_masks[task_id] = model.get_masks(task_id)
        # Consolidate task masks to keep track of parameters to-update or not
        curr_head_keys = ["last.{}.weight".format(task_id), "last.{}.bias".format(task_id)]
        if task_id == 0:
            consolidated_masks = deepcopy(per_task_masks[task_id])
        else:
            for key in per_task_masks[task_id].keys():
                # Skip output head from other tasks
                # Also don't consolidate output head mask after training on new tasks; continue
                if "last" in key:
                    if key in curr_head_keys:
                        consolidated_masks[key] = deepcopy(per_task_masks[task_id][key])
                    continue

                # Or operation on sparsity
                if consolidated_masks[key] is not None and per_task_masks[task_id][key] is not None:
                    consolidated_masks[key] = 1-((1-consolidated_masks[key])*(1-per_task_masks[task_id][key]))

        
        #Get feature bases for orthogonal projection        
        mat_list = get_representation_matrix(model, device, data_buffer[task_id], task_id, model.get_masks(task_id), size=(-1, 784))
        get_feature_bases(model, task_id, mat_list, feature_mat, args.k_threshold, device)


        # Print Sparsity
        sparsity_per_layer = print_sparsity(consolidated_masks)
        all_sparsity = global_sparsity(consolidated_masks)
        print("Global Sparsity: {}".format(all_sparsity))
        sparsity_matrix.append(all_sparsity)
        sparsity_per_task[task_id] = sparsity_per_layer

        # Test
        print ('-'*40)
        test_loss, test_acc = test(args, model, device, xtest, ytest,  criterion, task_id, curr_task_masks=per_task_masks[task_id], mode="test")
        print('Test: loss={:.3f} , acc={:5.1f}%'.format(test_loss,test_acc))

        # save accuracy
        jj = 0
        for ii in np.array(task_list)[0:task_id+1]:
            xtest = data[ii]['test']['x']
            ytest = data[ii]['test']['y']
            _, acc_matrix[task_id,jj] = test(args, model, device, xtest, ytest,criterion, jj, curr_task_masks=per_task_masks[jj], mode="test")
            jj +=1

        print('Accuracies =')
        for i_a in range(task_id+1):
            print('\t',end='')
            for j_a in range(i_a + 1):
                print('{:5.1f} '.format(acc_matrix[i_a,j_a]),end='')
            print()

        # update task id
        task_id +=1


    print('-'*40)
    # Simulation Results
    print ('Task Order : {}'.format(np.array(task_list)))
    print ('Diagonal Final Avg Accuracy: {:5.2f}%'.format( np.mean([acc_matrix[i,i] for i in range(len(taskcla))] )))
    print ('Final Avg accuracy: {:5.2f}%'.format( np.mean([acc_matrix[-1,i] for i in range(len(taskcla))] )))

    bwt=np.mean((acc_matrix[-1]-np.diag(acc_matrix))[:-1])
    print ('Backward transfer: {:5.2f}%'.format(bwt))
    print('[Elapsed time = {:.1f} ms]'.format((time.time()-tstart)*1000))
    print('-'*40)



if __name__ == "__main__":
    # Training parameters
    parser = argparse.ArgumentParser(description='Sequential PMNIST with GPM')
    parser.add_argument('--batch_size_train', type=int, default=10, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--batch_size_test', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--n_epochs', type=int, default=5, metavar='N',
                        help='number of training epochs/task (default: 200)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--n_tasks', type=int, default=10, metavar='S',
                        help='number of tasks (default: 5)')
    parser.add_argument('--pc_valid',default=0.05,type=float,
                        help='fraction of training data used for validation')
    # Optimizer parameters
    parser.add_argument('--optim', type=str, default="adam", metavar='OPTIM',
                        help='optimizer choice')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--lr_min', type=float, default=1e-6, metavar='LRM',
                        help='minimum lr rate (default: 1e-5)')
    parser.add_argument('--lr_patience', type=int, default=6, metavar='LRP',
                        help='hold before decaying lr (default: 6)')
    parser.add_argument('--lr_factor', type=int, default=2, metavar='LRF',
                        help='lr decay factor (default: 2)')
    # CUDA parameters
    parser.add_argument('--gpu', type=str, default="0", metavar='GPU',
                        help="GPU ID for single GPU training")
    # CSNB parameters
    parser.add_argument('--sparsity', type=float, default=0.2, metavar='SPARSITY',
                        help="Target current sparsity for each layer")

    # Model parameters
    parser.add_argument("--dataset", default='pmnist', type=str, help="Dataset to train and test on.")
    parser.add_argument("--n_hidden", default=100, type=int, help="Number of hidden units.")
    parser.add_argument('--name', type=str, default='hard')
    parser.add_argument('--soft', type=float, default=0.0)
    parser.add_argument('--soft_grad', type=float, default=1.0)

    #ETIL parameters
    parser.add_argument("--sim_threshold", default=2.5, type=float, help="Threshold for ETIL to determine similar tasks.")
    parser.add_argument("--k_threshold", default=0.999, type=float, help="Threshold for choose Orthogonal projection bases of SVD.")
    parser.add_argument("--sample_size", default=300, type=int, help="Sample size for ETIL.")

    args = parser.parse_args()
    args.sparsity = 1 - args.sparsity

    print('='*100)
    print('Arguments =')
    for arg in vars(args):
        print('\t'+arg+':',getattr(args,arg))
    print('='*100)

    main(args)