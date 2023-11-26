import os, sys
import numpy as np
import torch
from torchvision import datasets, transforms
from sklearn.utils import shuffle
from torchvision import datasets,transforms
import json
from torch.utils.data import Dataset
from sklearn.utils import shuffle
from PIL import Image
from collections import Counter

def get(seed=0, pc_valid=0,idrandom=3, n_tasks=10):
    size=[1,28,28]

    data = {}
    taskcla = []

    data_femnist, taskcla_femnist, size_femnist = read_femnist(seed=seed,args=0, pc_valid=pc_valid, n_tasks=n_tasks)
    if n_tasks == 35:
        all_femnist = [data_femnist[x]['name'] for x in range(n_tasks+1)]
    else:
        all_femnist = [data_femnist[x]['name'] for x in range(n_tasks)]
    
    print("all femnist:", all_femnist)

    f_name = 'mixemnist_random_'+str(n_tasks*2)

    with open(f_name,'r') as f_random_seq:
        random_sep = f_random_seq.readlines()[idrandom].split()
    print("random_sep:" ,random_sep)

    cnt = 0
    if n_tasks == 10:
        num = 20
    elif n_tasks == 35:
        num = 35
    for task_id in range(num):
        if 'fe-mnist'in random_sep[task_id]:
            femnist_id = all_femnist.index(random_sep[task_id])
            data[cnt] = data_femnist[femnist_id]
            taskcla.append((cnt,data_femnist[femnist_id]['ncla']))
            cnt += 1
    print(taskcla)
    return data,taskcla,size

def read_femnist(seed=0,fixed_order=False,pc_valid=0.10,remain=0,args=0, n_tasks=10):

    print('Read FEMNIST')
    data={}
    taskcla=[]
    size=[3,32,32]
    class_per_task = 62

    # MNIST
    mean=(0.1307,)
    std=(0.3081,)
    dat={}

    data_type = 'small'

    if n_tasks == 10:
        train_dataset = FEMMNISTTrain(root_dir='./data/femnist/'+data_type+'/iid/train10/',
                                    transform=transforms.Compose([transforms.Pad(padding=2,fill=0), transforms.ToTensor(),transforms.Normalize(mean,std)]))
        dat['train'] = train_dataset

        test_dataset = FEMMNISTTest(root_dir='./data/femnist/'+data_type+'/iid/test10/',
                                    transform=transforms.Compose([transforms.Pad(padding=2,fill=0), transforms.ToTensor(),transforms.Normalize(mean,std)]))
        dat['test'] = test_dataset
    elif n_tasks == 35:
        train_dataset = FEMMNISTTrain(root_dir='./data/femnist/'+data_type+'/iid/train35/',
                                    transform=transforms.Compose([transforms.Pad(padding=2,fill=0), transforms.ToTensor(),transforms.Normalize(mean,std)]))
        dat['train'] = train_dataset

        test_dataset = FEMMNISTTest(root_dir='./data/femnist/'+data_type+'/iid/test35/',
                                    transform=transforms.Compose([transforms.Pad(padding=2,fill=0), transforms.ToTensor(),transforms.Normalize(mean,std)]))
        dat['test'] = test_dataset

    users = [x[0] for x in set([user for user,image,target in torch.utils.data.DataLoader(dat['train'],batch_size=1,shuffle=True)])]
    users.sort()
    print('users: ',users)
    print('users length: ',len(users))
    # # totally 47 classes, each tasks 5 classes
    #
    for task_id,user in enumerate(users):
        data[task_id]={}
        data[task_id]['name'] = 'fe-mnist-'+str(user)
        data[task_id]['ncla'] = 62

    training_c = 0
    testing_c = 0

    for s in ['train','test']:
        print('s: ',s)
        loader=torch.utils.data.DataLoader(dat[s],batch_size=1,shuffle=True)

        for task_id,user in enumerate(users):
            data[task_id][s]={'x': [],'y': []}

        count_label = []
        for user,image,target in loader:

            label=target.numpy()[0]

            data[users.index(user[0])][s]['x'].append(image)
            data[users.index(user[0])][s]['y'].append(label)
            count_label.append(label)

        print('count: ',Counter(count_label))

        # print('testing_c: ',testing_c)
    print('training len: ',sum([len(value['train']['x']) for key, value in data.items()]))
    print('testing len: ',sum([len(value['test']['x']) for key, value in data.items()]))


    # # "Unify" and save
    for n,user in enumerate(users):
        for s in ['train','test']:
            data[n][s]['x']=torch.stack(data[n][s]['x']).view(-1,size[0],size[1],size[2])
            data[n][s]['y']=torch.LongTensor(np.array(data[n][s]['y'],dtype=int)).view(-1)

    # Validation
    for t in data.keys():
        r=np.arange(data[t]['train']['x'].size(0))
        r=np.array(shuffle(r,random_state=seed),dtype=int)
        # print('len(r): ',len(r))
        nvalid=int(pc_valid*len(r))
        ivalid=torch.LongTensor(r[:nvalid])
        itrain=torch.LongTensor(r[nvalid:])
        data[t]['valid']={}
        data[t]['valid']['x']=data[t]['train']['x'][ivalid].clone()
        data[t]['valid']['y']=data[t]['train']['y'][ivalid].clone()
        data[t]['train']['x']=data[t]['train']['x'][itrain].clone()
        data[t]['train']['y']=data[t]['train']['y'][itrain].clone()


    return data,taskcla,size





########################################################################################################################

# customize dataset class

class FEMMNISTTrain(Dataset):
    """Federated EMNIST dataset."""

    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.size=[3,32,32]

        self.x = []
        self.y = []
        self.user = []
        for file in os.listdir(root_dir):
            with open(root_dir+file) as json_file:
                data = json.load(json_file) # read file and do whatever we need to do.
                for key, value in data['user_data'].items():
                    print(key)
                    for type, data in value.items():
                        if type == 'x':
                            temp = torch.from_numpy(np.array(data))
                            temp = temp.view(-1, 1, 28, 28)
                            temp_padded = torch.nn.functional.pad(temp, [2, 2, 2, 2], value=0)
                            temp_padded = temp_padded.expand(temp_padded.size(0),3,temp_padded.size(2),temp_padded.size(3))
                            self.x.append(temp_padded)
                        elif type == 'y':
                            self.y.append(data)

                    for _ in range(len(data)):
                        self.user.append(key)

        #number of class
        print(len(set([b for a in self.y for b in a])))
        #number of class
        self.x=torch.cat(self.x,0).view(-1,3,self.size[1],self.size[2])
        self.y=torch.LongTensor(np.array([d for f in self.y for d in f],dtype=int)).view(-1).numpy()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):

        user = self.user[idx]
        x = self.x[idx].float()
        y = self.y[idx]
        return user,x,y






class FEMMNISTTest(Dataset):
    """Federated EMNIST dataset."""

    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.size=[3,32,32]

        self.x = []
        self.y = []
        self.user = []
        for file in os.listdir(root_dir):
            with open(root_dir+file) as json_file:
                data = json.load(json_file) # read file and do whatever we need to do.
                for key, value in data['user_data'].items():
                    for type, data in value.items():
                        if type == 'x':
                            temp = torch.from_numpy(np.array(data))
                            temp = temp.view(-1, 1, 28, 28)
                            temp_padded = torch.nn.functional.pad(temp, [2, 2, 2, 2], value=0)
                            temp_padded = temp_padded.expand(temp_padded.size(0),3,temp_padded.size(2),temp_padded.size(3))
                            self.x.append(temp_padded)
                        elif type == 'y':
                            self.y.append(data)

                    for _ in range(len(data)):
                        self.user.append(key)

        self.x=torch.cat(self.x,0).view(-1,3, self.size[1],self.size[2])
        self.y=torch.LongTensor(np.array([d for f in self.y for d in f],dtype=int)).view(-1).numpy()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):

        user = self.user[idx]
        x = self.x[idx].float()
        y = self.y[idx]

        return user,x,y