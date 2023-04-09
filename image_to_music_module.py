import torch
import torch.nn as nn
import torchvision
from torchvision import datsets, models, transforms

import time
import os
import copy

import torch.optim as optim
from torch.optim import lr_scheduler
from glob import glob0
from torch.utils.data import DataLoader, Dataset

import random
from PIL import Image
import shutil

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

def load(dir, net, optim):
    dict_model = torch.load(dir)
    net.laod_state_dict(dict_model['net'])
    optim.load_state_dict(dict_model['optim'])

    return net, optim


def test_data(image_lst, transforms, result_dir, name):
    test_lst = []
    for i in image_lst:
        result = i.split('/')
        result = result[7]
        result = result.split('.')
        result = result[0]
        test_lst.append(result)

    test_dataset = Test_Dataset(image_lst, transforms)
    test_loader = DataLoader(
        test_dataset, batch_size=len(test_lst), shuffle=False
    )
    outputs_lst = []
    for inputs in test_laoder:
        # input = inputs[0].to(device)
        inputs = inputs.to(device)
        outputs = model(inputs)
        outputs_lst.append(outputs)
    torch.save(outputs_lst, os.path.join(result_dir, name))

    return outputs_lst
    


class Test_Dataset(Dataset):
    
    def __init__(self, img_path_lst, transforms=data_transforms['train']):
        self.img_path_lst = img_path_lst
        self.transforms = transforms
        self.img_lst = []
        for img_path in self.img_path_lst:
            self.img_lst.append(Image.open(img_path).convert('RGB'))

    def __len__(self):
        return len(self.img_path_lst)
    
    def __getitem__(self, idx):
        Image = self.img_lst[idx]
        Image = self.transforms(Image)

        return Image
    

def image_to_music(tensor_list, song_csv_path, top_k):
    song = pd.read_csv(song_csv_path, index_col=0)
    song_vec = np.array(song[['normalized_valence', 'normalized_arousal']])

    #tensor to numpy
    image_vec = tensor_list[0].detach().cpu().numpy()
    similarity = 1/(1 + euclidean_distances(image_vec, song_vec))

    for i in range(0, len(image_vec)):
        sim_list = list(enumerate(similarity[i]))
        sim_list = sorted(sim_list, key=lambda x:x[1], reverse = True)
        top_k_list = sim_list[0:top_k-1]
        tag_indices, score = [i[0] for i in top_k_list], [i[1] for i in top_k_list]
        print('image',i+1,test_image_lst[i].split('/')[-1])
        for k in range(0, len(tag_indices)):
            print('top',k,'track name:',song.iloc[tag_indices[k]]['track'],'artist', song.iloc[tag_indices[k]]['artist'], 'valence: ',song.iloc[tag_indices[k]]['normalized_valence'], 'arousal: ',song.iloc[tag_indices[k]]['normalized_arousal'])
            print(score[k])
        print('-'*50)
