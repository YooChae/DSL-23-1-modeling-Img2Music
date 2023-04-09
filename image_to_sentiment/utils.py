import torch
from torch.utils.data import Dataset

from glob import glob
import random
from PIL import Image
import os
#min-max normalization (0-1 사이의 값으로)
def min_max_normalize(lst):
    normalized = []
    
    for value in lst:
        normalized_num = (value - min(lst)) / (max(lst) - min(lst))
        normalized.append(normalized_num)
    
    return normalized

def reg_to_classification(lst):
  classification=[]

  for value in lst:
    if value>=0.5: classification.append(1)
    else: classification.append(0)

  return classification

class Dataset(Dataset):
  def __init__(self, img_path_lst, valence_lst, arousal_lst, val_clf_lst, aro_clf_lst, transforms):
    self.img_path_lst=img_path_lst
    self.transforms=transforms
    self.img_lst=[]
    for img_path in self.img_path_lst:
      self.img_lst.append(Image.open(img_path).convert('RGB'))
    self.valence_lst=valence_lst
    self.arousal_lst=arousal_lst
    self.val_clf_lst=val_clf_lst
    self.aro_clf_lst=aro_clf_lst

  def __len__(self):
    return len(self.img_path_lst)

  def __getitem__(self, idx):
    Image=self.img_lst[idx]
    Image=self.transforms(Image)
    valence=self.valence_lst[idx]
    arousal=self.arousal_lst[idx]
    valence_clf=self.val_clf_lst[idx]
    arousal_clf=self.aro_clf_lst[idx]

    return Image, valence, arousal, valence_clf, arousal_clf

def save(ckpt_dir, net, optim, net_name):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    torch.save({'net':net.state_dict(), 'optim': optim.state_dict()},
               "./%s/model_%s.pth" % (ckpt_dir, net_name))
  
def load(ckpt_dir, net, optim):
    if not os.path.exists(ckpt_dir):
        epoch = 0
        return net, optim, epoch

    ckpt_lst = os.listdir(ckpt_dir)
    ckpt_lst.sort(key = lambda f: int(''.join(filter(str.isdigit, f))))

    dict_model = torch.load('./%s/%s'%(ckpt_dir, ckpt_lst[-1]))
    net.load_state_dict(dict_model['net'])
    optim.load_state_dict(dict_model['optim'])
    epoch = int(ckpt_lst[-1].split('epoch')[1].split('pth')[0])

    return net, optim, epoch
