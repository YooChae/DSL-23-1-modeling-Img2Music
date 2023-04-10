from utils import *
import torch
from torch.utils.data import DataLoader

from glob import glob
from PIL import Image
import os
import pandas as pd
from torchvision import transforms

def dataloader(data_path):

    Images_path_lst=glob(os.path.join(data_path, 'images/*'))
    oasis = pd.read_csv(os.path.join(data_path, 'OASIS.csv'))
    oasis['normalized_valence']=min_max_normalize(oasis['Valence_mean'])
    oasis['normalized_arousal']=min_max_normalize(oasis['Arousal_mean'])
    oasis['valence_classification']=reg_to_classification(oasis['normalized_valence'])
    oasis['arousal_classification']=reg_to_classification(oasis['normalized_arousal'])

#이 순서대로 label도 필요하니까
    train_image_lst=glob(os.path.join(data_path, '/train/*'))
    test_image_lst=glob(os.path.join(data_path, '/test/*'))
    
    train_lst=[]
    for i in train_image_lst:
        result=i.split('/')
        result=result[7]
        result=result.split('.')
        result=result[0]
        train_lst.append(result)

    test_lst=[]
    for i in test_image_lst:
        result=i.split('/')
        result=result[7]
        result=result.split('.')
        result=result[0]
        test_lst.append(result)

#train_image_lst는 train시킬 이미지들 path
#train_lst는 이미지들 파일명
#train, test 폴더 안에 들어 있는 파일 순서대로 train_label(valence, arousal), test_label 만들어야함
    train_valence_lst=[]
    train_arousal_lst=[]
    test_valence_lst=[]
    test_arousal_lst=[]

    train_val_clf_lst=[]
    train_aro_clf_lst=[]
    test_val_clf_lst=[]
    test_aro_clf_lst=[]

    theme=list(oasis['Theme'])
    valence=list(oasis['normalized_valence'])
    arousal=list(oasis['normalized_arousal'])
    valence_clf=list(oasis['valence_classification'])
    arousal_clf=list(oasis['arousal_classification'])

    for i in train_lst:
        idx=theme.index(i)
        train_valence_lst.append(valence[idx])
        train_arousal_lst.append(arousal[idx])
        train_val_clf_lst.append(valence_clf[idx])
        train_aro_clf_lst.append(arousal_clf[idx])
        

    for j in test_lst:
        idx=theme.index(i)
        test_valence_lst.append(valence[idx])
        test_arousal_lst.append(arousal[idx])
        test_val_clf_lst.append(valence_clf[idx])
        test_aro_clf_lst.append(arousal_clf[idx])

#train_image_lst는 train시킬 이미지들 path
#train_lst는 이미지들 파일명
#train_valence_lst는 train 폴더에 있는 이미지 순서대로 valence 라벨값
#train_arousal_lst는 train 폴더에 있는 이미지 순서대로 arousal 라벨값
#train_val_clf_lst는 train 폴더에 있는 이미지 순서대로 valence 0,1 분류된 값
# train / val 에 대한 transform(전처리)를 각각 정의한 딕셔너리 만들기

    data_transforms = {'train': transforms.Compose( [transforms.RandomResizedCrop(224), # 224로 바꿔준다.
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])] ), # rgb임으로 3개씩; 대중적으로 널리 알려진 값
                    'val': transforms.Compose( [transforms.Resize(256),
                                                transforms.CenterCrop(224), # 최종적으로 train과 같이 224로 바꿔준다.
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])] ) # 위와 동일
                    }

    train_dataset=Dataset(train_image_lst, train_valence_lst, train_arousal_lst, train_val_clf_lst, train_aro_clf_lst, data_transforms['train'])
    test_dataset=Dataset(test_image_lst, test_valence_lst, test_arousal_lst, test_val_clf_lst, test_aro_clf_lst, data_transforms['val'])
    if len(train_dataset) <=0:
      train_dataset = torch.load(os.path.join(data_path, "train_dataset.pt"))
      test_dataset = torch.load(os.path.join(data_path, "test_dataset.pt"))

    train_loader=DataLoader(
        train_dataset, batch_size=32, shuffle=True
    )

    test_loader=DataLoader(
        test_dataset, batch_size=32, shuffle=False
    )
    dataloaders = {'train':train_loader, 'test': test_loader}
    dataset_sizes = {'train':len(train_dataset), 'test':len(test_dataset)}
    
    return dataloaders, dataset_sizes