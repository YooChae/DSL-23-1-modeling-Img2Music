import torch
import torch.nn as nn
import numpy as np
import time
import os
import copy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_model_together(dataloaders, dataset_sizes, model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time() # 시작시점 기록

    # 모델의 weight(가중치)를 깊은 복사하여, 이 변수가 변해도 해당 모델의 weight가 변하지 않도록 한다.
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = np.inf

    for epoch in range(num_epochs): # epochs만큼 반복
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train','test']: # train -> validation 순차 진행
            if phase == 'train':
                model.train() # train일 때는, layer의 batch-normal과 drop-out 등 을 활성화시킨다.
            else:
                model.eval() # validation(=evaluation) 때는, layer의 batch-normal과 drop-out 등 을 비활성화시킨다.
                # 즉, 이전에 train에서 정해진 값들로 정해지고, update는 없다.

            running_loss = 0.0
            running_corrects = 0
            running_corrects_valence = 0
            running_corrects_arousal = 0

            for inputs, valence, arousal, valence_label, arousal_label in dataloaders[phase]: # batch마다
                inputs = inputs.to(device) # gpu위에 올린다; [4, 3, 224, 224]
                arousal = arousal.type(torch.float32)
                arousal = arousal.unsqueeze(-1)
                arousal = arousal.to(device) # gpu위에 올린다; [4]
                valence = valence.type(torch.float32)
                valence = valence.unsqueeze(-1)
                valence = valence.to(device) 

                vector = torch.cat([valence, arousal], dim=1)
                
                valence_label = valence_label.unsqueeze(-1).to(device)
                arousal_label = arousal_label.unsqueeze(-1).to(device)

                true = torch.cat([valence_label, arousal_label], dim=1).to(device)

                optimizer.zero_grad() # batch마다 gradient 0으로 초기화

                with torch.set_grad_enabled(phase == 'train'): # train일 때, True로 새로 정의된 변수들의 연산들이 추적되어 gradient가 계산된다.
                    outputs = model(inputs) # 실수값
                    preds = outputs

                    labels_valence = torch.tensor([1 if i >= 0.5 else 0 for i in outputs[:,0]])
                    labels_arousal = torch.tensor([1 if i >= 0.5 else 0 for i in outputs[:,1]])
                    labels_valence = labels_valence.unsqueeze(-1).to(device)
                    labels_arousal = labels_arousal.unsqueeze(-1).to(device)
                    labels = torch.cat([labels_valence, labels_arousal], dim=1).to(device)  
                    # print(labels.shape)
                    # print(outputs.shape)
                    # print(outputs.dtype)
                    # print(outputs)
                    loss = criterion(outputs, vector) # loss-function

                    if phase == 'train':
                        loss.backward() # back-propagation; gradient 계산
                        optimizer.step() # parameters(weights) update
                
                running_loss += loss.item() * inputs.size(0) # inputs.size(0) = batch-size; 4
                running_corrects += torch.sum(true == labels.data)
                running_corrects_valence += torch.sum(true[:,0] == labels_valence.data)
                running_corrects_arousal += torch.sum(true[:,1] == labels_arousal.data)
            if phase == 'train':
                scheduler.step()
            
            epoch_loss = running_loss / dataset_sizes[phase]
            # epoch_acc = running_corrects.double() / dataset_sizes[phase]
            # epoch_acc_valence = running_corrects_valence.double() / dataset_sizes[phase]
            # epoch_acc_arousal = running_corrects_arousal.double() / dataset_sizes[phase]

            # print('{} Loss: {:.4f}\t Acc: {:.4f}\t Acc_Valence: {:.4f}\t Acc_Arousal: {:.4f}' \
            # .format(phase, epoch_loss, epoch_acc, epoch_acc_valence, epoch_acc_arousal))
            
            print('{} Loss: {:.4f}\t '.format(phase, epoch_loss))
            # validation의 목적; 이전에 train 시킨 후에, 정해진 각 epochs마다 모델 성능을 비교하여, 가능 좋은 값을 찾아낸다.
            if phase == 'test' and epoch_loss < best_loss:
                # best_acc = epoch_acc
                # best_acc_valence = epoch_acc_valence
                # best_acc_arousal = epoch_acc_arousal
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
        
        print()

    time_elapsed = time.time() - since
    print('Training time: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best Validation Loss: {:.4f}\t '.format(best_loss))
    # print('Best Validation Loss: {:.4f}\t Best Acc: {:.4f}\n Best_Acc_Valence: {:.4f}\t Best_Acc_Arousal: {:.4f}'.format(best_loss, best_acc, best_acc_valence, best_acc_arousal))

    model.load_state_dict(best_model_wts) # 가장 좋은 성능을 보인 가중치로 바꿔주기
    return model

