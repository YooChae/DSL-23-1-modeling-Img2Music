import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

from pycaret.regression import *

file_path = os.path.join(os.getcwd())

X1 = pd.read_csv(os.path.join(file_path, 'song_music_8_characteristics.csv'))
X2 = pd.read_csv(os.path.join(file_path, 'song_lyric_embedding_pca_11.csv'))
X3 = pd.read_csv(os.path.join(file_path, 'song_lyric_VA.csv')).loc[:,['valence_mean','arousal_mean']]
y = pd.read_csv(os.path.join(file_path, 'song_normalized_VA_label.csv'))

X_arousal = pd.concat([X1,X3],axis=1)
X_valence = pd.concat([X1,X2,X3],axis=1)

X_arousal_train, X_arousal_test, y_arousal_train, y_arousal_test = train_test_split(X_arousal, y, test_size=0.2, random_state=42)
X_valence_train, X_valence_test, y_valence_train, y_valence_test = train_test_split(X_valence, y, test_size=0.2, random_state=42)


X_arousal_train = X_arousal_train.reset_index(drop=True)
X_arousal_test = X_arousal_test.reset_index(drop=True)
y_arousal_train = y_arousal_train.reset_index(drop=True)
y_arousal_test = y_arousal_test.reset_index(drop=True)

X_valence_train = X_valence_train.reset_index(drop=True)
X_valence_test = X_valence_test.reset_index(drop=True)
y_valence_train = y_valence_train.reset_index(drop=True)
y_valence_test = y_valence_test.reset_index(drop=True)

arousal_train = pd.concat([X_arousal_train, y_arousal_train]).drop(['normalized_valence'],axis=1)
valence_train = pd.concat([X_valence_train, y_valence_train]).drop(['normalized_arousal'], axis=1)
arousal_test = pd.concat([X_arousal_test, y_arousal_test]).drop(['normalized_valence'],axis=1)
valence_test = pd.concat([X_valence_test, y_valence_test]).drop(['normalized_arousal'], axis=1)

#arousal prediction
setup = setup(data=arousal_train, target='normalized_arousal',
              fold_strategy = 'kfold', fold=5,
              categorical_features = ['mode'],
              use_gpu = True,
              session_id = 777)

model = compare_models(sort='mse', n_select=3)

tuned_model = [tune_model(i,choose_better = True, optimize='mse',fold=5) for i in model]

stacked_model=stack_models(tuned_model, optimize='mse',fold=5, meta_model=tuned_model[0]) #meta=gbr

prediction_train=predict_model(stacked_model, data=X_train) 
compare_train=pd.concat([prediction_train, y_train['normalized_arousal']], axis=1)
compare_train['prediction_binary']=compare_train['prediction_label'].apply(lambda x:0 if x<0.5 else 1)
compare_train['real_binary']=compare_train['normalized_arousal'].apply(lambda x:0 if x<0.5 else 1)
accuracy_train = accuracy_score(compare_train['real_binary'], compare_train['prediction_binary'])
mse_train=mean_squared_error(compare_train['normalized_arousal'],compare_train['prediction_label'])


prediction_test=predict_model(stacked_model, data=X_test) 
compare_test=pd.concat([prediction_test, y_test['normalized_arousal']], axis=1)
compare_test['prediction_binary']=compare_test['prediction_label'].apply(lambda x:0 if x<0.5 else 1)
compare_test['real_binary']=compare_test['normalized_arousal'].apply(lambda x:0 if x<0.5 else 1)
accuracy_test = accuracy_score(compare_test['real_binary'], compare_test['prediction_binary'])
mse_test=mean_squared_error(compare_test['normalized_arousal'],compare_test['prediction_label'])


print('train mse : ', mse_train, 'train_accuracy : ', accuracy_train,
      'test_mse : ', mse_test, 'test_accuracy : ', accuracy_test)




#valence_prediction
# setup = setup(data=valence_train, target='normalized_valence',
#               fold_strategy = 'kfold', fold=5,
#               categorical_features = ['mode'],
#               use_gpu = True,
#               session_id = 777)

# model = compare_models(sort='mse', n_select=3)

# tuned_model = [tune_model(i,choose_better = True, optimize='mse', fold=5) for i in model]

# stacked_model = stack_models(tuned_model, optimize='mse', fold=5, meta_model=tuned_model[2])


# prediction_train=predict_model(stacked_model, data=X_train)
# compare_train = pd.concat([prediction_train, y_train['normalized_valence']], axis=1)
# compare_train['prediction_binary'] = compare_train['prediction_label'].apply(lambda x:0 if x<0.5 else 1)
# compare_train['real_binary'] = compare_train['normalized_valence'].apply(lambda x: 0 if x<0.5 else 1)
# accuracy_train = accuracy_score(compare_train['normalized_valence'],compare_train['prediction_label'])
# mse_train = mean_squared_error(compare_train['normalized_valence'])

# prediction_test = predict_model(stacked_model, data=X_test)
# compare_test = pd.concat([prediction_test, y_test['normalized_valence']], axis=1)
# compare_test['prediction_binary'] = compare_test['prediction_label'].apply(lambda x:0 if x<0.5 else 1)
# compare_test['real_binary'] = compare_test['normalized_valence'].apply(lambda x:0 if x<0.5 else 1)
# accuracy_test = accuracy_score(compare_test['real_binary'], compare_test['prediciton_binary'])
# mse_test = mean_squared_error(compare_test['normalized_valence'], compare_test['prediction_label'])


# print('train mse : ', mse_train, 'train_accuracy : ', accuracy_train,
#       'test_mse : ', mse_test, 'test_accuracy : ', accuracy_test)


save_model(stacked_model, os.path.join(file_path))