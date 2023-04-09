from preprocess_module import remove, word_find_in_lexicon
from tqdm import tqdm

import os
import pandas as pd
import numpy as np

import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords



file_path = os.path.join(os.getcwd())

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

lyric = pd.read_csv(os.path.join(file_path, 'song_lyric.csv'))

lyric['cleaned'] = remove(lyric['lyrics_body'])._removeNonAscii
lyric['cleaned'] = remove().lemmatization
lyric['lyrics_word_lst'] = lyric['cleaned'].apply(lambda x:x.split(' '))

lexicon=pd.read_table(os.path.join(file_path, 'NRC-VAD-Lexicon.txt'), sep='\t') #원파일 VAD 순서대로
lexicon['word']=lexicon['word        valence        arousal        domain'].apply(lambda x:x.split('        ')[0])
lexicon['valence']=lexicon['word        valence        arousal        domain'].apply(lambda x:x.split('        ')[1])
lexicon['arousal']=lexicon['word        valence        arousal        domain'].apply(lambda x:x.split('        ')[2])
lexicon=lexicon.drop(['word        valence        arousal        domain'],axis=1)
lexicon=lexicon.astype({'valence':'float'})
lexicon=lexicon.astype({'arousal':'float'})

lexicon_word_lst=list(lexicon['word'])
lexicon_valence_lst=list(lexicon['valence'])
lexicon_arousal_lst=list(lexicon['arousal'])

all_valence_lst=[]
all_arousal_lst=[]
all_percent_lst=[]
for i in tqdm(range(len(lyric))):
  valence_lst, arousal_lst, not_use_percent= word_find_in_lexicon(lyric.iloc[i,15])
  all_valence_lst.append(valence_lst)
  all_arousal_lst.append(arousal_lst)
  all_percent_lst.append(not_use_percent)

lyric['lyric_word_valence_lst']=all_valence_lst
lyric['lyric_word_arousal_lst']=all_arousal_lst
lyric['not_use_percent']=all_percent_lst

lyric['valence_mean']=lyric['lyric_word_valence_lst'].apply(lambda x:np.mean(x))
lyric['arousal_mean']=lyric['lyric_word_arousal_lst'].apply(lambda x:np.mean(x))
lyric=lyric.reset_index(drop=True)


lyric.to_csv(os.path.join(file_path, 'song_lyric_VA.csv'), index=0)