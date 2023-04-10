import pandas as pd
import numpy as np
import os

import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

import gensim
from gensim.models import Word2Vec
from gensim.models import KeyedVectors

file_path = os.path.join(os.getcwd())

class remove():

    def __init__(self, s, text, stopwords):
        self.s = s
        self.text = text
        self.stopwords = nltk.download('stopwords')

    def _removeNonAscii(self):
        return "".join(i for i in self.s if ord(i)<128)

    def lemmatization(self):

        #make lower case
        self.text =  self.text.lower()

        # remove_stop_words
        self.text = self.text.split()
        stops = set(self.stopwords.words("english"))
        self.text = [w for w in self.text if not w in stops]
        self.text = " ".join(self.text)

        # remove punctuation
        tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
        self.text = tokenizer.tokenize(self.text)
        self.text = " ".join(self.text)

        #lemmatization
        lemmatizer = WordNetLemmatizer()
        words = word_tokenize(self.text)
        words = [lemmatizer.lemmatize(word.lower()) for word in words]
        self.text = ' '.join(words)
        return self.text

## For lyric_to_vector
class doc_vec():

    def __init__(self, corpus, googleNews, TranferedModel):
        self.corpus = corpus
        self.TranferedModel = TranferedModel
        self.googleNews = os.path.join(file_path, 'models', 'GoogleNews-vectors-negative300.bin')
    
    def make_corpus(document_list):
        corpus = []
        for words in document_list:
            corpus.append(words.split())
        
        return corpus

    def model_train(self, corpus):
        PreTrainedKeyvector = gensim.models.keyedvectors.KeyedVectors.load_word2vec_format(self.googleNews, binary=True, limit=20)

        self.TranferedModel = gensim.models.Word2Vec(vector_size=PreTrainedKeyvector.vector_size, min_count=1)

        self.TranferedModel.build_vocab([list(PreTrainedKeyvector.key_to_index.keys())])
        self.TranferedModel.build_vocab(corpus, update=True)
        self.TranferedModel.wv.vectors_lockf=np.ones(len(self.TranferedModel.wv),dtype=np.float32)

        self.TranferedModel.wv.intersect_word2vec_format(self.googleNews, binary=True, lockf=1.0)

        self.TranferedModel.train(corpus, total_examples=len(corpus), epochs=100)

        return self.TranferedModel


    def get_document_vectors(self, document_list):

        document_embedding_list = []

        # 각 문서에 대해서
        for line in document_list:
            doc2vec = None
            count = 0
            for word in line.split():
                if word in self.TranferedModel.wv.key_to_index:
                    count += 1
                    #해당 문서에 있는 모든 단어들의 벡터값 더하기
                    if doc2vec is None:
                        doc2vec = self.TranferedModel.wv[word]
                    else:
                        doc2vec = doc2vec + self.TranferedModel.wv[word]

            if doc2vec is not None:
                doc2vec = doc2vec / count
                document_embedding_list.append(doc2vec)

        return document_embedding_list


class word2vec_csv():

    def __init__(self, df, document_embedding_list):
        self.df = df
        self.document_embdding_list = document_embedding_list

    def word2vec_to_csv(df, document_embedding_list):

        for i in range(0,300):
            new_column = []
            for row in document_embedding_list:
                new_column.append(row[i])
            column_name = 'doc_vecotr_'+str(i)
            df[column_name] = new_column
        
        df.to_csv("song_with_word2vec.csv", index=False)


##For lyric_VA
def word_find_in_lexicon(word_lst):
    valence_lst = []
    arousal_lst = []
    unknown_word_num = 0
    for i in range(len(word_lst)):
        if word_lst[i] in lexion_word_lst:
            idx = lexicon_word.index(word_lst[i])
            valence_lst.append(lexcion_valence_lst[idx])
            arousal_lst.append(lexicon_arousal_lst[idx])
        else:
            unknown_word_num+=1
        
    return valence_lst, arousal_lst, unknown_word_num/len(word_lst)
    #valence_lst, arousal_lst, percentage of unfounded words in lexicon


##For song_music_8_characteristics.csv
def min_max_normalize(lst):
    normalized = []

    for value in lst:
        normalized_num = (value - min(lst)) / (max(lst) - min(lst))
        normalized.append(normalized_num)

    return normalized

def log_normalize(df, log_columns):
    for i in log_columns:
        df[i] = df[i] + np.abs(np.min(df[i]))
        df[i] = np.log(df[i])

    return df[i]