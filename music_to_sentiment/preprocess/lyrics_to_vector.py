from preprocess_module import remove, doc_vec, word2vec_csv

import os
import pandas as pd

import nltk

file_path = os.path.join(os.getcwd())

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')


song_lyric = pd.read_csv(os.path.join(file_path, 'song_lyric.csv'), index_col=0)

song_lyric = song_lyric.dropna(axis=0)

song_lyric['cleaned'] = remove(song_lyric['lyrics_body'])._removeNonAscii
song_lyric['cleaned'] = remove().lemmatization


df = song_lyric[['artist','album','track','danceability','energy','key','loudness','mode','speechiness','instrumentalness','liveness','valence','tempo','lyrics_body','cleaned']]


corpus = doc_vec.make_corpus(df['cleaned'])

model = doc_vec.model_train(corpus)

document_embedding_list = doc_vec.get_document_vectors(model, df['cleaned'])


word2vec_csv.word2vec_to_csv(document_embedding_list)
