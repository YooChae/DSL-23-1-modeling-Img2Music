from preprocess_module import min_max_normalize, log_normalize

import os
import pandas as pd

file_path = os.path.join(os.getcwd())

song_df=pd.read_csv(os.path.join(file_path, 'song_with_word2vec.csv'), index_col=0)

#original
music_x=song_df.loc[:,['danceability','key','loudness','mode','speechiness','instrumentalness','liveness','tempo']]

#after scaling
log_columns=['speechiness','instrumentalness','liveness']
log_normalize(music_x, log_columns)


columns=['danceability','key','loudness','mode','speechiness','instrumentalness','liveness','tempo']
for i in columns:
  music_x[i]=min_max_normalize(music_x[i]) #minmaxscaling

music_x.to_csv(os.path.join(file_path, 'song_music_8_characteristics.csv',index=0))