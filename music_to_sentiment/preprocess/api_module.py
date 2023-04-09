import pandas as pd
import numpy as np
import os 

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

from musixmatch import Musixmatch


def concat_playlist(creator, playlist_id):
    playlists = []

    for url in playlist_id:
        playlist_features_list = ["artist","album","track_name",  "track_id","danceability","energy","key","loudness","mode", "speechiness","instrumentalness","liveness","valence","tempo", "duration_ms","time_signature"]

        playlist_df = pd.DataFrame(columns = playlist_features_list)

        playlist = sp.user_playlist_tracks(creator, playlist_id)["items"]
        for track in playlist:

            playlist_features = {}

            playlist_features["artist"] = track["track"]["album"]["artists"][0]["name"]
            playlist_features["album"] = track["track"]["album"]["name"]
            playlist_features["track_name"] = track["track"]["name"]
            playlist_features["track_id"] = track["track"]["id"]

            audio_features = sp.audio_features(playlist_features["track_id"])[0]
            for feature in playlist_features_list[4:]:
                playlist_features[feature] = audio_features[feature]

            track_df = pd.DataFrame(playlist_features, index = [0])
            playlist_df = pd.concat([playlist_df, track_df], ignore_index = True)

        playlists.append(playlist_df)

    playlists_concat = pd.concat(playlists, ignore_index = True)

    return playlists_concat



def lyrics(playlists_concat, musixmatch_id):

    artist_list = playlists_concat['artist'].values.tolist()
    track_list = playlists_concat['track_name'].values.tolist()

    track_zip = {x:y for x,y in zip(artist_list, track_list)}

    lyric_df = pd.DataFrame(columns=['artist','track_name','lyrics'])


    for artist, track in track_zip.items():

        lyric = musixmatch_id.matcher_lyrics_get(artist, track)['message']['body']

        lyrics_body = eval(lyric[11:-1])['lyrics_body']

        lyric_df.append(pd.DataFrame([[artist, track, lyrics_body]], columns=['artist','track_name','lyrics']), ignore_index=True)


    return lyric_df


def music_to_csv(playlists_concat, lyric_df):

    file_path = os.path.join(os.getcwd())

    music_concat = pd.merge(playlists_concat, lyric_df, 'right', on=['artist','track_name'])

    music_concat.drop_duplicates(subset=['artist','track_name'], inplace=True)
    
    music_concat.to_csv(os.path.join(file_path, 'song_lyric.csv'))