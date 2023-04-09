from api_module import *


spotify_id = ' '
spotify_pw = ' '

musixmatch_id = musixmatch_jh = Musixmatch(' ')

client_credentials_manager = SpotifyClientCredentials(client_id=spotify_id, client_secret=spotify_pw)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

playlist_id = [' ', ' ']

creator = 'spotify'


playlists_concat = concat_playlist(playlist_id, creator)

lyric_df = lyrics(playlists_concat, musixmatch_id)


music_to_csv(playlists_concat, lyric_df)
