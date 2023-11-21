import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances, cosine_similarity

# Load the data
seed_pool_df = pd.read_csv("spotify_seed_tracks.csv")
rec_pool_df = pd.read_csv("top_200_tracks.csv")

# Feature columns
feature_cols = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness',
                'liveness', 'valence', 'tempo']
dist_feature_cols = ['euclidean_dist_features','manhattan_dist_features', 'cosine_dist_features']

def get_distances(x,y):
    euclidean_dist =  euclidean_distances(x.values.reshape(1, -1), y.values.reshape(1, -1)).flatten()[0]
    manhattan_dist =  manhattan_distances(x.values.reshape(1, -1), y.values.reshape(1, -1)).flatten()[0]
    cosine_dist = 1 - cosine_similarity(x.values.reshape(1, -1), y.values.reshape(1, -1)).flatten()[0]
    return [euclidean_dist,manhattan_dist,cosine_dist]

def get_reco_songs(seed_track_data, rec_pool_df=rec_pool_df):
    rec_pool_df['all_distances_features'] = rec_pool_df.apply(lambda x: get_distances(x[feature_cols],seed_track_data[feature_cols]), axis=1)
    rec_pool_df[dist_feature_cols] = rec_pool_df['all_distances_features'].apply(pd.Series)
    # recommendation_df = rec_pool_df[rec_pool_df['track_id']!=seed_track_data['track_id']].sort_values('cosine_dist_features')[:10]
    recommendation_df = rec_pool_df.sort_values('cosine_dist_features')[:10]
    top_10_reco_songs = recommendation_df[['track_name','artist_name']].reset_index()
    top_10_reco_songs = top_10_reco_songs.drop(columns = 'index')
    return top_10_reco_songs

# Page 1: Recommendation Engine - SB19 to Top 200
def page1():
    st.image('spotify_logo.png', width = 600)
    st.title('Spotify Recommender Engine for SB19 Songs')
    chosen_song = st.text_input('Input SB19 Song to get recommendations from:')
    try:
        seed_track_data = seed_pool_df[seed_pool_df['track_name']==chosen_song].iloc[0]
        st.dataframe(data=get_reco_songs(seed_track_data), width = 700)
    except:
        st.error("That is not a valid SB19 song. Please try again!") 

# Page 2: Recommendation Engine - Top 200 to SB19
def page2():
    st.image('spotify_logo.png', width = 600)
    st.title('Spotify Recommender Engine for Top 200 Songs')
    chosen_song = st.text_input('Input Top 200 song to get SB19 song recommendations from:')
    seed_pool_df = pd.read_csv("top_200_tracks.csv")
    rec_pool_df = pd.read_csv("spotify_seed_tracks.csv")
    try:
        seed_track_data = seed_pool_df[seed_pool_df['track_name']==chosen_song].iloc[0]
        st.dataframe(data=get_reco_songs(seed_track_data, rec_pool_df=rec_pool_df), width = 700)
    except:
        st.error("That is not a valid Top 200 song. Please try again!")

# Page 3: Recommendation Engine - Audio Features to SB19
def page3():
    st.image('spotify_logo.png', width = 600)
    st.title('Spotify Recommender Engine - Audio Features to SB19')
    seed_pool_df = pd.read_csv("spotify_seed_tracks.csv")

    feature_values = {feature: st.slider(f'{feature}', float(0.0), float(round(seed_pool_df[feature].max())), float(seed_pool_df[feature].mean())) for feature in feature_cols}

    user_input_data = pd.DataFrame([feature_values])

    if st.button('Generate Recommendations'):
        st.dataframe(data=get_reco_songs(user_input_data.iloc[0], rec_pool_df=seed_pool_df).head(), width = 700)
    

def main():
    page = st.sidebar.radio("Choose a page:", ("Recommendation Engine - SB19 to Top 200 Songs", "Recommendation Engine - Top 200 Songs to SB19", "Recommendation Engine - Audio Features to SB19"))
    if page == "Recommendation Engine - SB19 to Top 200 Songs":
        page1()
    elif page == "Recommendation Engine - Top 200 Songs to SB19":
        page2()
    else:
        page3()


if __name__ == "__main__":
    main()
