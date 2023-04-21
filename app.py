import numpy as np
import streamlit as st
import pandas as pd
import pickle

st.write(''' # Board Game Rating Prediction App''')

st.sidebar.header('User Input Parameters')

def user_input_features():
  Minimum_Players = st.sidebar.slider('Min Players', 1, 20, 2)
  Maximum_Players = st.sidebar.slider('Max Players', 1, 20, 4)
  Play_time = st.sidebar.slider('Play Time (minutes)', 0, 180, 60)
  Min_age = st.sidebar.slider('Min Player Age', 0, 100 ,8)
  Game_Genre =st.sidebar.multiselect("Game Category",['Strategy Games', 'Thematic Games', 'Wargames', 'Family Games', 'Customizable Games', 'Abstract Games', 'Party Games', "Children's Games"],max_selections=4)
  complexity=st.sidebar.slider('Complexity', 0, 5, 2)

  user_input_data = {'Min Players': Minimum_Players,
               'Max Players': Maximum_Players,
               'Play Time': Play_time,
               'Min Age': Min_age,
               'Game_Genre': [Game_Genre],
               'Complexity Average':complexity}


  features = pd.DataFrame(user_input_data, index=[0])

  return features

features = user_input_features()


for i in ['Strategy Games', 'Thematic Games', 'Wargames', 'Family Games', 'Customizable Games', 'Abstract Games', 'Party Games', "Children's Games","nan"]:
  if i in features['Game_Genre']:
    features[i]=1
  else:
    features[i]=0

features["BG Age_Ancient"]=0
features["BG Age_Vintage"]=0
features["BG Age_Modern"]=1
features["BG Age_Historical"]=0
features.drop(columns=['Game_Genre'],inplace=True, errors='ignore')

start_model=st.button("Start Model")
if start_model:
  with open('board_game_rating.pkl', 'rb') as f:
    model = pickle.load(f)

  pred = model.predict(features)

  st.subheader('User Input Parameters')
  st.write(features)

  st.subheader('Predicted Rating out of 10')
  st.write(pred)
  st.write("Mean Squared Error for this model is 0.62")
