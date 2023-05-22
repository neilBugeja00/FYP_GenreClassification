import streamlit as st
import numpy as np
import json
import csv
import sys
import numpy as np 
import pandas as pd
from pydub import AudioSegment
from presets import Preset
import librosa as librosa
import librosa.display
from tensorflow import keras
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.cm as cm

#=========================DECLARATIONS======================
sr = 22050

TOTAL_SAMPLES = 29*sr
NUM_SLICES = 3
SAMPLES_PER_SLICE = int(TOTAL_SAMPLES / NUM_SLICES)

csv.field_size_limit(sys.maxsize)
csv_detection_path = './Resources/song_detection.csv'

librosa_preset = Preset(librosa)
librosa_preset['sr'] = 22050

labels = ['classical', 'folk', 'hiphop', 'jazz', 'pop', 'rock']

enre_dict = {
        0 : "classical",
        1 : "folk",
        2 : "hiphop",
        3 : "jazz",
        4 : "pop",
        5 : "rock",
        }
#=========================METHODS======================


#Three extract relevant are required to snip the song in 3 parts
#and test the genre of every snippet

def csv_write_data(audio_path, csv_path):
    rows = []
    
    song, sr = librosa.load(audio_path, duration=29)
    
    for s in range(NUM_SLICES):
        start_sample = SAMPLES_PER_SLICE * s
        end_sample = start_sample + SAMPLES_PER_SLICE
                
                
        mfcc = librosa.feature.mfcc(y=song[start_sample:end_sample], sr=sr, n_mfcc=40)
        mfcc = mfcc.T

        rows.append([4,json.dumps(mfcc.tolist())])
    
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Write the header row
        writer.writerow(["genre", "mfcc"])

        # Write the values rows
        for row in rows:
            writer.writerow(row)     


#Read MFCC data from CSV
def csv_read_data(csv_path):
    # Load data from CSV file
    with open(csv_path, 'r') as csvfile:
        reader = csv.reader(csvfile)

        # Skip the header row
        next(reader)

        # Initialize lists to hold genre and MFCC data
        genres = []
        mfcc = []

        # Iterate over each row of the CSV file
        for row in reader:
            # Extract genre and MFCC data from the row
            genre = int(row[0])
            mfcc_data = json.loads(row[1])

            # Append genre and MFCC data to lists
            genres.append(genre)
            mfcc.append(mfcc_data)

    # Convert lists to numpy arrays
    X = np.array(mfcc)
    y = np.array(genres)

    return X, y

def save_snippets(wav_file):
    wav = AudioSegment.from_wav(wav_file)
    wav = wav[1000*0:1000*10]
    wav.export("Resources/extracted_0.wav", format='wav')
    
    wav = AudioSegment.from_wav(wav_file)
    wav = wav[1000*10:1000*20]
    wav.export("Resources/extracted_1.wav", format='wav')
    
    wav = AudioSegment.from_wav(wav_file)
    wav = wav[1000*20:1000*30]
    wav.export("Resources/extracted_2.wav", format='wav')

#Create mfcc of every snippet
def create_mfcc(wav_file_0, wav_file_1, wav_file_2):
    #First snippet
    y, sr = librosa.load(wav_file_0, duration=10)
    mfcc0 = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    fig, ax = plt.subplots(figsize=(10, 4))
    librosa.display.specshow(mfcc0,
                        x_axis='time',
                        sr=sr)
    plt.colorbar()
    plt.title('MFCC')
    plt.ylabel('MFCC Coefficients')
        
    FigureCanvasAgg(fig)
    plt.savefig('Resources/mfcc_0.png')
    
    #Second snippet
    y, sr = librosa.load(wav_file_1, duration=10)
    mfcc1 = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    fig, ax = plt.subplots(figsize=(10, 4))
    librosa.display.specshow(mfcc1,
                        x_axis='time',
                        sr=sr)
    plt.colorbar()
    plt.title('MFCC')
    plt.ylabel('MFCC Coefficients')
        
    FigureCanvasAgg(fig)
    plt.savefig('Resources/mfcc_1.png')
    
    #First snippet
    y, sr = librosa.load(wav_file_2, duration=10)
    mfcc2 = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    #fig = plt.Figure()
    fig, ax = plt.subplots(figsize=(10, 4))
    librosa.display.specshow(mfcc2,
                        x_axis='time',
                        sr=sr)
    plt.colorbar()
    plt.title('MFCC')
    plt.ylabel('MFCC Coefficients')
        
    FigureCanvasAgg(fig)
    plt.savefig('Resources/mfcc_2.png')
    

def predict(model, X, idx):
    
    genre_dict = {
        0 : "classical",
        1 : "folk",
        2 : "hiphop",
        3 : "jazz",
        4 : "pop",
        5 : "rock",
        }
        
    predictions = model.predict(X)
    genre = np.argmax(predictions[idx])
    
    prediction = genre_dict[genre]
    
    return prediction, predictions



#=====================Design & Entering of WAV File=================
st.write("""# Music Genre Classification""")
st.write("##### A music genre classification tool intented to be used on OpenAI's Jukebox original pieces of music")
file = st.file_uploader(
    "Upload your WAV File, and watch the CNN model classify the music genre.", type=["wav"])



#------------------------Model Loading----------------------------
model = keras.models.load_model('./MFCC 10secs Model')

#=========================Start Genre Classification==========================
if file is not None:   
    
    #Write song into CSV file    
    csv_write_data(file, csv_detection_path)
    
    #Load song from csv file
    X, y = csv_read_data(csv_detection_path)
    
    #Save snippets of the song as WAV files
    save_snippets(file)
    
    #Creating melspectogram of every snippet
    create_mfcc('./Resources/extracted_0.wav', './Resources/extracted_1.wav','./Resources/extracted_2.wav')

    #Prediction of every snippet
    prediction_snippet0, predictions0 = predict(model,X,0)
    prediction_snippet1, predictions1 = predict(model,X,1)
    prediction_snippet2, predictions2 = predict(model,X,2)

    
    #Bar Graph Predictions List
    predictions0_list = predictions0.tolist()[0]
    predictions1_list = predictions1.tolist()[1]
    predictions2_list = predictions2.tolist()[2]
    
    #Data Frame for plotting graph
    df0 = pd.DataFrame(predictions0_list, index=labels)
    df1 = pd.DataFrame(predictions1_list, index=labels)
    df2 = pd.DataFrame(predictions2_list, index=labels)
    

    
    #==============Website Design==================
    st.write(f"")
    st.write(f"")
    
    st.write(f"### Full Song:")
    st.audio(file, "audio/mp3")
    
    st.write(f"")
    st.write(f"")
    st.write(f"")
    st.write(f"")
    
    #Creating tabs
    tab1, tab2, tab3 = st.tabs(["Snippet 1", "Snippet 2", "Snippet 3"])
    
    #Snippet 1
    with tab1:
        st.write("### Snippet 1")
        st.audio('Resources/extracted_0.wav', "audio/mp3")
        st.write(f"")
        
        st.write(f"### Genre Prediction Snippet 1: "+prediction_snippet0)
        st.bar_chart(df0)
        st.write(f"")
        
        st.write(f"### MFCC First Snippet")
        st.image("Resources/mfcc_0.png", use_column_width=True)  
        
        
        
        
    
    #Snippet 2
    with tab2:
        st.write("### Snippet 2")
        st.audio('Resources/extracted_1.wav', "audio/mp3")
        
        st.write(f"### Genre Prediction Snippet 2: "+prediction_snippet1)
        #st.pyplot(fig_1)
        st.bar_chart(df1)
        
        st.write(f"### MFCC Second Snippet")
        st.image("Resources/mfcc_1.png", use_column_width=True)   

    
    
    #Snippet 3
    with tab3:
        st.write("### Snippet 3")
        st.audio('Resources/extracted_2.wav', "audio/mp3")
        
        st.write(f"### Genre Prediction Snippet 3: "+prediction_snippet2)
        #st.pyplot(fig_2)
        st.bar_chart(df2)
        
        st.write(f"### MFCC Third Snippet")
        st.image("Resources/mfcc_2.png", use_column_width=True)
        
    
    
    
    
    

