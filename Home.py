import streamlit as st
import base64
import matplotlib as mpl
import numpy as np
import json
import csv
import sys
import numpy as np 
import chardet
from pydub import AudioSegment
from presets import Preset
import librosa as librosa
import librosa.display
from tensorflow import keras
from bing_image_downloader import downloader
from keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv2D, MaxPooling2D
from keras.models import Model
from keras.initializers import glorot_uniform
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.cm as cm
from matplotlib.colors import Normalize

#=========================DECLARATIONS======================
sr = 22050

TOTAL_SAMPLES = 29*sr
NUM_SLICES = 3
SAMPLES_PER_SLICE = int(TOTAL_SAMPLES / NUM_SLICES)

csv.field_size_limit(sys.maxsize)
csv_detection_path = './Resources/song_detection.csv'

librosa_preset = Preset(librosa)
librosa_preset['sr'] = 22050

labels = ['folk', 'pop','rock','country','classical','jazz','hiphop']

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
    #fig = plt.Figure()
    fig, ax = plt.subplots(figsize=(10, 4))
    plt.imshow(mfcc0, interpolation='nearest', origin='lower', aspect='auto', cmap=cm.coolwarm)
    plt.colorbar()
    plt.title('MFCCs')
    plt.xlabel('Frame')
    plt.ylabel('MFCC Coefficients')
        
    FigureCanvasAgg(fig)
    plt.savefig('Resources/mfcc_0.png')
    
    #Second snippet
    y, sr = librosa.load(wav_file_1, duration=10)
    mfcc1 = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    #fig = plt.Figure()
    fig, ax = plt.subplots(figsize=(10, 4))
    plt.imshow(mfcc1, interpolation='nearest', origin='lower', aspect='auto', cmap=cm.coolwarm)
    plt.colorbar()
    plt.title('MFCCs')
    plt.xlabel('Frame')
    plt.ylabel('MFCC Coefficients')
        
    FigureCanvasAgg(fig)
    plt.savefig('Resources/mfcc_1.png')
    
    #First snippet
    y, sr = librosa.load(wav_file_2, duration=10)
    mfcc2 = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    #fig = plt.Figure()
    fig, ax = plt.subplots(figsize=(10, 4))
    plt.imshow(mfcc2, interpolation='nearest', origin='lower', aspect='auto', cmap=cm.coolwarm)
    plt.colorbar()
    plt.title('MFCCs')
    plt.xlabel('Frame')
    plt.ylabel('MFCC Coefficients')
        
    FigureCanvasAgg(fig)
    plt.savefig('Resources/mfcc_2.png')
    

def predict(model, X, idx):
    
    genre_dict = {
        0 : "folk",
        1 : "pop",
        2 : "rock",
        3 : "country",
        4 : "classical",
        5 : "jazz",
        6 : "hiphop",
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
model = keras.models.load_model('./Model Final/MFCC_Model')


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
    
    #Bar Graph Configuration
    my_cmap = mpl.colormaps['gnuplot']
    
    #Bar Graph Snippet 1
    predictions0_list = predictions0.tolist()[0]
    
    fig_0, ax = plt.subplots(figsize=(6, 4.5))
    ax.bar(x=labels, height=predictions0_list, color=my_cmap(np.linspace(0, 1, len(labels))))
    ax.set_title(
        "Snippet 1: Probability Distribution Over Different Genres")
    plt.xlabel("Predicted Genre")
    plt.ylabel("Probability")
    
    
    
    #Bar Graph Snippet 2
    predictions1_list = predictions1.tolist()[1]
    
    fig_1, ax_1 = plt.subplots(figsize=(6, 4.5))
    ax_1.bar(x=labels, height=predictions1_list, color=my_cmap(np.linspace(0, 1, len(labels))))
    ax_1.set_title(
        "Snippet 2: Probability Distribution Over Different Genres")
    plt.xlabel("Predicted Genre")
    plt.ylabel("Probability")
    
    
    
    #Bar Graph Snippet 1
    predictions2_list = predictions2.tolist()[2]
    
    fig_2, ax = plt.subplots(figsize=(6, 4.5))
    ax.bar(x=labels, height=predictions2_list, color=my_cmap(np.linspace(0, 1, len(labels))))
    ax.set_title(
        "Snippet 3: Probability Distribution Over Different Genres")
    plt.xlabel("Predicted Genre")
    plt.ylabel("Probability")


    
    #==============Website Design==================
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
        st.pyplot(fig_0)
        st.write(f"")
        
        st.bar_chart(predictions0_list, labels)
        
        st.write(f"### Mel Spectrogram First Snippet")
        st.image("Resources/mfcc_0.png", use_column_width=True)  
        
    
    #Snippet 2
    with tab2:
        st.write("### Snippet 2")
        st.audio('Resources/extracted_1.wav', "audio/mp3")
        
        st.write(f"### Genre Prediction Snippet 2: "+prediction_snippet1)
        st.pyplot(fig_1)
        
        st.write(f"### Mel Spectrogram Second Snippet")
        st.image("Resources/mfcc_1.png", use_column_width=True)   

    
    
    #Snippet 3
    with tab3:
        st.write("### Snippet 3")
        st.audio('Resources/extracted_2.wav', "audio/mp3")
        
        st.write(f"### Genre Prediction Snippet 3: "+prediction_snippet2)
        st.pyplot(fig_2)
        
        st.write(f"### Mel Spectrogram Third Snippet")
        st.image("Resources/mfcc_2.png", use_column_width=True)
        
    
    
    
    
    

