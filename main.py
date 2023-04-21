import streamlit as st
import matplotlib as mpl
import numpy as np
from pydub import AudioSegment
from presets import Preset
import librosa as librosa
import librosa.display
from bing_image_downloader import downloader
from keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv2D, MaxPooling2D
from keras.models import Model
from keras.initializers import glorot_uniform
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.cm as cm
from matplotlib.colors import Normalize

#=========================METHODS======================
def cnn(input_shape=(288, 432, 4), classes=9):
    def step(dim, X):
        X = Conv2D(dim, kernel_size=(3, 3), strides=(1, 1))(X)
        X = BatchNormalization(axis=3)(X)
        X = Activation('relu')(X)
        return MaxPooling2D((2, 2))(X)
    X_input = Input(input_shape)
    X = X_input
    layer_dims = [8, 16, 32, 64, 128, 256]
    for dim in layer_dims:
        X = step(dim, X)

    X = Flatten()(X)
    X = Dense(classes, activation='softmax',
              name=f'fc{classes}',  kernel_initializer=glorot_uniform(seed=9))(X)
    model = Model(inputs=X_input, outputs=X, name='cnn')
    return model

#Three extract relevant are required to snip the song in 3 parts
#and test the genre of every snippet

def extract_relevant_0(wav_file, t1, t2):
    wav = AudioSegment.from_wav(wav_file)
    wav = wav[1000*t1:1000*t2]
    wav.export("extracted_0.wav", format='wav')
    
def extract_relevant_1(wav_file, t1, t2):
    wav = AudioSegment.from_wav(wav_file)
    wav = wav[1000*t1:1000*t2]
    wav.export("extracted_1.wav", format='wav')
    
def extract_relevant_2(wav_file, t1, t2):
    wav = AudioSegment.from_wav(wav_file)
    wav = wav[1000*t1:1000*t2]
    wav.export("extracted_2.wav", format='wav')        


def create_melspectrogram(wav_file_0, wav_file_1, wav_file_2):
    #First snippet
    y, sr = librosa.load(wav_file_0, duration=3)
    mels = librosa.feature.melspectrogram(y=y, sr=sr)
    fig = plt.Figure()
    FigureCanvasAgg(fig)
    plt.imshow(librosa.power_to_db(mels, ref=np.max))
    plt.savefig('melspectrogram_0.png')
    
    #Second snippet
    y, sr = librosa.load(wav_file_1, duration=3)
    mels = librosa.feature.melspectrogram(y=y, sr=sr)
    fig = plt.Figure()
    FigureCanvasAgg(fig)
    plt.imshow(librosa.power_to_db(mels, ref=np.max))
    plt.savefig('melspectrogram_1.png')
    
    #Third snippet
    y, sr = librosa.load(wav_file_2, duration=3)
    mels = librosa.feature.melspectrogram(y=y, sr=sr)
    fig = plt.Figure()
    FigureCanvasAgg(fig)
    plt.imshow(librosa.power_to_db(mels, ref=np.max))
    plt.savefig('melspectrogram_2.png')


def predict(image_data, model):
    image = img_to_array(image_data).reshape((1, 288, 432, 4))
    prediction = model.predict(image / 255)
    prediction = prediction.reshape((9, ))
    class_label = np.argmax(prediction)
    return class_label, prediction





#Definitions
librosa_preset = Preset(librosa)
librosa_preset['sr'] = 44100


#=====================Design & Entering of WAV File=================
st.write("""# Music Genre Classifier""")
st.write("##### An altered version by Neil Bugeja of Eric Zacharia's 'Convolutional Neural Network Classifier'")
file = st.file_uploader(
    "Upload your WAV File, and watch the CNN model classify the music genre.", type=["wav"])
class_labels = ['blues', 'classical', 'country',
                'disco', 'hiphop', 'metal', 'pop', 'reggae', 'rock']



#Model Loading
model = cnn(input_shape=(288, 432, 4), classes=9)
model.load_weights("CNNModelWeights.h5")


#=========================Start Classification==========================
if file is not None:
    #Predicting genre of each snippet
    extract_relevant_0(file, 0, 10)
    extract_relevant_1(file, 10, 20)
    extract_relevant_2(file, 20, 30)
    
    #Creating melspectogram of every snippet
    create_melspectrogram("extracted_0.wav","extracted_1.wav","extracted_2.wav")
    
    #image data of every snippet
    image_data_0 = load_img('melspectrogram_0.png',
                          color_mode='rgba', target_size=(288, 432))
    
    image_data_1 = load_img('melspectrogram_1.png',
                          color_mode='rgba', target_size=(288, 432))
    
    image_data_2 = load_img('melspectrogram_2.png',
                          color_mode='rgba', target_size=(288, 432))
    
    
    #Prediction of every snippet

    class_label_0, prediction_0 = predict(image_data_0, model)
    class_label_1, prediction_1 = predict(image_data_1, model)
    class_label_2, prediction_2 = predict(image_data_2, model)
    
    prediction_0 = prediction_0.reshape((9,))
    prediction_1 = prediction_1.reshape((9,))
    prediction_2 = prediction_2.reshape((9,))
    
    #Bar Graph Configuration
    color_data = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    my_cmap = mpl.colormaps['jet']
    my_norm = Normalize(vmin=0, vmax=9)

    #Graph snippet 1
    fig_0, ax = plt.subplots(figsize=(6, 4.5))
    ax.bar(x=class_labels, height=prediction_0,
           color=my_cmap(my_norm(color_data)))
    plt.xticks(rotation=45)
    ax.set_title(
        "Snippet 1: Probability Distribution Over Different Genres")
    
    #Graph snippet 2
    fig_1, ax = plt.subplots(figsize=(6, 4.5))
    ax.bar(x=class_labels, height=prediction_1,
           color=my_cmap(my_norm(color_data)))
    plt.xticks(rotation=45)
    ax.set_title(
        "Snippet 2: Probability Distribution Over Different Genres")
    
    #Graph snippet 2
    fig_2, ax = plt.subplots(figsize=(6, 4.5))
    ax.bar(x=class_labels, height=prediction_2,
           color=my_cmap(my_norm(color_data)))
    plt.xticks(rotation=45)
    ax.set_title(
        "Snippet 3: Probability Distribution Over Different Genres")




    #==============Website Design==================
    st.write(f"### Full Song:")
    st.audio(file, "audio/mp3")
    
    st.write(f"")
    st.write(f"")
    st.write(f"")
    st.write(f"")
    
    
    #Snippet 1
    st.write("### Snippet 1")
    st.audio('extracted_0.wav', "audio/mp3")
    st.write(f"")
    
    st.write(f"### Genre Prediction Snippet 1: {class_labels[class_label_0]}")
    st.pyplot(fig_0)
    st.write(f"")
    
    st.write(f"### Mel Spectrogram First Snippet")
    st.image("melspectrogram_0.png", use_column_width=True)  
    
    st.write(f"")
    st.write(f"")
    st.write(f"")
    st.write(f"")
    
    #Snippet 2
    st.write("### Snippet 2")
    st.audio('extracted_1.wav', "audio/mp3")
    
    st.write(f"### Genre Prediction Snippet 2: {class_labels[class_label_1]}")
    st.pyplot(fig_1)
    
    st.write(f"### Mel Spectrogram Second Snippet")
    st.image("melspectrogram_1.png", use_column_width=True)   
    
    st.write(f"")
    st.write(f"")
    st.write(f"")
    st.write(f"")

    
    
    #Snippet 3
    st.write("### Snippet 3")
    st.audio('extracted_2.wav', "audio/mp3")
    
    st.write(f"### Genre Prediction Snippet 3: {class_labels[class_label_2]}")
    st.pyplot(fig_2)
    
    st.write(f"### Mel Spectrogram Third Snippet")
    st.image("melspectrogram_2.png", use_column_width=True)
    
    
    
    
    


