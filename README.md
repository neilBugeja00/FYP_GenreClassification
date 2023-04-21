# FYP_GenreClassification
Modified version of ericzacharia's "MusicGenreClassifier" 
Link: https://github.com/ericzacharia/MusicGenreClassifier

The program uses ericzacharia's CNN model to detect the genre of a musical sample, but modifications were made to suite my FYP's purpose.



Modifications made:
The purpose of this repo is to have a program that can detect the genre of audio samples given from OpenAI's Jukebox.

Each 30 second WAV audio file is split in 3 parts (10 seconds each).
The program will detect the genre of each snippet and display it for the user through a bar chart based on the probability of it being a particular genre.


To run the program download all the libraries in the "requirements.txt" and enter the following code in the terminal:

streamlit run main.py
