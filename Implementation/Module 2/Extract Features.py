import csv
import sys
import os
import json
import librosa
import numpy as np

#Declarations
audio_path = "./Modified_FMA"
mfcc_10sec_csv_path = "./csv/mfcc_10sec_fma.csv"
mfcc_30sec_csv_path = "./csv/mfcc_30sec_fma.csv"
melSpec_csv_path = "./csv/melSpec_fma.csv"

sr=22050

csv.field_size_limit(sys.maxsize)


#Splitting the samples in NUM_SLICES
TOTAL_SAMPLES = 29 * sr
NUM_SLICES = 3
SAMPLES_PER_SLICE = int(TOTAL_SAMPLES / NUM_SLICES)


#Write MFCC to CSV
def mfcc_10sec_csv_write_data(audio_path, csv_path):
    #Create a list of rows for writing to CSV
    rows = []

    #Generate MFCC for every song & writing in JSON
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(audio_path)):
        for file in filenames:
            #Generate MFCC
            song, sr = librosa.load(os.path.join(dirpath, file),duration=29)

            for s in range(NUM_SLICES):
                start_sample = SAMPLES_PER_SLICE * s
                end_sample = start_sample + SAMPLES_PER_SLICE
                
                
                mfcc = librosa.feature.mfcc(y=song[start_sample:end_sample], sr=sr, n_mfcc=40)
                mfcc = mfcc.T

                rows.append([i-1,json.dumps(mfcc.tolist())])
        
            #Writing CSV
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)

            # Write the header row
            writer.writerow(["genre", "mfcc"])

            # Write the values rows
            for row in rows:
                writer.writerow(row)
                
  
#MFCC 30 seconds                
def mfcc_30sec_csv_write_data(audio_path, csv_path):
    #Create a list of rows for writing to CSV
    rows = []

    #Generate MFCC for every song & writing in JSON
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(audio_path)):
        for file in filenames:
            #Generate MFCC
            song, sr = librosa.load(os.path.join(dirpath, file),duration=29)
                
            mfcc = librosa.feature.mfcc(y=song, sr=sr, n_mfcc=40)
            mfcc = mfcc.T

            rows.append([i-1,json.dumps(mfcc.tolist())])
    
            #Writing CSV
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)

            # Write the header row
            writer.writerow(["genre", "mfcc"])

            # Write the values rows
            for row in rows:
                writer.writerow(row)
                
        print(dirpath)



#Write Mel SPec to CSV
def melspec_csv_write_data(audio_path, csv_path):
    #Create a list of rows for writing to CSV
    rows = []

    #Generate MFCC for every song & writing in JSON
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(audio_path)):
        for file in filenames:
            #Generate MFCC
            song, sr = librosa.load(os.path.join(dirpmfcc_csv_pathath, file),duration=29)

            for s in range(NUM_SLICES):
                start_sample = SAMPLES_PER_SLICE * s
                end_sample = start_sample + SAMPLES_PER_SLICE
                
                
                melspec = librosa.feature.melspectrogram(y=song[start_sample:end_sample], sr=sr)
                melspec = melspec.T

                rows.append([i-1,json.dumps(melspec.tolist())])
        
            #Writing CSV
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)

            # Write the header row
            writer.writerow(["genre", "melspec"])

            # Write the values rows
            for row in rows:
                writer.writerow(row)
                


#Main
mfcc_10sec_csv_write_data(audio_path,mfcc_10sec_csv_path)
mfcc_30sec_csv_write_data(audio_path,mfcc_30sec_csv_path)
melspec_csv_write_data(audio_path,melSpec_csv_path)







