import numpy as np
import time
import pandas as pd
import matplotlib.pylab as plt
import tensorflow as tf
import tensorflow_hub as hub
import os
import glob
import keras

from scipy.io import wavfile
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from timeit import default_timer as timer
from os.path import expanduser

home = expanduser("~")
path_data = f'{home}/Downloads/vox1_dev_wav_split'
path_csv = f'{home}/Downloads/'

def main():
    #nationality_per_id = nationality_checker(path_csv)
    list_of_files = []
    labels = []
    # get a list of all the folder names (f.path for path) in the directory specified in path
    list_subfolders_with_paths = [f.name for f in os.scandir(path_data) if f.is_dir()]
    for folder in list_subfolders_with_paths:
        id_path = (f"{path_data}/{folder}")
        list_ID_subfolders_with_paths = [f.name for f in os.scandir(id_path) if f.is_dir()]
        for id in list_ID_subfolders_with_paths:
            for filename in glob.iglob(f"{id_path}/{id}/*.*", recursive=True):
                list_of_files.append(filename)
                #label = nationality_per_id.get_nationality(id)
                #labels.append(id)

    audio_file = list_of_files[0]
    audio = tf.io.read_file(audio_file)
    print(audio.shape)
    #print(audio)
    audio, _ = tf.audio.decode_wav(audio) # _ here means a throwable variable as we don't use it
    print(audio.shape)
    #print(audio)
    waveform = tf.cast(audio, tf.float32)
    print(waveform.shape)
    #print(waveform)
    audio_squeezed = tf.squeeze(waveform, axis=-1)
    print(audio_squeezed.shape)
    #print(audio_squeezed)
    equal_length = tf.concat([audio_squeezed], 0)
    print(equal_length.shape)
    #print(equal_length)
    spectrogram = tf.signal.stft(equal_length, frame_length=255, frame_step=128)
    print(spectrogram.shape)
    #print(spectrogram)
    spectrogram = tf.abs(spectrogram)
    print(spectrogram.shape)
    #print(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, -1)
    print(spectrogram.shape)
    #print(spectrogram)


if __name__ == "__main__":
    main()
