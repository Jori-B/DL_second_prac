import pandas as pd
import datatable
import random


import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
import wave_processing

from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers
from tensorflow.keras import models
from IPython import display

def preprocess_dataset(files):
    files_ds = tf.data.Dataset.from_tensor_slices(files)
    output_ds = files_ds.map(get_waveform_and_label, num_parallel_calls = AUTOTUNE)
    output_ds = output_ds.map(
        get_spectogram_and_label_id, num_parallel_calls = AUTOTUNE)
    return output_ds

def get_spectogram_and_label_id(audio, label):
    spectogram = get_spectogram(audio)
    spectogram = tf.expand_dims(spectogram, -1)
    label_id = tf.argmax(label == commands)
    return spectogram, label_id

#Apply short time fourier transform on waves
def get_spectrogram(waveform):
    # Padding for files with less than 16000 samples
    zero_padding = tf.zeros([16000] - tf.shape(waveform), dtype=tf.float32)

    # Concatenate audio with padding so that all audio clips will be of the
    # same length
    waveform = tf.cast(waveform, tf.float32)
    equal_length = tf.concat([waveform, zero_padding], 0)
    spectrogram = tf.signal.stft(
    equal_length, frame_length=255, frame_step=128)



    spectrogram = tf.abs(spectrogram)

    return spectrogram

#This function reads a .wav file and returns the wave with the label
def get_waveform_and_label(file_path):
    label = get_label(file_path)
    audio_binary = tf.io.read_file(file_path)
    waveform = decode_audio(audio_binary)
    return waveform, label

#returns the WAV encoded audio as a tensor
def decode_audio(audio_binary):
    audio, _ = tf.audio.decode_wav(audio_binary)
    return tf.squeeze(audio, axis=-1)
