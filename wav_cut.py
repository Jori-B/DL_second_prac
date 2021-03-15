import numpy as np
import time
import pandas as pd
import matplotlib.pylab as plt
import tensorflow as tf
import tensorflow_hub as hub
import os
import glob
import keras

from pydub import AudioSegment
from pydub.utils import make_chunks
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from timeit import default_timer as timer
from os.path import expanduser

home = expanduser("~")
path_data = f'{home}/Downloads/vox1_wav'
path_copy = f'{home}/Downloads/vox1_wav_split'

chunk_length_ms = 500

def main():
    list_subfolders_with_paths = [f.name for f in os.scandir(path_data) if f.is_dir()]
    for folder in list_subfolders_with_paths:
        id_path = (f"{path_data}/{folder}")
        list_ID_subfolders_with_paths = [f.name for f in os.scandir(id_path) if f.is_dir()]
        for id in list_ID_subfolders_with_paths:
            for filename in glob.iglob(f"{id_path}/{id}/**/*.*", recursive=True):
                wav_file = AudioSegment.from_file(filename)
                chunks = make_chunks(wav_file, chunk_length_ms) #Make chunks of one sec
                for i, chunk in enumerate(chunks):
                    silence = AudioSegment.silent(duration=chunk_length_ms-len(wav_file)+1)
                    chunk_padded = chunk + silence
                    if not os.path.exists(f"{path_copy}/{id_path}/{id}"):
                        os.makedirs(f"{path_copy}/{id_path}/{id}")
                    chunk_padded.export(f"{path_copy}/{id_path}/{id}/wav_file{i}", format="wav")



if __name__ == "__main__":
    main()
