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

# Set seed for experiment reproducibility
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

#This function prints the number of examples per class
def print_data_info(data_dir, nationalities):
    filenames = tf.io.gfile.glob(str(data_dir) + '/*/*')
    filenames = tf.random.shuffle(filenames)
    num_samples = len(filenames)
    print('Number of total examples:', num_samples)
    print("-------------------------")
    for i in range(0, 7):
        print('Number of examples in :', nationalities[i],
            len(tf.io.gfile.listdir(str(data_dir/nationalities[i]))))
    print('Example file tensor:', filenames[0])

def main():
    nationalities = ['Australia', 'India', 'Norway', 'USA', 'Canada', 'Ireland', 'UK']

    #Read the nationalities data
    meta_data = pd.read_csv('vox1_nationality.csv').drop(['index'], axis = 1)

    data_dir = pathlib.Path('data_formatted/')

    #Create a numpy array of the labels
    nationalities = np.array(tf.io.gfile.listdir(str(data_dir)))
    nationalities = nationalities[nationalities != '.DS_Store']

    #Store all of the filenames in an array of strings
    filenames = tf.io.gfile.glob(str(data_dir) + '/*/*')
    filenames = tf.random.shuffle(filenames)

    #Split test/train/dev 80:10:10
    train_size = int(0.8 * len(filenames))
    test_val_size = int(0.1 * len(filenames))
    train_files = filenames[:train_size]
    val_files = filenames[train_size: train_size + test_val_size]
    test_files = filenames[-test_val_size:]
    import pdb; pdb.set_trace()











if __name__ == "__main__":
    main()
