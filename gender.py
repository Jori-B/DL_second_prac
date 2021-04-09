import numpy as np
import time
import pandas as pd
import matplotlib.pylab as plt
import tensorflow as tf
import tensorflow_hub as hub
import os
import glob
import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.models import Model

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from timeit import default_timer as timer
from os.path import expanduser
from tensorflow.keras.layers.experimental import preprocessing as tfpreprocessing
from tensorflow.keras.layers import Dense, Input, Layer
home = expanduser("~")
path_data_train = f'{home}/Downloads/dataset_split_2/train'
path_data_test = f'{home}/Downloads/dataset_split_2/test'
path_data_validate = f'{home}/Downloads/dataset_split_2/validate'
path_csv = f'{home}/Downloads/'
num_epoch = 25
BATCH_SIZE = 10
label_encoder = preprocessing.LabelEncoder()

class nationality_checker:
    def __init__(self, path_csv):
        try:
            self.data = pd.read_csv(f'{path_csv}/vox1_meta.csv')
        except OSError:
            print("Could not open/read file vox1_meta.csv")
            sys.exit()
        self.num_classes = self.data['Gender'].nunique()

    def get_nationality(self, id):
        label = self.data.loc[self.data['VoxCeleb1_ID'] == id]['Gender'].values[0]
        return label

#this function is a mapping function that reads the data associated with pathname that is stored to reduce ram requirements
def audio_file_to_tensor(file, label):
    def _audio_file_to_tensor(file, label):
        SAMPLE_SELECTION = 80000
        path = f"{file.numpy().decode()}"
        audio, _ = tf.audio.decode_wav(tf.io.read_file(path),desired_samples=SAMPLE_SELECTION)
        waveform = tf.cast(audio, tf.float32)
        squeezed = tf.squeeze(waveform, axis=-1)
        padded = tf.zeros([SAMPLE_SELECTION] - tf.shape(squeezed),dtype=tf.float32)
        audio_squeezed = tf.concat([squeezed,padded],0)
        spectrogram = tf.signal.stft(audio_squeezed, frame_length=255, frame_step=128)
        spectrogram = tf.abs(spectrogram)
        spectrogram = tf.expand_dims(spectrogram, -1)
        return spectrogram, label

    file, label = tf.py_function(_audio_file_to_tensor,
                                 inp=(file, label),
                                 Tout=(tf.float32, tf.int64))
    file.set_shape([624, 129,1])
    label.set_shape([])
    return (file, label)


class callback(keras.callbacks.Callback):
    def __init__(self, logs={}):
        self.time = []
        self.acc = []
        self.loss = []
        self.val_loss = []
        self.val_acc = []

    def on_epoch_begin(self, epoch, logs={}):
        self.starttime = timer()

    def on_epoch_end(self, epoch, logs={}):
        print(logs)
        self.time.append(timer() - self.starttime)
        self.acc.append(logs['acc'])
        self.loss.append(logs['loss'])
        self.val_loss.append(logs['val_loss'])
        self.val_acc.append(logs['val_acc'])


def printHistory(history, model_num):

    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    #plt.show()
    plt.savefig(f"model_accuracy_model{model_num}.png")
    plt.clear()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    #plt.show()
    plt.savefig(f"model_loss_model{model_num}.png")


def main():
    #create csv class that reads in the labels
    nationality_per_id = nationality_checker(path_csv)
###############################################################################################################################33
    #loading test data from folder structure
    list_of_files = []
    labels = []
    # get a list of all the folder names (f.path for path) in the directory specified in path
    list_subfolders_with_paths = [f.name for f in os.scandir(path_data_train) if f.is_dir()]
    #loop over the folders of each nationality
    for nationality in list_subfolders_with_paths:
        id_path = (f"{path_data_train}/{nationality}")
        list_ID_subfolders_with_paths = [f.name for f in os.scandir(id_path) if f.is_dir()]
        #for each nationality loop over the folders with each id
        for id in list_ID_subfolders_with_paths:
            code_path = (f"{path_data_train}/{folder}/{id}")
            list_CODE_subfolders_with_paths = [f.name for f in os.scandir(code_path) if f.is_dir()]
            #for each id loop over the folders with different interviews
            for code in list_CODE_subfolders_with_paths:
                #gather all the files
                for filename in glob.iglob(f"{id_path}/{id}/{code}/*.*", recursive=True):
                    list_of_files.append(filename)
                    label = nationality_per_id.get_nationality(id)
                    labels.append(label)
    label_trans_train = label_encoder.fit(labels)  # fit a transformer that transforms label strings to numerical
    label_train = label_trans_train.transform(labels)
    dataset_train = tf.data.Dataset.from_tensor_slices((list_of_files, label_train))
    dataset_train = dataset_train.map(audio_file_to_tensor)
    train_dataset = dataset_train.cache().shuffle(len(list_of_files)).batch(BATCH_SIZE).repeat()
    train_dataset = train_dataset.prefetch(buffer_size=10)
    STEPS_PER_EPOCH = len(list_of_files)/BATCH_SIZE


##################################################################################################################################
    list_of_files_test = []
    labels_test = []
    # get a list of all the folder names (f.path for path) in the directory specified in path
    list_subfolders_with_paths = [f.name for f in os.scandir(path_data_test) if f.is_dir()]
    #loop over the folders of each nationality
    for folder in list_subfolders_with_paths:
        id_path = (f"{path_data_test}/{folder}")
        list_ID_subfolders_with_paths = [f.name for f in os.scandir(id_path) if f.is_dir()]
        #for each nationality loop over the folders with each id
        for id in list_ID_subfolders_with_paths:
            code_path = (f"{path_data_test}/{folder}/{id}")
            list_CODE_subfolders_with_paths = [f.name for f in os.scandir(code_path) if f.is_dir()]
            #for each id loop over the folders with different interviews
            for code in list_CODE_subfolders_with_paths:
                #gather all the files
                for filename in glob.iglob(f"{id_path}/{id}/{code}/*.*", recursive=True):
                    list_of_files_test.append(filename)
                    labels_test = nationality_per_id.get_nationality(id)
                    labels_test.append(labels_test)
    label_trans_test = label_encoder.fit(labels_test)  # fit a transformer that transforms label strings to numerical
    label_test = label_trans_test.transform(labels_test)
    dataset_test = tf.data.Dataset.from_tensor_slices((list_of_files_test, label_test))
    dataset_test = dataset_test.map(audio_file_to_tensor)
    test_dataset = dataset_test.cache().shuffle(len(list_of_files_test)).batch(BATCH_SIZE)
    test_dataset = test_dataset.prefetch(buffer_size=10)
###################################################################################################################################
    cb = callback()
    checkpoint_path_save = "training_1/cp.ckpt"

    save_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path_save,
                                                     save_weights_only=True,
                                                     verbose=1)

    input_shape = [624, 129, 1]

    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, 3, activation='relu'),
        layers.Conv2D(32, 3, activation='relu'),
        layers.Conv2D(32, 3, activation='relu'),

        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(48, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(12, activation='relu'),
        layers.Dense(12, activation='relu'),
        layers.Dense(12, activation='relu'),
        layers.Dense(nationality_per_id.num_classes),
        ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['acc'],
    )


    history = model.fit(train_dataset, epochs=num_epoch, steps_per_epoch=STEPS_PER_EPOCH,
                          callbacks=[cb,save_callback],
                          validation_data=test_dataset)
    printHistory(history,1)

if __name__ == "__main__":
    main()
