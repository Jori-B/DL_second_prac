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

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from timeit import default_timer as timer
from tensorflow.keras.layers.experimental import preprocessing as tfpreprocessing
from os.path import expanduser
home = expanduser("~")
path_data = f'{home}/Downloads/vox1_dev_wav_split'
path_csv = f'{home}/Downloads/'
num_epoch = 100
BATCH_SIZE = 10
label_encoder = preprocessing.LabelEncoder()

class nationality_checker:
    def __init__(self, path_csv):
        try:
            self.data = pd.read_csv(f'{path_csv}/vox1_meta.csv')
        except OSError:
            print("Could not open/read file vox1_meta.csv")
            sys.exit()
        self.num_classes = self.data['VoxCeleb1_ID'].nunique()

    def get_nationality(self, id):
        label = self.data.loc[self.data['VoxCeleb1_ID'] == id]['Gender'].values[0]
        if not label:
            label ='m'
        label = id
        return label


def decode_audio(audio_file, label):
    audio = tf.io.read_file(audio_file)
    #print(audio.shape)
    #print(audio)
    audio, _ = tf.audio.decode_wav(audio) # _ here means a throwable variable as we don't use it
    #print(audio.shape)
    #print(audio)
    waveform = tf.cast(audio, tf.float32)
    #print(waveform.shape)
    #print(waveform)
    audio_squeezed = tf.squeeze(waveform, axis=-1)
    #print(audio_squeezed.shape)
    #print(audio_squeezed)
    equal_length = tf.concat([audio_squeezed], 0)
    #print(equal_length.shape)
    #print(equal_length)
    spectrogram = tf.signal.stft(equal_length, frame_length=255, frame_step=128)
    #print(spectrogram.shape)
    #print(spectrogram)
    spectrogram = tf.abs(spectrogram)
    #print(spectrogram.shape)
    #print(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, -1)
    #print(spectrogram.shape)
    #print(spectrogram)


    return spectrogram, label


def audio_file_to_tensor(file, label):
    def _audio_file_to_tensor(file, label):
        path = f"{file.numpy().decode()}"
        audio, _ = tf.audio.decode_wav(tf.io.read_file(path))
        waveform = tf.cast(audio, tf.float32)
        audio_squeezed = tf.squeeze(waveform, axis=-1)
        spectrogram = tf.signal.stft(audio_squeezed, frame_length=255, frame_step=128)
        spectrogram = tf.abs(spectrogram)
        spectrogram = tf.expand_dims(spectrogram, -1)
        return spectrogram, label

    file, label = tf.py_function(_audio_file_to_tensor,
                                 inp=(file, label),
                                 Tout=(tf.float32, tf.int64))
    file.set_shape([124, 129, 1])
    label.set_shape([])
    return (file, label)



# https://stackoverflow.com/questions/57181551/can-i-write-a-keras-callback-that-records-and-returns-the-total-training-time/57182112
class callback(keras.callbacks.Callback):
    def __init__(self, logs={}):
        self.time = []
        self.acc = []
        self.loss = []
        #self.val_loss = []
        #self.val_acc = []

    def on_epoch_begin(self, epoch, logs={}):
        self.starttime = timer()

    def on_epoch_end(self, epoch, logs={}):
        print(logs)
        self.time.append(timer() - self.starttime)
        self.acc.append(logs['acc'])
        self.loss.append(logs['loss'])
        #self.val_loss.append(logs['val_loss'])
        #self.val_acc.append(logs['val_acc'])



# https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/
def printHistory(history, model_num):
    # list all data in history
    #print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    #plt.show()
    plt.savefig(f"model_accuracy_model{model_num}.png")
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    #plt.show()
    plt.savefig(f"model_loss_model{model_num}.png")

def prediction(model, image_batch, model_num, label_trans):
    predicted_batch = model.predict(image_batch)
    predicted_id = np.argmax(predicted_batch, axis=-1)
    predicted_label_batch = label_trans.inverse_transform(predicted_id)

    plt.figure(figsize=(10, 9))
    plt.subplots_adjust(hspace=0.5)
    for n in range(30):
       plt.subplot(6, 5, n + 1)
       plt.imshow(image_batch[n])
       plt.title(predicted_label_batch[n].title())
       plt.axis('off')
    _ = plt.suptitle("Model predictions")
    #plt.show()
    plt.savefig(f"predicted_images_model{model_num}.png")


def main():
    #get number of classes
    nationality_per_id = nationality_checker(path_csv)
    list_of_files = []
    labels = []
    # get a list of all the folder names (f.path for path) in the directory specified in path
    print("Loading data...")
    list_subfolders_with_paths = [f.name for f in os.scandir(path_data) if f.is_dir()]
    for id in list_subfolders_with_paths:
        id_path = (f"{path_data}/{id}")
        for filename in glob.iglob(f"{id_path}/**/*.*", recursive=True):
            list_of_files.append(filename)
            label = nationality_per_id.get_nationality(id)
            labels.append(label)
    print("Loading done")
    file_train, file_test, label_train, label_test = train_test_split(list_of_files, labels, test_size=0.1)
    filenames_train = tf.constant(file_train)
    filenames_test = tf.constant(file_test)



    label_trans = label_encoder.fit(labels)  # fit a transformer that transforms label strings to numerical
    label_train = label_trans.transform(label_train)  # transforms label train into numerical

    label_test = label_trans.transform(label_test)  # transforms label test into numerical

    # set all the data to tensorflow constants
    #filenames_train = tf.constant(file_train)
    #labels_train = tf.constant(label_train)
    #filenames_test = tf.constant(file_test)
    #labels_test = tf.constant(label_test)

    # creates the train dataset
    dataset_train = tf.data.Dataset.from_tensor_slices((filenames_train, label_train))
    dataset_train = dataset_train.map(audio_file_to_tensor)
    train_dataset = dataset_train.cache().shuffle(len(filenames_train)).batch(BATCH_SIZE).repeat()
    train_dataset = train_dataset.prefetch(buffer_size=100)

    # creates the test dataset
    dataset_test = tf.data.Dataset.from_tensor_slices((filenames_test, label_test))
    dataset_test = dataset_test.map(audio_file_to_tensor)
    predict =dataset_test.batch(1)
    test_dataset = dataset_test.cache().shuffle(len(file_test)).batch(BATCH_SIZE)
    train_datasetPredict = dataset_test.batch(32)

    test_dataset = test_dataset.prefetch(buffer_size=8)


    # calculates how many epoch are needed to run over the whole dataset
    STEPS_PER_EPOCH = len(file_train)/BATCH_SIZE
#    cb1 = callback()


#    feature_extractor_model = "https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/4"  # @param {type:"string"}
    # # feature_extractor_model = "https://tfhub.dev/tensorflow/resnet_50/feature_vector/1"  # @param {type:"string"}
#    feature_extractor_layer = hub.KerasLayer(
         # trainable = False freezes the variables in feature extractor layer,
         # so that the training only modifies the new classifier layer.
#        feature_extractor_model, input_shape=(224, 224, 3), trainable=False)
     # predictions = model(image_batch)
     # predictions.shape
     ###################################################################################################################
#    model = tf.keras.Sequential([
#        feature_extractor_layer,
#        tf.keras.layers.Dense(num_classes),  # add a new classification layer.
#    ])
#    model.compile(
#        optimizer=tf.keras.optimizers.Adam(),
#        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#        metrics=['acc'])
#    print("model 1 ##############################################################################################################")
#    model.summary()
#    history = model.fit(train_dataset, epochs=num_epoch, steps_per_epoch=STEPS_PER_EPOCH,
#                          callbacks=[cb1],
#                          validation_data=test_dataset)
#    printHistory(history, 1)
#    prediction(model, image_batch, 1, label_trans)




#    model1_avg_time = sum(cb1.time)/len(cb1.time)
#    model1_total_time = sum(cb1.time)



#    df = pd.DataFrame({'model1_acc': cb1.acc, 'model1_loss': cb1.loss,'model1_val_acc': cb1.val_acc, 'model1_val_loss': cb1.val_loss, 'model1_avg_time': model1_avg_time, 'model1_total_time': model1_total_time})

#    compression_opts = dict(method='zip', archive_name='out1.csv')
#    df.to_csv('out1.zip', index=False,compression=compression_opts)
#    for spectrogram, _ in train_dataset.take(1):
#        input_shape = spectrogram.shape
#    print(input_shape)
#    norm_layer = tfpreprocessing.Normalization()
#    #norm_layer.adapt(train_dataset.map(lambda x, _: x))
    cb = callback()
    #print(nationality_per_id.num_classes)
    input_shape = [124, 129, 1]
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, 3, activation='relu'),
        layers.Conv2D(16, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(12, activation='relu'),
        layers.Dense(nationality_per_id.num_classes),
        ])

    model.summary()

        #layers.Dropout(0.25),
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['acc'],
    )

#    print("model 1 ##############################################################################################################")
#    model.summary()
    history = model.fit(train_dataset, epochs=num_epoch, steps_per_epoch=STEPS_PER_EPOCH,
                          callbacks=[cb],
                          validation_data=test_dataset)

if __name__ == "__main__":
    main()
