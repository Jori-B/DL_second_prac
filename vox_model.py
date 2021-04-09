import numpy as np
import time
import pandas as pd
import matplotlib.pylab as plt
import tensorflow as tf
import tensorflow_hub as hub
import os
import sys
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
num_epoch = 50
BATCH_SIZE = 10
label_encoder = preprocessing.LabelEncoder()

class nationality_checker:
    def __init__(self, path_csv):
        try:
            self.data = pd.read_csv(f'{path_csv}/vox1_meta.csv')
        except OSError:
            print("Could not open/read file vox1_meta.csv")
            sys.exit()
        self.num_classes = self.data['Nationality'].nunique()

    def get_nationality(self, id):
        label = self.data.loc[self.data['VoxCeleb1_ID'] == id]['Nationality'].values[0]
        if not(label):
            label ='m'
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


def main():
    #get number of classes
###############################################################################################################################33
    nationality_per_id = nationality_checker(path_csv)
    list_of_files = []
    labels = []
    # get a list of all the folder names (f.path for path) in the directory specified in path
    list_subfolders_with_paths = [f.name for f in os.scandir(path_data_train) if f.is_dir()]
    for folder in list_subfolders_with_paths:
        #print(folder)
        id_path = (f"{path_data_train}/{folder}")
        list_ID_subfolders_with_paths = [f.name for f in os.scandir(id_path) if f.is_dir()]
        for id in list_ID_subfolders_with_paths:
        #    print(id)
            code_path = (f"{path_data_train}/{folder}/{id}")
            #print(code_path)
            list_CODE_subfolders_with_paths = [f.name for f in os.scandir(code_path) if f.is_dir()]
            #print(list_CODE_subfolders_with_paths)
            for code in list_CODE_subfolders_with_paths:
                #print(code)
                for filename in glob.iglob(f"{id_path}/{id}/{code}/*.*", recursive=True):
                    #print(filename)
                    list_of_files.append(filename)
                    label = nationality_per_id.get_nationality(id)
                    labels.append(label)
    #print(list_of_files)
    label_trans_train = label_encoder.fit(labels)  # fit a transformer that transforms label strings to numerical
    label_train = label_trans_train.transform(labels)
    dataset_train = tf.data.Dataset.from_tensor_slices((list_of_files, label_train))
    dataset_train = dataset_train.map(audio_file_to_tensor)
    train_dataset = dataset_train.cache().shuffle(len(list_of_files)).batch(BATCH_SIZE).repeat()
    train_dataset = train_dataset.prefetch(buffer_size=10)
    STEPS_PER_EPOCH = len(list_of_files)/BATCH_SIZE


##################################################################################################################################
    list_of_files_test = []
    labels = []

    # get a list of all the folder names (f.path for path) in the directory specified in path
    list_subfolders_with_paths = [f.name for f in os.scandir(path_data_test) if f.is_dir()]
    for folder in list_subfolders_with_paths:
        #print(folder)
        id_path = (f"{path_data_train}/{folder}")
        list_ID_subfolders_with_paths = [f.name for f in os.scandir(id_path) if f.is_dir()]
        for id in list_ID_subfolders_with_paths:
        #    print(id)
            code_path = (f"{path_data_train}/{folder}/{id}")
            #print(code_path)
            list_CODE_subfolders_with_paths = [f.name for f in os.scandir(code_path) if f.is_dir()]
            #print(list_CODE_subfolders_with_paths)
            for code in list_CODE_subfolders_with_paths:
                #print(code)
                for filename in glob.iglob(f"{id_path}/{id}/{code}/*.*", recursive=True):
                    #print(filename)
                    list_of_files_test.append(filename)
                    label = nationality_per_id.get_nationality(id)
                    labels.append(label)
    #print(list_of_files)
    label_trans_test = label_encoder.fit(labels)  # fit a transformer that transforms label strings to numerical
    label_test = label_trans_test.transform(labels)
    dataset_test = tf.data.Dataset.from_tensor_slices((list_of_files_test, label_test))
    dataset_test = dataset_test.map(audio_file_to_tensor)
    test_dataset = dataset_test.cache().shuffle(len(list_of_files_test)).batch(BATCH_SIZE)
    test_dataset = test_dataset.prefetch(buffer_size=10)


######################################################################################################################################



####################################################################################################################################################
    #print(filenames)
    #print(list_of_files)



    # calculates how many epoch are needed to run over the whole dataset

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
    checkpoint_path_save = "training_2/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path_save)

# Create a callback that saves the model's weights
    save_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path_save,
                                                 save_weights_only=True,
                                                 verbose=1)

    #print(nationality_per_id.num_classes)
    input_shape = [624,129,1]
    #stddev = 5
    #model = models.Sequential([
    #    layers.Input(shape=input_shape),
    #    layers.GaussianNoise(stddev),
    #    layers.Conv2D(64, 4, activation='relu'),
    #    layers.Conv2D(64, 4, activation='relu'),
    #    layers.Conv2D(64, 4, activation='relu'),
    #    layers.Conv2D(64, 4, activation='relu'),
    #    layers.Conv2D(64, 4, activation='relu'),
    #    layers.Conv2D(64, 4, activation='relu'),
    #    layers.MaxPooling2D(),
    #    layers.Flatten(),
    #    layers.Dense(512, activation='relu'),
    #    layers.Dropout(0.2),
    #    layers.Dense(512, activation='relu'),
    #    layers.Dense(512, activation='relu'),
    #    layers.Dense(512, activation='relu'),
    #    layers.Dense(512, activation='relu'),
    #    layers.Dense(512, activation='relu'),
    #    layers.Dense(512, activation='relu'),
    #    layers.Dense(512, activation='relu'),
    #    layers.Dense(nationality_per_id.num_classes),
    #    ])

    #model.summary()
    checkpoint_path = "training_1/cp.ckpt"

    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, 3, activation='relu'),
        layers.Conv2D(32, 3, activation='relu'),
        layers.Conv2D(16, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(12, activation='relu'),
        layers.Dense(12, activation='relu'),
        layers.Dense(12, activation='relu'),
        layers.Dense(nationality_per_id.num_classes),
    ])





        #layers.Dropout(0.25),
    #x = model.layers[-1].output
    #feature_extractor_layer = hub.KerasLayer(
            # trainable = False freezes the variables in feature extractor layer,
            # so that the training only modifies the new classifier layer.
    #        x, trainable=False)


        # Attach a classification head

        # Now wrap the hub layer in a tf.keras.Sequential model
    #model = tf.keras.Sequential([
    #    feature_extractor_layer,
    #    tf.keras.layers.Dense(num_classes)  # add a new classification layer.
    #])



    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['acc'],
    )
    #model.load_weights(checkpoint_path)


    #input_tensor = Input(shape=input_shape)
    #hidden = model.layers[-2].output
    #model2 = models.Sequential([
    #    layers.Input(shape=input_shape),
        #hidden,
        #layers.Dense(nationality_per_id.num_classes),

    #])
    #input = layers.Input(shape=input_shape, name="input")

    #hidden =  Dense(120, activation='relu')(model.layers[-2].output)
    #out = Dense(nationality_per_id.num_classes)(hidden)
#    model2 = Model(input,out)
#    model2 = Model(input, out)




    #model_no_output = hub.KerasLayer(model_no_output,trainable=False)
    #model_new_output = tf.keras.Sequential([
    #    model_no_output,
    #    tf.keras.layers.Dense(num_classes)  # add a new classification layer.
    #])

#    model2.compile(
#        optimizer=tf.keras.optimizers.Adam(),
#        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#        metrics=['acc'],
#    )


    #model = hub.load('https://tfhub.dev/google/vggish/1')
    #feature_extractor_layer = hub.KerasLayer(
        # trainable = False freezes the variables in feature extractor layer,
        # so that the training only modifies the new classifier layer.
    #    model, input_shape=input_shape, trainable=False)
    # Attach a classification head


    # Now wrap the hub layer in a tf.keras.Sequential model
    #model = tf.keras.Sequential([
    #    feature_extractor_layer,
    #    tf.keras.layers.Dense(nationality_per_id.num_classes)  # add a new classification layer.
    #])
    #waveform = np.zeros(3 * 16000, dtype=np.float32)

# Run the model, check the output.
#    embeddings = model(waveform)
#    embeddings.shape.assert_is_compatible_with([None, 128])
    #model = models.Sequential([
    #    layers.Input(shape=input_shape),
    #    layers.Conv2D(96, (7,7), strides=(1), activation='relu'),
    #    layers.MaxPooling2D(3),
    #    layers.Conv2D(256, (5,5), strides=(1), activation='relu'),
    #    layers.MaxPooling2D(2),
    #    layers.Conv2D(384, (3,3), strides=(1), activation='relu'),
    #    layers.Conv2D(256, (3,3), strides=(1), activation='relu'),
    #    layers.Conv2D(256, (3,3), strides=(1), activation='relu'),
    #    layers.Dense(4096, activation='relu'),
    #    layers.GlobalAveragePooling2D(),
    #    layers.Dense(1024, activation='relu'),
    #    layers.Dense(nationality_per_id.num_classes),
    #])
    #model.compile(
    #    optimizer=tf.keras.optimizers.Adam(),
    #    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    ##    metrics=['acc'],
    #)


    print("model 1 ##############################################################################################################")
    model.summary()
    history = model.fit(train_dataset, epochs=num_epoch, steps_per_epoch=STEPS_PER_EPOCH,
                          callbacks=[cb,save_callback],
                          validation_data=test_dataset)
    printHistory(history, 1)


if __name__ == "__main__":
    main()
