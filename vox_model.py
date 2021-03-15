import numpy as np
import time
import pandas as pd
import matplotlib.pylab as plt
import tensorflow as tf
import tensorflow_hub as hub
import os
import glob
import keras
import sys
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from timeit import default_timer as timer
from os.path import expanduser

home = expanduser("~")
path_data = f'{home}/Downloads/vox1_wav'
path_csv = f'{home}/Downloads/'
num_epoch = 1
BATCH_SIZE = 1
label_encoder = preprocessing.LabelEncoder()

class nationality_checker:
    def __init__(self, path_csv):
        try:
            self.data = pd.read_csv(f'{path_csv}/vox1_meta.csv')
        except OSError:
            print("Could not open/read file vox1_meta.csv")
            sys.exit()

    def get_nationality(self, id):
        label = self.data.loc[self.data['VoxCeleb1_ID'] == id]['Nationality'].values[0]
        if not(label):
            label ='USA'
        return label

def decode_audio(audio_binary, label):
    audio = tf.io.read_file(audio_binary)
    audio, _ = tf.audio.decode_wav(audio) # _ here means a throwable variable as we don't use it
    return tf.squeeze(audio, axis=-1), label




# https://stackoverflow.com/questions/57181551/can-i-write-a-keras-callback-that-records-and-returns-the-total-training-time/57182112
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
    nationality_per_id = nationality_checker(path_csv)
    list_of_files = []
    labels = []
    # get a list of all the folder names (f.path for path) in the directory specified in path
    list_subfolders_with_paths = [f.name for f in os.scandir(path_data) if f.is_dir()]
    for folder in list_subfolders_with_paths:
        id_path = (f"{path_data}/{folder}")
        list_ID_subfolders_with_paths = [f.name for f in os.scandir(id_path) if f.is_dir()]
        for id in list_ID_subfolders_with_paths:
            for filename in glob.iglob(f"{id_path}/{id}/**/*.*", recursive=True):
                list_of_files.append(filename)
                label = nationality_per_id.get_nationality(id)
                labels.append(id)

    #file_train, file_test, label_train, label_test = train_test_split(list_of_files, labels, test_size=0.1)
    filenames = tf.constant(list_of_files)


    label_trans = label_encoder.fit(labels)  # fit a transformer that transforms label strings to numerical
    labels = label_trans.transform(labels)  # transforms label train into numerical
    #label_test = label_trans.transform(label_test)  # transforms label test into numerical

    # set all the data to tensorflow constants
    #filenames_train = tf.constant(file_train)
    #labels_train = tf.constant(label_train)
    #filenames_test = tf.constant(file_test)
    #labels_test = tf.constant(label_test)

    # creates the train dataset
    dataset_train = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset_train = dataset_train.map(decode_audio)
    train_dataset = dataset_train.cache().shuffle(len(filenames)).batch(BATCH_SIZE).repeat()
    train_dataset = train_dataset.prefetch(buffer_size=1)

    # creates the test dataset
#    dataset_test = tf.data.Dataset.from_tensor_slices((filenames_test, labels_test))
#    dataset_test = dataset_test.map(im_file_to_tensor)
#    predict =dataset_test.batch(1)
#    test_dataset = dataset_test.cache().shuffle(len(file_test)).batch(BATCH_SIZE)
#    train_datasetPredict = dataset_test.batch(32)
    for audio_batch, labels_batch in train_dataset:
        print(image_batch.shape)
        print(labels_batch.shape)
        break
    #test_dataset = test_dataset.prefetch(buffer_size=8)


    # calculates how many epoch are needed to run over the whole dataset
#    STEPS_PER_EPOCH = len(file_train)/BATCH_SIZE
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




if __name__ == "__main__":
    main()
