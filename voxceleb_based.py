import os
import pathlib
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers
from tensorflow.keras import models

ENABLE_GPU = True
if ENABLE_GPU:
    physical_devices = tf.config.list_physical_devices("GPU")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Set seed for experiment reproducibility
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

DATA_DIR = "dataset_split_2"
output_path = "output8_voxceleb"
SAMPLE_SELECTION = 80000  # 5 sec
BATCH_SIZE = 64
EPOCHS = 32

if not os.path.isdir(output_path):
    os.makedirs(output_path)

USE_MEL_LOG_SPECTROGRAM = True
AUTO_TUNE = tf.data.experimental.AUTOTUNE


def get_data():
    data_dir = pathlib.Path(DATA_DIR)

    commands = np.array(tf.io.gfile.listdir(str(data_dir) + "/train"))
    print('Commands:', commands)

    train_files = get_file_names(str(data_dir) + "/train")
    val_files = get_file_names(str(data_dir) + "/test")
    test_files = get_file_names(str(data_dir) + "/validate")

    num_samples = len(train_files) + len(val_files) + len(test_files)

    print('Number of total examples:', num_samples)
    print('Training set size', len(train_files))
    print('Validation set size', len(val_files))
    print('Test set size', len(test_files))

    train_ds, input_shape = preprocess_dataset(train_files, commands, True)
    val_ds, _ = preprocess_dataset(val_files, commands, True)
    test_ds, _ = preprocess_dataset(test_files, commands, False)

    print('Input shape:', input_shape)
    num_labels = len(commands)

    return train_ds, val_ds, test_ds, num_labels, input_shape


def get_file_names(subdir):
    filenames = tf.io.gfile.glob(subdir + '/*/*/*/*')
    filenames = tf.random.shuffle(filenames)
    return filenames


def decode_audio(audio_binary):
    audio, _ = tf.audio.decode_wav(audio_binary, desired_samples=SAMPLE_SELECTION)
    return tf.squeeze(audio, axis=-1)


def get_label(file_path):
    # The general format is "PATH/TO/DATA/language/speaker_id/youtubecode/[0-9]*.wav
    parts = tf.strings.split(file_path, os.path.sep)
    return parts[-4]


def get_waveform_and_label(file_path):
    label = get_label(file_path)
    audio_binary = tf.io.read_file(file_path)
    waveform = decode_audio(audio_binary)
    return waveform, label


# This function applies a mel filterbank to the spectrogram
def transform_to_mel_spectogram(spectrogram, stft):
    num_spectrogram_bins = stft.shape[-1]
    lower_edge_hertz, upper_edge_hertz, num_mel_bins = 80.0, 7600.0, 80
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins, num_spectrogram_bins, SAMPLE_SELECTION, lower_edge_hertz,
        upper_edge_hertz)
    mel_spectrograms = tf.tensordot(spectrogram, linear_to_mel_weight_matrix, 1)
    mel_spectrograms.set_shape(spectrogram.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))

    # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
    log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)

    return log_mel_spectrograms


def get_spectrogram(waveform: tf.data.Dataset):
    # Padding for files with less than SAMPLE_SELECTION samples
    zero_padding = tf.zeros([SAMPLE_SELECTION] - tf.shape(waveform), dtype=tf.float32)

    # Concatenate audio with padding so that all audio clips will be of the
    # same length
    waveform = tf.cast(waveform, tf.float32)
    equal_length = tf.concat([waveform, zero_padding], 0)
    stft = tf.signal.stft(equal_length, frame_length=255, frame_step=128)

    spectrogram = tf.abs(stft)

    if USE_MEL_LOG_SPECTROGRAM:
        spectrogram = transform_to_mel_spectogram(spectrogram, stft)

    return spectrogram


def get_spectrogram_and_label_id(audio, label, commands):
    spectrogram = get_spectrogram(audio)
    spectrogram = tf.expand_dims(spectrogram, -1)
    label_id = tf.argmax(label == commands)
    return spectrogram, label_id


def preprocess_dataset(files, commands, cache):
    files_ds = tf.data.Dataset.from_tensor_slices(files)
    output_ds = files_ds.map(get_waveform_and_label, num_parallel_calls=AUTO_TUNE)
    output_ds = output_ds.map(lambda audio, label: get_spectrogram_and_label_id(audio, label, commands),
                              num_parallel_calls=AUTO_TUNE)

    dataset_shape = None
    for spectrogram, _ in output_ds.take(1):
        dataset_shape = spectrogram.shape
    if dataset_shape is None:
        raise ValueError("Input shape cannot be None!")

    if cache:
        output_ds = output_ds.batch(BATCH_SIZE)
        output_ds = output_ds.cache().prefetch(AUTO_TUNE)
    return output_ds, dataset_shape


def define_model(input_shape, ds_train, num_labels):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(96, (7, 7), strides=(1, 1), activation='relu'),
        layers.MaxPooling2D(3),
        layers.Conv2D(256, (5, 5), strides=(1, 1), activation='relu'),
        layers.MaxPooling2D(2),
        layers.Conv2D(384, (3, 3), strides=(1, 1), activation='relu'),
        layers.Conv2D(256, (3, 3), strides=(1, 1), activation='relu'),
        layers.Conv2D(256, (3, 3), strides=(1, 1), activation='relu'),
        layers.MaxPooling2D((3,2)),
        layers.Dense(4096, activation='relu'),
        layers.GlobalAveragePooling2D(),
        layers.Dense(1024, activation='relu'),
    ])

    model.summary()

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-2,
        decay_steps=10000,
        decay_rate=0.9)
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=lr_schedule),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )
    return model


def run_model(model, train_ds, val_ds, test_ds, run: int):
    history = model.fit(
        train_ds,
        verbose=1,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=8),
    )

    model.save(output_path + "/{}".format(run))

    test_audio = []
    test_labels = []

    for audio, label in test_ds:
        test_audio.append(audio.numpy())
        test_labels.append(label.numpy())

    test_audio = np.array(test_audio)
    test_labels = np.array(test_labels)

    y_pred = np.argmax(model.predict(test_audio), axis=1)
    y_true = test_labels

    test_acc = sum(y_pred == y_true) / len(y_true)
    print(f'Run {run}: Test set accuracy: {test_acc:.0%}')

    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]

    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    epochs_range = range(len(acc))
    plt.figure(figsize=(9, 9))
    plt.suptitle("Test Accuracy: {}".format(test_acc))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Training Accuracy")
    plt.plot(epochs_range, val_acc, label="Validation Accuracy")
    plt.legend(loc="lower right")
    plt.title("Training and Validation Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Training Loss")
    plt.plot(epochs_range, val_loss, label="Validation Loss")
    plt.legend(loc="upper right")
    plt.title("Training and Validation Loss")

    plt.savefig(output_path + "/{}.pdf".format(run))
    print("\nRun: {}, test_acc: {}\nacc: {}\nval_acc: {}\nloss: {}\nval_loss:{}\n".format(run, test_acc, acc,
                                                                                          val_acc, loss, val_loss))

    return test_acc


def run_experiment(run: int):
    train_ds, val_ds, test_ds, num_labels, input_shape = get_data()
    model = define_model(input_shape, train_ds, num_labels)
    accuracy = run_model(model, train_ds, val_ds, test_ds, run)
    return accuracy


if __name__ == "__main__":
    for run in range(0, 10):
        run_experiment(run)
