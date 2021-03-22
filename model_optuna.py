import os
import pathlib
import time

import matplotlib.pyplot as plt
import numpy as np
import optuna
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

# data_dir = pathlib.Path('/data/s2709198/vox1/dataset')
# mean (~100k) + 3stddevs (3*~80k)
# SAMPLE_SELECTION = 340000
data_dir = pathlib.Path('dataset_split')
SAMPLE_SELECTION = 80000

commands = np.array(tf.io.gfile.listdir(str(data_dir)))
commands = commands[commands != 'README.md']
print('Commands:', commands)

filenames = tf.io.gfile.glob(str(data_dir) + '/*/*/*/*')
filenames = tf.random.shuffle(filenames)
num_samples = len(filenames)

print('Number of total examples:', num_samples)
print('Example file tensor:', filenames[0])

train_size = int(0.6 * num_samples)
test_size = int(0.2 * num_samples)

train_files = filenames[:train_size]
val_files = filenames[train_size: train_size + test_size]
test_files = filenames[-test_size:]

print('Training set size', len(train_files))
print('Validation set size', len(val_files))
print('Test set size', len(test_files))


def decode_audio(audio_binary):
    audio, _ = tf.audio.decode_wav(audio_binary, desired_samples=SAMPLE_SELECTION)
    return tf.squeeze(audio, axis=-1)


def get_label(file_path):
    parts = tf.strings.split(file_path, os.path.sep)
    return parts[-4]
    # The general format is "PATH/TO/DATA/language/speaker_id/youtubecode/[0-9]*.wav

    # So, to get the speaker id, we get the third entry as counted from the end.
    # return tf.strings.regex_replace(input=parts[-3], pattern="id[0]*", rewrite="")


def get_waveform_and_label(file_path):
    label = get_label(file_path)
    audio_binary = tf.io.read_file(file_path)
    waveform = decode_audio(audio_binary)
    return waveform, label


AUTOTUNE = tf.data.experimental.AUTOTUNE
files_ds = tf.data.Dataset.from_tensor_slices(train_files)
waveform_ds = files_ds.map(get_waveform_and_label, num_parallel_calls=AUTOTUNE)


def get_spectrogram(waveform):
    # Padding for files with less than SAMPLE_SELECTION samples
    zero_padding = tf.zeros([SAMPLE_SELECTION] - tf.shape(waveform), dtype=tf.float32)

    # Concatenate audio with padding so that all audio clips will be of the
    # same length
    waveform = tf.cast(waveform, tf.float32)
    equal_length = tf.concat([waveform, zero_padding], 0)
    spectrogram = tf.signal.stft(
        equal_length, frame_length=255, frame_step=128)

    spectrogram = tf.abs(spectrogram)

    return spectrogram


for waveform, label in waveform_ds.take(1):
    label = label.numpy().decode('utf-8')
    spectrogram = get_spectrogram(waveform)


def get_spectrogram_and_label_id(audio, label):
    spectrogram = get_spectrogram(audio)
    spectrogram = tf.expand_dims(spectrogram, -1)
    label_id = tf.argmax(label == commands)
    return spectrogram, label_id


spectrogram_ds = waveform_ds.map(
    get_spectrogram_and_label_id, num_parallel_calls=AUTOTUNE)


def preprocess_dataset(files):
    files_ds = tf.data.Dataset.from_tensor_slices(files)
    output_ds = files_ds.map(get_waveform_and_label, num_parallel_calls=AUTOTUNE)
    output_ds = output_ds.map(
        get_spectrogram_and_label_id, num_parallel_calls=AUTOTUNE)
    return output_ds


train_ds = spectrogram_ds
val_ds = preprocess_dataset(val_files)
test_ds = preprocess_dataset(test_files)

batch_size = 64
train_ds = train_ds.batch(batch_size)
val_ds = val_ds.batch(batch_size)

train_ds = train_ds.cache().prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)

for spectrogram, _ in spectrogram_ds.take(1):
    input_shape = spectrogram.shape
print('Input shape:', input_shape)
num_labels = len(commands)


def define_model(trial):
    n_layers = trial.suggest_int("n_layers", low=3, high=10, step=1)

    norm_layer = preprocessing.Normalization()
    norm_layer.adapt(spectrogram_ds.map(lambda x, _: x))

    model = models.Sequential()
    model.add(layers.Input(shape=input_shape))
    model.add(preprocessing.Resizing(32, 32))
    model.add(norm_layer)

    for i in range(n_layers):
        filter_size = trial.suggest_int("n_units_l{}".format(i), low=32, high=512)
        model.add(layers.Conv2D(filter_size, 3, activation='relu', padding='same'))

        dropout_value = trial.suggest_float("dropout_{}".format(i), 0.0, 0.5)
        model.add(layers.Dropout(dropout_value))

        # There's already a maxpooling layer after this loop, so don't add a double one.
        if i == (n_layers - 1):
            break

        add_max_pooling = trial.suggest_int("maxpool_{}".format(i), 0, 1)
        if add_max_pooling == 1:
            model.add(layers.MaxPooling2D(padding='same'))

    model.add(layers.MaxPooling2D(padding='same'))

    dropout_value = trial.suggest_float("dropout_beforelast", 0.1, 0.5)
    model.add(layers.Dropout(dropout_value))

    model.add(layers.Flatten())

    dense_size = trial.suggest_int("n_units_last", low=64, high=512)
    model.add(layers.Dense(dense_size, activation='relu'))

    dropout_value = trial.suggest_float("dropout_last", 0.1, 0.5)
    model.add(layers.Dropout(dropout_value))
    model.add(layers.Dense(num_labels, activation='relu'))

    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )
    return model


def run_model(model):
    EPOCHS = 32
    history = model.fit(
        train_ds,
        verbose=0,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=4),
    )

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
    print(f'Test set accuracy: {test_acc:.0%}')
    return test_acc


def objective(trial):
    start_time = time.time()
    model = define_model(trial)
    accuracy = run_model(model)
    run_time = time.time() - start_time
    return accuracy


def print_all_trials(study):
    print("Number of finished trials: ", len(study.trials))
    for trial in study.trials:
        print("Trial {}: {:.2f}".format(trial.number, trial.values[0]))


def print_best_callback(study, trial):
    print(f"Best value: {study.best_value}, Best params: {study.best_trial.params}")
    print_all_trials(study)
    print("==============================")


study = optuna.create_study(directions=["maximize"])
study.optimize(objective, n_trials=200, callbacks=[print_best_callback])

print_all_trials(study)
