import os
import pathlib

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

DATA_DIR = "dataset_split_2"
output_path = "output"
SAMPLE_SELECTION = 80000  # 5 sec
BATCH_SIZE = 32

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
    filenames = tf.io.gfile.glob(subdir + '/*/*/*/*.wav')
    filenames = tf.random.shuffle(filenames)
    return filenames


def decode_audio(audio_binary):
    audio, _ = tf.audio.decode_wav(audio_binary, desired_samples=SAMPLE_SELECTION)
    return tf.squeeze(audio, axis=-1)


def get_label(file_path):
    parts = tf.strings.split(file_path, os.path.sep)
    return parts[-4]
    # The general format is "PATH/TO/DATA/language/speaker_id/youtubecode/[0-9]*.wav


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

    # return mel_spectrograms
    return log_mel_spectrograms


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
    spectrogram = transform_to_mel_spectogram(spectrogram, spectrogram)

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


def define_model(trial, input_shape, ds_train, num_labels):
    n_layers = trial.suggest_int("n_layers", low=3, high=10, step=1)

    norm_layer = preprocessing.Normalization()
    norm_layer.adapt(ds_train.map(lambda x, _: x))

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


def run_model(model, train_ds, val_ds, test_ds, trial_id):
    EPOCHS = 2
    history = model.fit(
        train_ds,
        verbose=1,
        validation_data=val_ds,
        epochs=EPOCHS,
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

    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]

    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    print("\nModel: {}\nacc: {}\nval_acc: {}\nloss: {}\nval_loss:{}\n".format(trial_id, acc, val_acc, loss, val_loss))

    epochs_range = range(len(acc))

    plt.figure(figsize=(9, 9))
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
    plt.savefig(output_path + "/" + str(trial_id) + ".pdf")

    return test_acc


def print_all_trials(study):
    print("Number of finished trials: ", len(study.trials))
    for trial in study.trials:
        print("Trial {}: {:.2f}\nparams: {}".format(trial.number, trial.values[0], trial.params))


def print_best_callback(study, trial):
    print(f"Best value: {study.best_value}, Best params: {study.best_trial.params}, "
          f"ID: {study.best_trial._trial_id}, value: {study.best_trial.value}")
    print_all_trials(study)
    print("==============================")


def objective(trial, train_ds, val_ds, test_ds, input_shape, num_labels):
    model = define_model(trial, input_shape, train_ds, num_labels)
    accuracy = run_model(model, train_ds, val_ds, test_ds, trial._trial_id)
    return accuracy


def run_experiment():
    train_ds, val_ds, test_ds, num_labels, input_shape = get_data()
    study = optuna.create_study(directions=["maximize"])
    study.optimize(lambda trial: objective(trial, train_ds, val_ds, test_ds, input_shape, num_labels),
                   n_trials=100, callbacks=[print_best_callback])

    print_all_trials(study)


if __name__ == "__main__":
    run_experiment()
