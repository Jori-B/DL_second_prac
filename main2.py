import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers
from tensorflow.keras import models
from IPython import display

# Set seed for experiment reproducibility
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

data_dir = pathlib.Path('dataset')

commands = np.array(tf.io.gfile.listdir(str(data_dir)))
commands = commands[commands != '.DS_Store']
print('Commands:', commands)

filenames = tf.io.gfile.glob(str(data_dir) + '/*/*/*/*')
filenames = tf.random.shuffle(filenames)
num_samples = len(filenames)

print('Number of total examples:', num_samples)
print('Number of examples per label:',
      len(tf.io.gfile.listdir(str(data_dir / commands[0]))))
print('Example file tensor:', filenames[0])

num_samples = len(filenames)
print('Number of total examples:', num_samples)
print('Example file tensor:', filenames[0])

train_size = int(0.6 * num_samples)
test_size = int(0.2 * num_samples)

train_files = filenames[:train_size]
val_files = filenames[train_size: train_size + test_size]
test_files = filenames[-test_size:]

print('total test size ', len(filenames))
print('Training set size', len(train_files))
print('Validation set size', len(val_files))
print('Test set size', len(test_files))

SAMPLE_SELECTION = 16000


def decode_audio(audio_binary):
    audio, _ = tf.audio.decode_wav(audio_binary, desired_samples=SAMPLE_SELECTION)
    return tf.squeeze(audio, axis=-1)


def get_label(file_path):
    parts = tf.strings.split(file_path, os.path.sep)
    return parts[-4]

def get_waveform_and_label(file_path):
    label = get_label(file_path)
    audio_binary = tf.io.read_file(file_path)
    waveform = decode_audio(audio_binary)
    return waveform, label

AUTOTUNE = tf.data.AUTOTUNE
files_ds = tf.data.Dataset.from_tensor_slices(train_files)
waveform_ds = files_ds.map(get_waveform_and_label, num_parallel_calls=AUTOTUNE)

rows = 3
cols = 3
n = rows * cols
fig, axes = plt.subplots(rows, cols, figsize=(10, 12))
for i, (audio, label) in enumerate(waveform_ds.take(n)):
    r = i // cols
    c = i % cols
    ax = axes[r][c]
    ax.plot(audio.numpy())
    ax.set_yticks(np.arange(-1.2, 1.2, 0.2))
    label = label.numpy().decode('utf-8')
    ax.set_title(label)

#plt.show()

#This function applies a mel filterbank to the spectrogram
def transform_to_mel_spectogram(spectrogram, stft):
    num_spectrogram_bins = stft.shape[-1]
    lower_edge_hertz, upper_edge_hertz, num_mel_bins = 80.0, 7600.0, 80
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins, num_spectrogram_bins, 16000, lower_edge_hertz,
        upper_edge_hertz)
    mel_spectrograms = tf.tensordot(
        spectrogram, linear_to_mel_weight_matrix, 1)
    mel_spectrograms.set_shape(spectrogram.shape[:-1].concatenate(
        linear_to_mel_weight_matrix.shape[-1:]))

    # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
    log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)
    return mel_spectrograms

def get_spectrogram(waveform: tf.data.Dataset):
    # Padding for files with less than SAMPLE_SELECTION samples
    zero_padding = tf.zeros([SAMPLE_SELECTION] - tf.shape(waveform), dtype=tf.float32)

    # Concatenate audio with padding so that all audio clips will be of the
    # same length
    waveform = tf.cast(waveform, tf.float32)
    equal_length = tf.concat([waveform, zero_padding], 0)
    stft = tf.signal.stft(
        equal_length, frame_length=255, frame_step=128)


    spectrogram = tf.abs(stft)

    mel_spectrogram = transform_to_mel_spectogram(spectrogram, stft)

    return mel_spectrogram


for waveform, label in waveform_ds.take(1):
    label = label.numpy().decode('utf-8')
    spectrogram = get_spectrogram(waveform)


display.display(display.Audio(waveform, rate=SAMPLE_SELECTION))


# def plot_spectrogram(spectrogram, ax):
#     # Convert to frequencies to log scale and transpose so that the time is
#     # represented in the x-axis (columns).
#     log_spec = np.log(spectrogram.T)
#     height = log_spec.shape[0]
#     X = np.arange(SAMPLE_SELECTION, step=height + 1)
#     Y = range(height)
#     print(X)
#     print(Y)
#     print(log_spec)
#     ax.pcolormesh(X, Y, log_spec)


# fig, axes = plt.subplots(2, figsize=(12, 8))
# timescale = np.arange(waveform.shape[0])
# axes[0].plot(timescale, waveform.numpy())
# axes[0].set_title('Waveform')
# axes[0].set_xlim([0, SAMPLE_SELECTION])
# plot_spectrogram(spectrogram.numpy(), axes[1])
# axes[1].set_title('Spectrogram')
# plt.show()


def get_spectrogram_and_label_id(audio, label):
    spectrogram = get_spectrogram(audio)
    spectrogram = tf.expand_dims(spectrogram, -1)
    label_id = tf.argmax(label == commands)
    return spectrogram, label_id


spectrogram_ds = waveform_ds.map(
    get_spectrogram_and_label_id, num_parallel_calls=AUTOTUNE)


# rows = 3
# cols = 3
# n = rows * cols
# fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
# for i, (spectrogram, label_id) in enumerate(spectrogram_ds.take(n)):
#     r = i // cols
#     c = i % cols
#     ax = axes[r][c]
#     plot_spectrogram(np.squeeze(spectrogram.numpy()), ax)
#     ax.set_title(commands[label_id.numpy()])
#     ax.axis('off')
#
# plt.show()


def preprocess_dataset(files):
    files_ds = tf.data.Dataset.from_tensor_slices(files)
    output_ds = files_ds.map(get_waveform_and_label, num_parallel_calls=AUTOTUNE)
    output_ds = output_ds.map(
        get_spectrogram_and_label_id, num_parallel_calls=AUTOTUNE)
    return output_ds


train_ds = spectrogram_ds
val_ds = preprocess_dataset(val_files)
test_ds = preprocess_dataset(test_files)

batch_size = 128
train_ds = train_ds.batch(batch_size)
val_ds = val_ds.batch(batch_size)

train_ds = train_ds.cache().prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)

for spectrogram, _ in spectrogram_ds.take(1):
    input_shape = spectrogram.shape

num_labels = len(commands)

norm_layer = preprocessing.Normalization()
norm_layer.adapt(spectrogram_ds.map(lambda x, _: x))

model = models.Sequential([
    layers.Input(shape=input_shape),
    preprocessing.Resizing(32, 32),
    norm_layer,
    layers.Conv2D(16, 4, activation='relu'),
    layers.Conv2D(32, 4, activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_labels),
])

model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
)

EPOCHS = 10
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2),
)

metrics = history.history
#plt.plot(history.epoch, metrics['loss'], metrics['val_loss'])
#plt.legend(['loss', 'val_loss'])
#plt.show()

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

confusion_mtx = tf.math.confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_mtx, xticklabels=commands, yticklabels=commands,
            annot=True, fmt='g')
plt.xlabel('Prediction')
plt.ylabel('Label')
plt.show()

# sample_file = data_dir / 'no/01bb6a2a_nohash_0.wav'
#
# sample_ds = preprocess_dataset([str(sample_file)])
#
# for spectrogram, label in sample_ds.batch(1):
#     prediction = model(spectrogram)
#     plt.bar(commands, tf.nn.softmax(prediction[0]))
#     plt.title(f'Predictions for "{commands[label[0]]}"')
#     plt.show()
