import argparse
import contextlib
import json
import math
import os
import wave

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from pydub import AudioSegment
from pydub.silence import split_on_silence
from sklearn.model_selection import train_test_split
from tensorflow import keras


def main(args):
    signal, sr = librosa.load(args.example_file, sr=22050)
    display_sample_file(signal, sr)
    display_signal_spectrum(signal, sr)
    display_ft_spectogram(signal, sr)
    save_mfcc(args)
    inputs, targets = load_data(args.dataset_path)
    inputs = np.asarray(inputs).astype(np.object)
    targets = np.asarray(targets).astype(np.int)
    inputs_train, inputs_test, targets_train, targets_test = train_test_split(inputs, targets, test_size=0.2)
    model = keras.Sequential(
        [
            keras.layers.Flatten(input_shape=inputs.shape[0:]),
            keras.layers.Dense(512, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(256, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(64, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(5, activation="softmax"),
        ]
    )
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    history = model.fit(
        inputs_train, targets_train, validation_data=(inputs_test, targets_test), epochs=50, batch_size=32
    )

    plot_history(history)


def display_sample_file(signal, sr):
    librosa.display.waveplot(signal, sr=sr)
    plt.xlabel("Time")
    plt.ylabel("Amplitute")
    plt.show()


def display_signal_spectrum(signal, sr):
    fft_value = np.fft.fft(signal)
    magnitude = np.aps(fft_value)
    frequency = np.linspace(0, sr, len(magnitude))

    plt.plot(frequency, magnitude)
    plt.xlabel("Frequency")
    plt.ylabel("Magnitude")
    plt.show()

    left_frequency = frequency[: int(len(frequency) / 2)]
    left_magnitude = magnitude[: int(len(magnitude) / 2)]

    plt.plot(left_frequency, left_magnitude)
    plt.xlabel("Left Frequency")
    plt.ylabel("left Magnitude")
    plt.show()


def display_ft_spectogram(signal, sr):
    n_fft = 2048
    hop_length = 512
    stft = librosa.core.stft(signal, hop_length=hop_length, n_fft=n_fft)
    spectogram = np.abs(stft)
    log_spectogram = librosa.amplitude_to_db(spectogram)
    librosa.display.specshow(log_spectogram, sr=sr, hop_length=hop_length)
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.colorbar()
    plt.show()

    MFFCs = librosa.feature.mfcc(signal, n_fft=n_fft, hop_length=hop_length, n_mfcc=13)
    librosa.display.specshow(
        MFFCs,
        sr=sr,
        hop_length=hop_length,
    )
    plt.xlabel("Time")
    plt.ylabel("MFCC")
    plt.colorbar()
    plt.show()


def save_mfcc(args, n_mfcc=13, n_fft=2048, hop_length=512):
    data = {
        "mapping": [],
        "mfcc": [],
        "labels": [],
    }

    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(args.data_dir)):

        if dirpath is not args.data_dir:
            dirpath_components = dirpath.split("/")
            semantic_label = dirpath_components[-1]
            data["mappings"].append(semantic_label)
            print("\nProcessing {}".format(semantic_label))

            for f in filenames:
                file_path = os.path.join(dirpath, f)
                signal, sr = librosa.load(file_path, sr=args.sample_rate)
                with contextlib.closing(wave.open(file_path, "r")) as file:
                    frames = file.getnframes()
                    rate = file.getframerate()
                    duration = frames / float(rate)
                    DURATION = duration

                sound = AudioSegment.from_file(file_path, format="wav")
                dBFS = sound.dBFS
                chunks = split_on_silence(
                    sound,
                    min_silence_len=1000,
                    silence_thresh=dBFS - 16,
                    keep_silence=200,
                )

                target_length = 120 * 1000
                output_chunks = [chunks[0]]
                for chunk in chunks[1:]:
                    if len(output_chunks[-1]) < target_length:
                        output_chunks[-1] += chunk
                    else:
                        output_chunks.append(chunk)

                num_segments = len(output_chunks)
                SAMPLES_PER_TRACK = args.sample_rate * DURATION
                num_samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
                expected_num_mfcc_vectors_per_segment = math.ceil(num_samples_per_segment / hop_length)

                for s in range(num_segments):
                    start_sample = num_samples_per_segment * s
                    finish_sample = start_sample + num_samples_per_segment

                    mfcc = librosa.feature.mfcc(
                        signal[start_sample:finish_sample], sr=sr, n_fft=n_fft, n_mfcc=n_mfcc, hop_length=hop_length
                    )
                    mfcc = mfcc.T  # Transpose

                    if len(mfcc) == expected_num_mfcc_vectors_per_segment:
                        data["mfcc"].append(mfcc.tolist())
                        data["labels"].append(i - 1)
                        print("{}, segment:{}".format(file_path, s))

    with open(args.dataset_path, "w") as fp:
        json.dump(data, fp, indent=4)


def load_data(path):
    with open(path, "r") as fp:
        data = json.load(fp)
    inputs = np.array(data["mfcc"], dtype="object")
    targets = np.array(data["labels"], dtype="object")

    return inputs, targets


def plot_history(history):
    fig, axs = plt.subplot(2)

    axs[0].plot(history.history["accuracy"], label="Train Accuracy")
    axs[0].plot(history.history["val_accuracy"], label="Test Accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legent(loc="lower right")
    axs[0].set_title("Accuracy eval")

    axs[1].plot(history.history["loss"], label="Train Error")
    axs[1].plot(history.history["val_loss"], label="Test Error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legent(loc="upper right")
    axs[1].set_title("Error eval")

    plt.show()


if __name__ == "__name__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--example-file", type=str, default="audio/JOR/BmxpV9bna6A.wav")
    parser.add_argument("--data-dir", type=str, default="audio")
    parser.add_argument("--dataset-path", type=str, default="data.json")
    parser.add_argument("sample-rate", type=int, default=22050)

    args = parser.parse_args()
    main(args)
