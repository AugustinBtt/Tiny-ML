import random
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import librosa
import os
import numpy as np
import glob
import re
import shutil
import subprocess
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError



# download the extract loudest section program (to be compiled): 'https://github.com/petewarden/extract_loudest_section.git'
KWS_path = 'Path to your project'
audio_files_directory = 'Path to all of the audio files'
my_words_dir = os.path.join(audio_files_directory, 'Words')  # Your target words in .ogg format
background_noise_dir = os.path.join(audio_files_directory, 'background_noise')  # background noises in .wav format

binary_path = 'Path to the extract loudest section program binary code'

file_name = 'speech_commands_v0.02.tar.gz'
downloaded_file_path = os.path.join(background_noise_dir, file_name)

if not os.path.exists(background_noise_dir):
    os.makedirs(background_noise_dir)
    print("background noise dir created")

dataset_directory = os.path.join(audio_files_directory, 'target_dataset')
if not os.path.exists(dataset_directory):
    os.makedirs(dataset_directory)
    print('final dataset dir created')


if os.path.exists(my_words_dir):
    # URL for target words or background noises: 'https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz'

    # iterate over the content of the folder (in case you have several target words)
    for word_folder in os.listdir(my_words_dir):
        folder_path = os.path.join(my_words_dir, word_folder)

        if os.path.isdir(folder_path):
            wav_files_dir = os.path.join(folder_path, 'wav_files')
            converted_wav_dir = os.path.join(folder_path, '16_bit')
            trimmed_wav_dir = os.path.join(folder_path, 'trimmed_wavs')
            # create the 'wav_files' and 'trimmed_wavs' subfolders if they don't exist
            os.makedirs(wav_files_dir, exist_ok=True)
            os.makedirs(trimmed_wav_dir, exist_ok=True)

            if not os.listdir(wav_files_dir):  # checking if files have been converted to wav already
                print(f"{wav_files_dir} is empty. Proceeding with conversion...")
                for file_name in os.listdir(folder_path):
                    if file_name.endswith('.ogg'):
                        ogg_path = os.path.join(folder_path, file_name)
                        wav_path = os.path.join(wav_files_dir, os.path.splitext(file_name)[0] + '.wav')
                        try:
                            # convert ogg to wav
                            audio = AudioSegment.from_ogg(ogg_path)
                            audio.export(wav_path, format="wav")
                            print(f"Converted {file_name} to .wav")
                        except CouldntDecodeError as e:
                            print(f"Failed to convert {file_name}. Error: {e}")
                    else:
                        print(f"Skipping non-.ogg file: {file_name}")
            else:
                print(f"{wav_files_dir} is not empty. Skipping conversion.")

            if not os.listdir(trimmed_wav_dir):
                print('Converting to 16 bits')
                if not os.path.exists(converted_wav_dir):
                    os.makedirs(converted_wav_dir)
                    print(f"Created directory: {converted_wav_dir}")

                wav_files = glob.glob(os.path.join(wav_files_dir, "*.wav"))
                for wav_file in wav_files:
                    converted_file_path = os.path.join(converted_wav_dir, os.path.basename(wav_file))
                    subprocess.run([
                                'ffmpeg', '-i', wav_file, '-acodec', 'pcm_s16le', '-ar', '16000', converted_file_path
                            ], check=True)
                    print(f"Converted {wav_file} to 16-bit: {converted_file_path}")

                # run the extract_loudest_section compiled program
                if os.path.exists(binary_path) and os.path.getsize(binary_path) > 0:
                    print("Running the extract_loudest_section tool on converted .wav files")

                    for wav_file in glob.glob(os.path.join(converted_wav_dir, "*.wav")):
                        subprocess.run([binary_path, wav_file, trimmed_wav_dir])
                    print('Loudest sections extracted successfully')
                else:
                    print("Binary is missing or not executable. Check the path and permissions.")

                data_index = {}
                os.chdir(trimmed_wav_dir)
                search_path = os.path.join(trimmed_wav_dir, '*.wav')  # search for all .wav

                # indexing the files based on their names
                for wav_path in glob.glob(search_path):
                    original_wav_path = wav_path
                    parts = os.path.basename(wav_path).split('_')
                    if len(parts) > 2:
                        wav_path = parts[0] + '_' + ''.join(parts[1:])
                    matches = re.search(r'([^/_]+)_([^/_]+)\.wav', wav_path)
                    if not matches:
                        raise Exception(f'File name not in a recognized form: "{wav_path}"')
                    word = matches.group(1).lower()
                    instance = matches.group(2).lower()
                    if word not in data_index:
                        data_index[word] = {}
                    if instance in data_index[word]:
                        raise Exception(f'Audio instance already seen: "{wav_path}"')
                    data_index[word][instance] = original_wav_path

                # create the dataset directory structure and move the target audio files to the corresponding folder
                os.makedirs(dataset_directory, exist_ok=True)
                for word in data_index:
                    word_dir = os.path.join(dataset_directory, word)
                    os.makedirs(word_dir, exist_ok=True)
                    for instance in data_index[word]:
                        wav_path = data_index[word][instance]
                        output_path = os.path.join(word_dir, f'{instance}.wav')
                        shutil.copyfile(wav_path, output_path)
            else:
                print(f"{trimmed_wav_dir} is not empty. Skipping trimming.")
        else:
            print(f"Skipping non-directory item: {word_folder}")


# MODEL AND TRAINING
print('Moving to training')
# Constants
TRAINING_STEPS = [6000, 1000]
LEARNING_RATE = [0.001, 0.0001]
BATCH_SIZE = 32
SAVE_STEP_INTERVAL = 1000

# Audio processing constants
WINDOW_STRIDE = 20
SAMPLE_RATE = 16000

WINDOW_SIZE_MS = 30.0
FEATURE_BIN_COUNT = 40
FEATURE_SLICE_COUNT = 49
FEATURE_ELEMENT_COUNT = FEATURE_BIN_COUNT * FEATURE_SLICE_COUNT

# Data split percentages
VALIDATION_PERCENTAGE = 0.1
TESTING_PERCENTAGE = 0.1

WANTED_WORDS = "Left"
number_of_labels = WANTED_WORDS.count(',') + 1
number_of_total_labels = number_of_labels + 2
equal_percentage_of_training_samples = int(100.0 / number_of_total_labels)
SILENT_PERCENTAGE = equal_percentage_of_training_samples
UNKNOWN_PERCENTAGE = equal_percentage_of_training_samples


def pitch_shift(audio, sample_rate):
    audio = np.array(audio)
    # shifting pitch between -2 and +2 semitones
    shift_steps = np.random.randint(-2, 3)
    return librosa.effects.pitch_shift(y=audio, sr=sample_rate, n_steps=shift_steps)

def add_noise(audio, noise_factor=0.005):
    noise = np.random.randn(len(audio))
    augmented_audio = audio + noise_factor * noise
    return augmented_audio

def shift_audio(audio, sample_rate, shift_max=0.2):
    # shifting the audio left or right
    shift = np.random.randint(int(sample_rate * -shift_max), int(sample_rate * shift_max))
    # fill the gaps with silence
    shifted_audio = np.roll(audio, shift)
    if shift > 0:
        shifted_audio[:shift] = 0
    else:
        shifted_audio[shift:] = 0
    return shifted_audio


def create_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Input(shape=input_shape),

        layers.Conv2D(8, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal'),
        layers.BatchNormalization(),

        layers.Flatten(),

        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.005)),
        layers.Dropout(0.3),

        layers.Dense(num_classes, activation='softmax')
    ])

    optimizer = tf.keras.optimizers.Adam(clipvalue=1.0)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def preprocess_audio(input_data, sample_rate, window_size_ms, window_stride_ms, feature_bin_count, fixed_length=49, augment=True):
    if isinstance(input_data, str):
        # load the file if input_data is a filepath
        audio = tf.io.read_file(input_data)
        waveform, _ = tf.audio.decode_wav(audio, desired_channels=1)
        waveform = tf.squeeze(waveform, axis=-1)
    elif isinstance(input_data, np.ndarray):
        waveform = tf.convert_to_tensor(input_data, dtype=tf.float32)
    else:
        raise ValueError("input_data must be a file path or a numpy array of audio samples")

    # normalizing audio waveform between -1 and 1
    max_val = tf.reduce_max(tf.abs(waveform))
    waveform = tf.divide(waveform, max_val + 1e-8)

    if augment:
        waveform_np = waveform.numpy()  # numpy array for librosa
        try:
            if np.random.rand() < 0.3:
                waveform_np = pitch_shift(waveform_np, sample_rate)
            if np.random.rand() < 0.3:
                waveform_np = add_noise(waveform_np)
            if np.random.rand() < 0.3:
                waveform_np = shift_audio(waveform_np, sample_rate)
        except Exception as e:
            print(f"Augmentation error: {e}")
            return None
        waveform = tf.convert_to_tensor(waveform_np, dtype=tf.float32)

    window_size_samples = int(sample_rate * window_size_ms / 1000)
    window_stride_samples = int(sample_rate * window_stride_ms / 1000)

    spectrogram = tf.signal.stft(
        waveform,
        frame_length=window_size_samples,
        frame_step=window_stride_samples,
        fft_length=window_size_samples,
        window_fn=tf.signal.hann_window
    )

    spectrogram = tf.abs(spectrogram)

    # clipping very small values
    spectrogram = tf.clip_by_value(spectrogram, clip_value_min=1e-10, clip_value_max=tf.reduce_max(spectrogram))

    # spectrogram to log-mel spectrogram
    mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=feature_bin_count,
        num_spectrogram_bins=tf.shape(spectrogram)[-1],
        sample_rate=sample_rate
    )
    mel_spectrogram = tf.tensordot(
        spectrogram,
        mel_weight_matrix,
        1
    )

    # make sure values are positive before applying log
    mel_spectrogram = tf.clip_by_value(mel_spectrogram, clip_value_min=1e-10, clip_value_max=tf.reduce_max(mel_spectrogram))

    # small constant to avoid log(0)
    log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)

    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)
    # clipping avoid large values
    mfccs = tf.clip_by_value(mfccs, -1.0, 1.0)

    # we want a fixed number of frames (pad or truncate)
    mfcc_shape = tf.shape(mfccs)

    if mfcc_shape[0] < fixed_length:
        mfccs_padded = tf.pad(mfccs, [[0, fixed_length - mfcc_shape[0]], [0, 0]])
    else:
        mfccs_padded = mfccs[:fixed_length, :]

    if tf.reduce_any(tf.math.is_nan(mfccs_padded)):
        print("NaNs found in padded/truncated MFCCs")
        return None

    return mfccs_padded


def load_data(data_dir, wanted_words, silent_percentage, unknown_percentage):
    wanted_words_list = wanted_words.split(',')
    X = []  # input features
    y = []  # labels

    # mapping from word to label
    word_to_label = {}
    current_label = 0

    # assign labels to wanted words
    for word in wanted_words_list:
        word_to_label[word.strip()] = current_label
        current_label += 1

    silence_label = current_label
    current_label += 1
    unknown_label = current_label

    skipped_samples = 0

    for word_folder in os.listdir(data_dir):
        word_dir = os.path.join(data_dir, word_folder)
        if os.path.isdir(word_dir):
            # label for word
            print(f'word to label: {word_to_label}')
            print('Loading...')
            if word_folder in word_to_label:
                label = word_to_label[word_folder]
            else:
                label = unknown_label  # unknown label to other words
            for filename in os.listdir(word_dir):
                if filename.endswith('.wav'):
                    try:
                        filepath = os.path.join(word_dir, filename)
                        mfcc = preprocess_audio(filepath, SAMPLE_RATE, WINDOW_SIZE_MS, WINDOW_STRIDE, FEATURE_BIN_COUNT, augment=True)

                        # skip invalid samples
                        if np.isnan(mfcc).sum() > 0 or np.isinf(mfcc).sum() > 0:
                            print(f"Skipping invalid sample: {filename}")
                            skipped_samples += 1
                            continue  # skip this sample

                        X.append(mfcc)
                        y.append(label)
                    except Exception as e:
                        print(f"Error processing file {filename}: {e}")
                        skipped_samples += 1
        else:
            print(f"Skipping non-directory item: {word_dir}")

    # empty segments for silence
    num_silence_samples = int(len(X) * silent_percentage / 100)
    silent_audio = np.zeros(int(SAMPLE_RATE * 1.0))  # 1 second of silence
    silence_mfcc = preprocess_audio(
        silent_audio, SAMPLE_RATE, WINDOW_SIZE_MS, WINDOW_STRIDE, FEATURE_BIN_COUNT
    )
    for _ in range(num_silence_samples):
        X.append(silence_mfcc)
        y.append(silence_label)  # silence label


    noise_files = glob.glob(os.path.join(background_noise_dir, '*.wav'))
    for file in noise_files:
        noise_audio, sr = librosa.load(file, sr=16000)  # 16kHz sampling rate

        num_segments = len(noise_audio) // sr  # divided by the number of samples per second

        for i in range(num_segments):
            # extract 1 second segment
            start = i * sr
            end = start + sr
            segment = noise_audio[start:end]

            # if shorter than 1 second padding
            if len(segment) < sr:
                segment = np.pad(segment, (0, sr - len(segment)), mode='constant')

            noise_mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=FEATURE_BIN_COUNT)
            if noise_mfcc.shape[1] < FEATURE_SLICE_COUNT:
                # padding if fewer than needed frames
                pad_width = FEATURE_SLICE_COUNT - noise_mfcc.shape[1]
                noise_mfcc = np.pad(noise_mfcc, ((0, 0), (0, pad_width)), mode='constant')
            elif noise_mfcc.shape[1] > FEATURE_SLICE_COUNT:
                # truncate to needed frames
                noise_mfcc = noise_mfcc[:, :FEATURE_SLICE_COUNT]

            if isinstance(noise_mfcc, np.ndarray):
                noise_mfcc = tf.convert_to_tensor(noise_mfcc)

            X.append(tf.transpose(noise_mfcc))
            y.append(unknown_label)


    # random selection of samples from non target words
    unknown_path = os.path.join(audio_files_directory, 'train_unknown')
    unknown_candidates = os.listdir(unknown_path)
    num_unknown_samples = int(len(X) * unknown_percentage / 100)
    selected_unknown_files = []

    while len(selected_unknown_files) < num_unknown_samples:
        unknown_word = random.choice(unknown_candidates)
        unknown_dir = os.path.join(unknown_path, unknown_word)
        if not os.path.isdir(unknown_dir):
            continue
        for filename in os.listdir(unknown_dir):
            if filename.endswith('.wav'):
                selected_unknown_files.append(os.path.join(unknown_dir, filename))
            if len(selected_unknown_files) >= num_unknown_samples:
                break

    for unknown_file in selected_unknown_files:
        mfcc = preprocess_audio(unknown_file, SAMPLE_RATE, WINDOW_SIZE_MS, WINDOW_STRIDE, FEATURE_BIN_COUNT)
        X.append(mfcc)
        y.append(unknown_label)

    X = np.array(X)
    y = np.array(y)

    num_target_samples = len([label for label in y if label in word_to_label.values()])
    num_silence_samples = len([label for label in y if label == silence_label])
    num_unknown_samples = len([label for label in y if label == unknown_label])

    print(f"Total number of samples in X before splitting: {len(X)}")
    print(f"Number of target word samples: {num_target_samples}")
    print(f"Number of silence samples: {num_silence_samples}")
    print(f"Number of unknown samples: {num_unknown_samples}")
    print(f"Skipped samples: {skipped_samples}")

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=(VALIDATION_PERCENTAGE + TESTING_PERCENTAGE),
                                                        random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=TESTING_PERCENTAGE / (
            VALIDATION_PERCENTAGE + TESTING_PERCENTAGE), random_state=42)

    return X_train, y_train, X_val, y_val, X_test, y_test


print("Load and preprocess the data")
X_train, y_train, X_val, y_val, X_test, y_test = load_data(dataset_directory, WANTED_WORDS, SILENT_PERCENTAGE, UNKNOWN_PERCENTAGE)

NUM_CLASSES = max(y_train) + 1

X_train = X_train.reshape((-1, FEATURE_SLICE_COUNT, FEATURE_BIN_COUNT, 1))
X_val = X_val.reshape((-1, FEATURE_SLICE_COUNT, FEATURE_BIN_COUNT, 1))
X_test = X_test.reshape((-1, FEATURE_SLICE_COUNT, FEATURE_BIN_COUNT, 1))

y_train = to_categorical(y_train, num_classes=NUM_CLASSES)
y_val = to_categorical(y_val, num_classes=NUM_CLASSES)
y_test = to_categorical(y_test, num_classes=NUM_CLASSES)

# Be sure the input shape matches Arduino expectations!
input_shape = (FEATURE_SLICE_COUNT, FEATURE_BIN_COUNT, 1)

model = create_model(input_shape, NUM_CLASSES)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE[0]),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

checkpoint_dir = os.path.join(KWS_path, 'models')
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(filepath='models/checkpoint.keras', save_freq=SAVE_STEP_INTERVAL * BATCH_SIZE),
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6),
]

print("Training...")
print(f"Size of X_train: {len(X_train)}")
print(f"Size of X_val: {len(X_val)}")
print(f"Size of X_test: {len(X_test)}")

print(f"BATCH_SIZE: {BATCH_SIZE}")
print(f"TRAINING_STEPS: {TRAINING_STEPS}")

steps_per_epoch = max(len(X_train) // BATCH_SIZE, 1)
initial_epochs = TRAINING_STEPS[0] // steps_per_epoch
final_epochs = TRAINING_STEPS[1] // steps_per_epoch

# 1st training phase
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=initial_epochs, batch_size=BATCH_SIZE, callbacks=callbacks)

model.optimizer.learning_rate.assign(LEARNING_RATE[1])

# 2nd training phase
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=final_epochs, batch_size=BATCH_SIZE, callbacks=callbacks)


model_save_dir = os.path.join(KWS_path, 'models')
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)

model.save('models/KWS_custom_saved_model.keras')

test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc}')


# flatten inputs before TF Lite conversion
def flatten_input_for_tflite(X):
    return X.reshape((-1, FEATURE_ELEMENT_COUNT))

X_train_flattened = flatten_input_for_tflite(X_train)
X_test_flattened = flatten_input_for_tflite(X_test)
X_val_flattened = flatten_input_for_tflite(X_val)


def add_flatten_to_4d_input(original_model):
    # input layer expecting 1D flattened input
    inputs = tf.keras.Input(shape=(FEATURE_SLICE_COUNT * FEATURE_BIN_COUNT,))

    # reshape to 4D
    reshaped_inputs = tf.keras.layers.Reshape((FEATURE_SLICE_COUNT, FEATURE_BIN_COUNT, 1))(inputs)
    # passing the reshaped input to the original model
    outputs = original_model(reshaped_inputs)
    # new model
    return tf.keras.Model(inputs=inputs, outputs=outputs)


# original model wrap with FlattenTo4DInputModel
original_model = tf.keras.models.load_model('models/KWS_custom_saved_model.keras')
model_for_tflite = add_flatten_to_4d_input(original_model)

converter = tf.lite.TFLiteConverter.from_keras_model(model_for_tflite)
converter.optimizations = [tf.lite.Optimize.DEFAULT]


# representative dataset for quantization
def representative_dataset_gen():
    for i in range(len(X_train_flattened)):
        data = X_train_flattened[i].astype(np.float32)
        yield [np.expand_dims(data, axis=0)]

converter.representative_dataset = representative_dataset_gen
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8


tflite_model = converter.convert()
tflite_model_path = os.path.join(model_save_dir, 'tf_lite', 'model_quantized.tflite')
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)

print(f"Quantized TensorFlow Lite model saved at {tflite_model_path}")


def run_tflite_inference_testSet(tflite_model_path, test_data, test_labels, model_type="Float"):
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    if model_type == "Quantized":
        input_scale, input_zero_point = input_details["quantization"]
        test_data = test_data / input_scale + input_zero_point
        test_data = test_data.astype(input_details["dtype"])

    correct_predictions = 0
    for i in range(len(test_data)):
        interpreter.set_tensor(input_details["index"], np.expand_dims(test_data[i], axis=0))
        interpreter.invoke()
        output = interpreter.get_tensor(output_details["index"])[0]
        top_prediction = np.argmax(output)
        correct_predictions += (top_prediction == np.argmax(test_labels[i]))

    accuracy = (correct_predictions * 100) / len(test_data)
    print(f'{model_type} model accuracy is {accuracy:.2f}% (Number of test samples={len(test_data)})')
    return accuracy


run_tflite_inference_testSet(tflite_model_path, X_test_flattened, y_test, model_type="Quantized")


# converting the model into C source code (TF lite micro)
tf_micro_path = os.path.join(model_save_dir, 'tf_micro', 'model_quantized.cc')
subprocess.run(['xxd', '-i', tflite_model_path, tf_micro_path], check=True)
REPLACE_TEXT = tflite_model_path.replace('/', '_').replace('.', '_')

with open(tf_micro_path, 'r') as file:
    cc_content = file.read()

cc_content = cc_content.replace(REPLACE_TEXT, 'g_model')

with open(tf_micro_path, 'w') as file:
    file.write(cc_content)
print(f"Converted model saved as C source file: {tf_micro_path}")
