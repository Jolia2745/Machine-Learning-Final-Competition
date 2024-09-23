import os
import librosa
import pickle
import numpy as np


# Function to convert audio file to Mel Spectrogram and save as an image
def audio_to_image(file_path, save_path):
    '''
    Process an audio file to extract various features and save them to a file.
    - Mel spectrogram
    - Harmonic component of the audio
    - Mel Frequency Cepstral Coefficients (MFCCs)
    - Chroma features
    '''
    y, sr = librosa.load(file_path, sr=None)

    # Separate harmonic (human voice) and percussive components
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    # Mel spectrogram of the harmonic component
    mel_spec = librosa.feature.melspectrogram(y=y_harmonic, sr=sr, n_mels=128, n_fft=2048, hop_length=512, fmax=8000) # n_mels: # of rows
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max) # converts a power spectrogram (amplitude squared) to decibel (dB) units

    # MFCCs from the harmonic component
    mfccs = librosa.feature.mfcc(y=y_harmonic, sr=sr, n_mfcc=20)

    # Chroma features from the harmonic component (separate male and female)
    chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr) # 12 rows, represents 12 pitch classes

    # Stack all features into a single array
    features = np.vstack([log_mel_spec, mfccs, chroma]) # 2-D numpy arrays. rows: (128 + 20 + 12 = 160 rows) 

    with open(save_path, 'wb') as f:
        pickle.dump(features, f)

# Process each audio file
def save_image(audio_dir, image_dir):
    for filename in os.listdir(audio_dir):
        if filename.endswith('.mp3'):
            file_path = os.path.join(audio_dir, filename)
            save_path = os.path.join(image_dir, filename.replace('.mp3', '.pkl'))
            if not os.path.exists(save_path):
                audio_to_image(file_path, save_path)
                print(f"Processed and saved: {save_path}")
            #else:
                print(f"Already been saved: {save_path}")


train_audio = './data/train_mp3s'
train_image = './data_v2/train_images'
test_audio = './data/test_mp3s'
test_image = './data_v2/test_images'

os.makedirs("./data_v2", exist_ok = True)
os.makedirs(train_image, exist_ok=True)  
os.makedirs(test_image, exist_ok=True) 
print("Now begin to process dataset...")
save_image(train_audio, train_image)
save_image(test_audio, test_image)
print("Finish!")