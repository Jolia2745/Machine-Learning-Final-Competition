import os
import librosa
import pickle
import numpy as np

'''
n_fft: Number of FFT (Fast Fourier Transform) points. Defines the size of the FFT window, 
affecting frequency resolution. A larger n_fft provides finer frequency resolution 
but coarser time resolution, and vice versa.

hop_length: The number of samples between successive frames, e.g., the step size for each window. 
Smaller hop_length increases overlap between windows and provides smoother temporal resolution 
at a higher computational cost.

n_mfcc: Number of Mel Frequency Cepstral Coefficients to extract from an audio signal. 
MFCCs capture the timbral aspects of sound and are commonly used in speech recognition 
and music information retrieval tasks.
'''


def augment_audio(y, sr):
    ''' Apply random data augmentation to the audio. '''
    # Random choice for the augmentation type
    choice = np.random.choice(['none', 'pitch_shift', 'add_noise'], p=[0.5, 0.3, 0.2])

    '''
    pitch_shift(y: numpy.ndarray, *, sr: float, n_steps: float, bins_per_octave: int = 12, res_type: str = 'soxr_hq', scale: bool = Fals
    e, **kwargs: Any) -> numpy.ndarray
    '''
    if choice == 'pitch_shift':
        # Pitch shifting
        y = librosa.effects.pitch_shift(y , sr = sr, n_steps = np.random.randint(-2, 3))

    # elif choice == 'time_stretch':   this may change the length of the audio
    #     # Time stretching
    #     rate = np.random.uniform(0.8, 1.2)
    #     y = librosa.effects.time_stretch(y,  rate = rate)
    elif choice == 'add_noise':
        # Adding noise
        noise = np.random.randn(len(y)) * 0.005
        y += noise

    return y

# Function to convert audio file to Mel Spectrogram and save as an image
def audio_to_image(file_path, save_path):
    '''
    Process an audio file to extract various features and save them to a file.
    - Mel spectrogram
    - Harmonic component of the audio
    - Mel Frequency Cepstral Coefficients (MFCCs)
    - Chroma features
    '''
    signal, sr = librosa.load(file_path, sr=None)
    # print(signal.shape) # (132300,)

    # Apply data augmentation
    # signal = augment_audio(signal, sr)

    n_fft = 512
    hop_length = 256
    fmax = sr / 2  # 22050 Hz

    # Extract MFCCs
    mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=40)
    mfccs_processed = np.mean(mfccs.T, axis=0)
    
    # Extract Mel Spectrogram
    mel_spec = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=128)
    #print(111, mel_spec.shape) # (128, 259)
    mel_spec_processed = np.mean(librosa.power_to_db(mel_spec, ref=np.max), axis=1)
    #print(111, mel_spec_processed.shape) #(128,)
    
    # Extract Chroma STFT
    chroma = librosa.feature.chroma_stft(y=signal, sr=sr, n_chroma=12)
    #print(222, chroma.shape)  #(12, 259) 
    chroma_processed = np.mean(chroma.T, axis=0)
    # print(222, chroma_processed.shape) (12, )
    
    # Extract Spectral Centroid
    spec_centroid = librosa.feature.spectral_centroid(y=signal, sr=sr)
    #print(333, spec_centroid.shape) # (1, 259)
    spec_centroid_processed = np.mean(spec_centroid.T, axis=0)
    
    # Extract Spectral Contrast
    spec_contrast = librosa.feature.spectral_contrast(y=signal, sr=sr)
    #print(444, spec_contrast.shape)  # (7, 259)
    spec_contrast_processed = np.mean(spec_contrast.T, axis=0)
    
    # Extract Tonnetz
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(signal), sr=sr)
    tonnetz_processed = np.mean(tonnetz.T, axis=0)

    # print(mfccs.shape, mfccs_processed.shape, chroma.shape,  chroma_processed.shape, spec_contrast.shape,spec_centroid_processed.shape,  tonnetz.shape,  tonnetz_processed.shape)
    #(40, 259) (40,) (12, 259) (12,) (7, 259) (1,) (6, 259) (6,)
    # output: # (XX, 259)
    
    # Combine all features
    features = np.vstack([mfccs, mel_spec, chroma, spec_centroid, spec_contrast, tonnetz])   # 2-D numpy arrays
    # print(features.shape) # (194, 259)

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
                
            else:
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
