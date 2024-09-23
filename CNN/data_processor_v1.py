import os
import librosa
import pickle

# Function to convert audio file to Mel Spectrogram and save as an image
def audio_to_image(file_path, save_path):
    '''
    Change mp3 into Mel Spectrogram
    '''
    y, sr = librosa.load(file_path, sr=None)  # Load the audio file
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, fmax=8000)  # Extract Mel spectrogram
    log_mel_spec = librosa.power_to_db(mel_spec)  # 2-D numpy 
    with open(save_path, 'wb') as f:
        pickle.dump(log_mel_spec, f)
    
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




