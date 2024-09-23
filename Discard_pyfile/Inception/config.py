import torch
# some training parameters
EPOCHS = 50
BATCH_SIZE = 32
NUM_CLASSES = 4
image_height = 299
image_width = 299
channels = 1
save_model = "./Inception_model.pth"

train_audio = '../data/train_mp3s'
train_image = '../data/train_images'
train_label = '../data/train_label.txt'
test_audio = '../data/test_mp3s'
test_image = '../data/test_images'
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")