import tensorflow as tf
from config import image_height, image_width, channels, BATCH_SIZE , EPOCHS , NUM_CLASSES , \
save_model,train_audio, train_image , train_label , test_audio , test_image , device 
import pickle
import os
import numpy as np


def load_image(path):
    # Since path is a tensor, convert it to string within the TensorFlow graph
    path = path.decode('utf-8')
    with open(path, 'rb') as f:
        feature = pickle.load(f)
        print("load_image", np.array(feature, dtype=np.float32) )
    return np.array(feature, dtype=np.float32) 

# Load and preprocess images
'''
def load_and_preprocess_image(img_path):
    # Convert the loaded image to a float32 tensor
    # tf.py_function, build a bridge between normal python function and tensor (load pickle)
    feature = tf.py_function(load_image, [img_path], Tout=tf.float32)
    return feature
'''

# Process dataset
def preprocess_dataset(path):
    # Adding a channel dimension, resizing
    image = load_image(path)
    print("Original shape:", image.shape) 
    image = tf.expand_dims(image, -1)  # Adding channel at the last axis
    print("Shape after expand_dims:", image.shape)
    image = tf.image.resize(image, (image_height, image_width))  # image.resize: [batchsize, h, w, channels]
    image = tf.cast(image, tf.float32) / 255.0
    return image

# Load dataset paths and labels
def get_images_and_labels(image_dir, label_file):
    with open(label_file, "r") as f:
        labels = [int(label.strip()) for label in f]

    files = [f for f in os.listdir(image_dir) if f.endswith('.pkl')]
    files_sorted = sorted(files, key=lambda x: int(x.split('.')[0]))
    paths = [os.path.join(image_dir, f) for f in files_sorted]
    
    return paths, labels

def generator(image_paths, labels):
    if labels is None:
        for path in image_paths:
            yield path
    for path, label in zip(image_paths, labels):
        yield path, label

# Create a TensorFlow Dataset
def create_dataset(image_paths, labels, is_training=True):
    # path_ds = tf.data.Dataset.from_tensor_slices(image_paths) # turn each element in paths into symbolic tensor
    if is_training:
        dataset = tf.data.Dataset.from_generator(generator, args=(image_paths, labels), output_signature=(tf.TensorSpec(shape=(), dtype=tf.string), tf.TensorSpec(shape=(), dtype=tf.int32)))
        dataset = dataset.batch(BATCH_SIZE)
        dataset = dataset.map(lambda x, y: (preprocess_dataset(x), y), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.shuffle(buffer_size=len(image_paths))
    else:
        dataset = tf.data.Dataset.from_generator(generator, args=(image_paths, None), output_signature=(tf.TensorSpec(shape=(), dtype=tf.string),))
        dataset = dataset.batch(BATCH_SIZE)
        dataset = dataset.map(preprocess_dataset, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    return dataset


# Generate training, validation, and testing datasets
def generate_datasets(image_dir = train_image, label_file = train_label, test_image = test_image, split_ratio=0.8):
    all_image_paths, all_labels = get_images_and_labels(image_dir, label_file)

    dataset = create_dataset(all_image_paths, all_labels)

    # Splitting the dataset into train and validation sets
    train_count = int(len(all_image_paths) * split_ratio)
    valid_count = len(all_image_paths) - train_count

    train_dataset = dataset.take(train_count)
    valid_dataset = dataset.skip(train_count)

    # Optionally, if you have a separate test set
    test_dataset = create_dataset(test_image, None, is_training=False)  # Assuming same process for demonstration

    return train_dataset, valid_dataset, test_dataset

train_loader, val_loader, test_loader  = generate_datasets()

