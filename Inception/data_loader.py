import tensorflow as tf
from config import image_height, image_width, channels, BATCH_SIZE , EPOCHS , NUM_CLASSES , \
save_model,train_audio, train_image , train_label , test_audio , test_image , device 
import pickle
import os
import numpy as np


def load_image(path):
    # Since path is a tensor, convert it to string within the TensorFlow graph
    path = path.decode('utf-8')
    print(1, type(path))
    with open(path, 'rb') as f:
        feature = pickle.load(f)
        print("load_image", np.array(feature, dtype=np.float32) )
    return np.array(feature, dtype=np.float32) 

# Load and preprocess images

def load_and_preprocess_image(img_path):
    # Convert the loaded image to a float32 tensor
    # tf.py_function, build a bridge between normal python function and tensor (load pickle)
    # change the tensor spec into float, and transmit the float to load_image
    feature = tf.py_function(load_image, [img_path], Tout=tf.float32)
    print(2, type(feature)) # <class 'tensorflow.python.framework.ops.SymbolicTensor'>
    feature.set_shape([None, None, 1]) 
    return feature


# Process dataset
def preprocess_dataset(path):
    # Adding a channel dimension, resizing
    image = load_and_preprocess_image(path)
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



'''  below: OOM: becuase of shuffle / load the whole dataset (11000) take out too man memory

def preprocess_image(image_paths): # get a 3 dimension tensor (channel, height, width)

    images = []
    for img_path in image_paths:
        with open(img_path, 'rb') as f:
            feature = pickle.load(f)
        feature_tensor = tf.constant(feature, dtype=tf.float32)
        img_tensor = tf.expand_dims(feature_tensor, axis=0)  # Adding a channel dimension
        img_tensor = tf.image.resize(img_tensor, (image_height, image_width))
        img_tensor = tf.cast(img_tensor, tf.float32) / 255.0
        images.append(img_tensor)
    return images

def load_labels(label = train_label):
    with open(label, "r") as f:
        return [int(label.strip()) for label in f]

def load_dataset_paths(image = train_image, file_extension='.pkl'):
    files = [f for f in os.listdir(image) if f.endswith(file_extension)]
    files_sorted = sorted(files, key=lambda x: int(x.split('.')[0]))           
    mel = [os.path.join(image, f) for f in files_sorted]
    
    return mel

def get_images_and_labels(): # for train set
    return load_dataset_paths(), load_labels()


def get_dataset(train):
    if train:
        all_image_path, all_image_label = get_images_and_labels()
        # load the dataset and preprocess images
        preloaded_images = preprocess_image(all_image_path)
        image_dataset = tf.data.Dataset.from_tensor_slices(preloaded_images)
        label_dataset = tf.data.Dataset.from_tensor_slices(all_image_label)
        dataset = tf.data.Dataset.zip((image_dataset, label_dataset))
        image_count = len(all_image_path)
    else:
        all_image_path, = get_images_and_labels()
        # load the dataset and preprocess images
        preloaded_images = preprocess_image(all_image_path)
        image_dataset = tf.data.Dataset.from_tensor_slices(preloaded_images)
        image_count = len(all_image_path)

    return dataset, image_count


def generate_datasets(split_ratio=0.8):
    # Assume that all data is in a single directory
    dataset, total_count = get_dataset(True) # represent the train set
    # Shuffling the dataset   use too much memory!!! OOM
    # dataset = dataset.shuffle(buffer_size=total_count, reshuffle_each_iteration=False)
    
    # Splitting the dataset into train and validation sets
    train_count = int(total_count * split_ratio)
    valid_count = total_count - train_count

    train_dataset = dataset.take(train_count)
    valid_dataset = dataset.skip(train_count)

    # Optionally, if you have a separate test set, load it similarly to get_dataset
    test_dataset, test_count = get_dataset(False)

    # Batching the datasets
    train_dataset = train_dataset.batch(batch_size= BATCH_SIZE)
    valid_dataset = valid_dataset.batch(batch_size= BATCH_SIZE)
    test_dataset = test_dataset.batch(batch_size= BATCH_SIZE)

    return train_dataset, valid_dataset, test_dataset, train_count, valid_count, test_count

'''