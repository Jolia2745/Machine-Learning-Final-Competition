print(1)
import data_loader as dl
# import pandas as pd
# import data_processor as dp
# import os
# from config import  EPOCHS , NUM_CLASSES , save_model,train_audio, train_image , train_label , test_audio , test_image , device 
# from inception_model import inception_v3
# import tensorflow as tf
# import numpy as np
import sys

#import tensorflow.keras
import pandas as pd
import sklearn as sk
import scipy as sp
import tensorflow as tf
import platform

print(f"Python Platform: {platform.platform()}")
print(f"Tensor Flow Version: {tf.__version__}")
#print(f"Keras Version: {tensorflow.keras.__version__}")
print()
print(f"Python {sys.version}")
print(f"Pandas {pd.__version__}")
print(f"Scikit-Learn {sk.__version__}")
print(f"SciPy {sp.__version__}")
gpu = len(tf.config.list_physical_devices('GPU'))>0
print("GPU is", "available" if gpu else "NOT AVAILABLE")
print(2)
'''
import data_loader as dl #
import pandas as pd
import data_processor as dp
import os
from config import  EPOCHS , NUM_CLASSES , save_model,train_audio, train_image , train_label , test_audio , test_image , device  #
from inception_model import inception_v3 #
import tensorflow as tf #  
import numpy as np 

def train_model(model, train_loader, val_loader, num_epochs= EPOCHS):

    # Define the optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)

    # Define the loss function
    criterion = tf.keras.losses.SparseCategoricalCrossentropy()

    for epoch in range(num_epochs):

        for step, (features, labels) in enumerate(train_loader):
            # features = tf.convert_to_tensor(features, dtype=tf.float32)
            # labels = tf.convert_to_tensor(labels, dtype=tf.int64)

            with tf.GradientTape() as tape:
                predictions = model(features, training=True)
                loss = criterion(labels, predictions)

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            running_loss = loss.numpy()

            print('[Epoch: %d, Batch: %5d] Loss: %.3f' %
                  (epoch + 1, step + 1, running_loss))

        val_loss = eval_model(model, val_loader, criterion)
        print(f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss}')


        # plt.figure(figsize=(10, 5))
        # plt.plot(train_accuracies, label='Train Accuracy')
        # plt.plot(val_accuracies, label='Validation Accuracy')
        # plt.title('Accuracy over epochs')
        # plt.xlabel('Epochs')
        # plt.ylabel('Accuracy')
        # plt.legend()
        # plt.grid(True)
        # plt.show()

def eval_model(model, val_loader, criterion):
    total = 0
    correct = 0
    total_loss = 0.0
    for features, labels in val_loader:
        # features = tf.convert_to_tensor(features, dtype=tf.float32)
        # labels = tf.convert_to_tensor(labels, dtype=tf.int64)

        predictions = model(features, training=False)
        loss = criterion(labels, predictions)
        total_loss += loss.numpy()

        predicted = tf.argmax(predictions, axis=1)
        correct += tf.reduce_sum(tf.cast(tf.equal(predicted, labels), tf.int32)).numpy()
        total += labels.shape[0]

    accuracy = 100 * correct / total
    print(f'Validation Accuracy: {accuracy}%')
    return total_loss / len(val_loader)


def save_model_parameter(model, path):
    model.save_weights(path)  # save_weights is tensor flow parameter
    print(f"Model saved to {path}")

def load_model(model, path):
    model.load_weights(path)
    model.compile()  # Recompile the model after loading weights
    print(f"Model loaded from {path}")

def print_model_parameters(model):
    for layer in model.layers:
        for weight in layer.weights:
            print(f"{weight.name}: {weight.shape}")


def prediction(model, test_loader):
    device = "gpu" if tf.config.experimental.list_physical_devices('GPU') else "cpu"
    all_predictions = []

    for batch in test_loader:
        feature_batch = batch  # Assuming batch is already in the correct format
        
        # Add a channel dimension if it's missing
        if len(feature_batch.shape) == 3:
            feature_batch = tf.expand_dims(feature_batch, axis=-1)

        predictions_batch = model(feature_batch, training=False)
        predicted_classes = tf.argmax(predictions_batch, axis=1)
        all_predictions.extend(predicted_classes.numpy())

    submission = pd.DataFrame({
        'id': np.arange(len(all_predictions)),  # create a range from 0 to len(all_predictions)-1
        'category': all_predictions  # 使用所有预测类别索引的列表
    })

    submission.to_csv('./submission.csv', index=False)


def main():
    
    

    print("Starting the main function...")
    criterion = tf.keras.losses.SparseCategoricalCrossentropy()
    os.makedirs(train_image, exist_ok=True)  
    os.makedirs(test_image, exist_ok=True)
    print("Directories created. Now begin to process dataset...")

    dp.save_image(train_audio, train_image)
    dp.save_image(test_audio, test_image)

    print("Dataset processed. Now begin to load dataset...")
    train_loader, val_loader, test_loader, _, _, _ = dl.generate_datasets()
    model = inception_v3.InceptionV3(NUM_CLASSES)
    print("Model loaded:", type(model))
    
    print("Now begin to train model...")
    train_model(model, train_loader, val_loader)
    save_model_parameter(model, save_model)
    
    print("Now begin to evaluate model...")
    eval_model(model, val_loader, criterion)
    
    print("Now begin to predict labels...")
    prediction(model, test_loader)

print(0)
main()
'''