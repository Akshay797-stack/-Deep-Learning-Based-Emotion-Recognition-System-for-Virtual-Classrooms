import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define dataset path
dataset_path = "dataset/"

# Define data generator for preprocessing
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Load images and labels
train_data = datagen.flow_from_directory(dataset_path, target_size=(48, 48), batch_size=32, class_mode="categorical", subset="training")
val_data = datagen.flow_from_directory(dataset_path, target_size=(48, 48), batch_size=32, class_mode="categorical", subset="validation")

print("Data Loaded Successfully!")
