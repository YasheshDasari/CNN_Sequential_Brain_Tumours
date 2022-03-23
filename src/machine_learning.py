import os
import tensorflow as tf
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt
import numpy as np


class Modeling:

    def __init__(self) -> None:
        pass

    def model_definition(self):
        # Disabling the GPUs to enable computing using local CPU
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

        # Model definition: Sequential CNN
        img_h = 200
        img_w = 200

        training_dir = "trial1/augmented data1/training"
        validation_dir = "trial1/augmented data1/testing"

        # Rescaling the images
        train_datagen = ImageDataGenerator(rescale=1 / 255)
        validation_datagen = ImageDataGenerator(rescale=1 / 255)

        # Flow training images in batches of 120 using train_datagen generator
        train_generator = train_datagen.flow_from_directory(
            training_dir,  # This is the source directory for training images
            classes=['no', 'yes'],
            target_size=(200, 200),  # All images will be resized to 200x200
            batch_size=120,
            # For binary classification problems:
            class_mode='binary')

        # Flow validation images in batches of 19 using valid_datagen generator
        validation_generator = validation_datagen.flow_from_directory(
            validation_dir,
            classes=['no', 'yes'],
            target_size=(200, 200),
            batch_size=19,
            class_mode='binary',
            shuffle=False)

        # Defining model parameters
        model = tf.keras.models.Sequential([
            tf.keras.layers.Rescaling(1. / 255, input_shape=(img_h, img_w, 3)),
            tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(200, 200, 3)),
            tf.keras.layers.MaxPooling2D(2, 2),

            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),

            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        # Compiling the model
        model.compile(optimizer=tf.optimizers.Adam(),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        history = model.fit(train_generator,
                            steps_per_epoch=13,
                            epochs=20,
                            verbose=1,
                            validation_data=validation_generator,
                            validation_steps=8)

        # Saving the trained CNN model
        model.save('CNN_MRI.h5')

    def predict(self):
        # Loading the pre-trained CNN model
        load_model = tf.keras.models.load_model('CNN_MRI.h5')
        path = "trial1/augmented data1/prediction/test/aug_no 92_0_5976.jpg"
        img = image.load_img(path, target_size=(200, 200))
        x = image.img_to_array(img)
        plt.imshow(x / 255.)
        x = np.expand_dims(x, axis=0)
        images = np.vstack([x])
        classes = load_model.predict(images, batch_size=10)
        if classes[0] < 0.5:
            print(path + " is healthy")
        else:
            print(path + " has a tumor")
