import os
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing import image
from tensorflow.keras import layers
from tensorflow import keras


class Modeling:

    def __init__(self) -> None:
        pass

    def model_definition(self):
        # Disabling the GPUs to enable computing on my CPU
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

        # Model definition: Sequential CNN
        img_h = 150
        img_w = 150
        model = tf.keras.models.Sequential([
            tf.keras.layers.Rescaling(1./255, input_shape=(img_h, img_w, 3)),
            tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
            tf.keras.layers.MaxPooling2D(2, 2),

            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),

            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        # print("Model summary:", model.summary())

        # Compiling the model
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

        training_dir = "trial1/augmented data1/training"
        validation_dir = "trial1/augmented data1/testing"

        train_ds = tf.keras.utils.image_dataset_from_directory(training_dir, seed=123,
                                                               image_size=(img_h, img_w), batch_size=120)

        val_ds = tf.keras.utils.image_dataset_from_directory(validation_dir, seed=123,
                                                             image_size=(img_h, img_w), batch_size=20)

        class_names = val_ds.class_names
        print(class_names)

        plt.figure(figsize=(10, 10))
        for images, labels in train_ds.take(1):
            for i in range(9):
                ax = plt.subplot(3, 3, i + 1)
                plt.imshow(images[i].numpy().astype("uint8"))
                plt.title(class_names[labels[i]])
                plt.axis("off")
            plt.show()

        history = model.fit(train_ds, epochs=15, validation_data=val_ds)

        pred_dir = "trial1/augmented data1/prediction/aug_Y10_0_1877.jpg"

        pred_img = tf.keras.utils.load_img(pred_dir, target_size=(img_h, img_w))
        img_array = tf.keras.utils.img_to_array(pred_img)

        img_array = tf.expand_dims(img_array, 0)  # Create a batch

        predictions = model.predict(img_array)
        print("Pred:", predictions)
        print("Pred[0]", predictions[0])
        score = tf.nn.softmax(predictions[0])

        print(
            "This image most likely belongs to {} with a {:.2f} percent confidence."
            .format(class_names[np.argmax(score)], 100 * np.max(score))
        )
