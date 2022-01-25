import os
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt


class Modeling:

    def __init__(self) -> None:
        pass

    def model_definition(self):
        # Disabling the GPUs to enable computing on my CPU
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

        # Model definition: Sequential CNN
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
            tf.keras.layers.MaxPooling2D(2, 2),

            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),

            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        # Compiling the model
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

        training_dir = "trial1/augmented data1/training"
        train_datagen = ImageDataGenerator(rescale=1.0 / 255, rotation_range=40,
                                           width_shift_range=0.2,
                                           height_shift_range=0.2,
                                           shear_range=0.2,
                                           zoom_range=0.2,
                                           horizontal_flip = True,
                                           fill_mode='nearest')

        train_generator = train_datagen.flow_from_directory(training_dir, batch_size=10, class_mode='binary',
                                                            target_size=(150, 150))
        validation_dir = "trial1/augmented data1/testing"
        validation_datagen = ImageDataGenerator(rescale=1.0 / 255)

        validation_generator = validation_datagen.flow_from_directory(validation_dir, batch_size=10,
                                                                      class_mode='binary', target_size=(150, 150))

        # Fitting the model with the training dataset while also defining the validation dataset
        history = model.fit_generator(train_generator, epochs=20, verbose=1, validation_data=validation_generator)

        acc = history.history['acc']
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs = range(len(acc))  # Get number of epochs

        plt.plot(epochs, acc, 'r', "Training Accuracy")
        plt.plot(epochs, val_acc, 'b', "Validation Accuracy")
        plt.title('Training and validation accuracy')
        plt.figure()

        plt.plot(epochs, loss, 'r', "Training Loss")
        plt.plot(epochs, val_loss, 'b', "Validation Loss")

        plt.title('Training and validation loss')
        plt.show()

        # model = tf.keras.models.Sequential([
        #     tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        #     tf.keras.layers.MaxPooling2D(2, 2),
        #
        #     tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        #     tf.keras.layers.MaxPooling2D(2, 2),
        #
        #     tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        #     tf.keras.layers.MaxPooling2D(2, 2),
        #
        #     tf.keras.layers.Flatten(),
        #     tf.keras.layers.Dense(512, activation='relu'),
        #     tf.keras.layers.Dense(1, activation='sigmoid')
        # ])
        # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
        #
        # TRAINING_DIR = "trial1/augmented data1/"
        # train_datagen = ImageDataGenerator(rescale=1.0 / 255,
        #                                    rotation_range=40,
        #                                    width_shift_range=0.2,
        #                                    height_shift_range=0.2,
        #                                    shear_range=0.2,
        #                                    zoom_range=0.2,
        #                                    horizontal_flip=True,
        #                                    fill_mode='nearest')
        #
        # train_generator = train_datagen.flow_from_directory(TRAINING_DIR,
        #                                                     batch_size=10,
        #                                                     class_mode='binary',
        #                                                     target_size=(150, 150))
        #
        # VALIDATION_DIR = "trial1/augmented data1/testing"
        # validation_datagen = ImageDataGenerator(rescale=1.0 / 255)
        #
        # validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR,
        #                                                               batch_size=10,
        #                                                               class_mode='binary',
        #                                                               target_size=(150, 150))
        #
        # history = model.fit_generator(train_generator,
        #                               epochs=2,
        #                               verbose=1,
        #                               validation_data=validation_generator)
