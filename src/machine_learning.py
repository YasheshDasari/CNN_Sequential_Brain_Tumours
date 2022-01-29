import os
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing import image


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

        print("Model summary:", model.summary())

        # Compiling the model
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

        training_dir = "trial1/augmented data1/training"
        train_datagen = ImageDataGenerator(rescale=1/255)
                                           # rotation_range=40,
                                           # width_shift_range=0.2,
                                           # height_shift_range=0.2,
                                           # shear_range=0.2,
                                           # zoom_range=0.2,
                                           # horizontal_flip=True,
                                           # fill_mode='nearest')

        train_generator = train_datagen.flow_from_directory(training_dir, batch_size=120, classes=['no', 'yes'],
                                                            class_mode='binary',
                                                            target_size=(150, 150))

        validation_dir = "trial1/augmented data1/testing"
        # os.path.join("trial1", "augmented data1", "testing")

        validation_datagen = ImageDataGenerator(rescale=1/255)

        validation_generator = validation_datagen.flow_from_directory(validation_dir, batch_size=20,
                                                                      classes=['no', 'yes'],
                                                                      class_mode='binary', target_size=(150, 150))

        # Fitting the model with the training dataset while also defining the validation dataset
        history = model.fit_generator(train_generator, steps_per_epoch=8, epochs=5, verbose=1,
                                      validation_data=validation_generator, validation_steps=8)

        # acc = history.history['acc']
        # val_acc = history.history['val_acc']
        # loss = history.history['loss']
        # val_loss = history.history['val_loss']

        prediction_dir = "trial1/augmented data1/prediction"
        # for i in os.listdir(prediction_dir):
        #     img = image.load_img(prediction_dir, target_size=(150, 150))
        #     x = image.img_to_array(img)
        #     plt.imshow(x/255.)
        #     x = np.expand_dims(x, axis=0)
        #     images = np.vstack([x])
        #     classes = model.predict(images, batch_size=10)
        #     print(classes[0])
        #     if classes[0]<0.5:
        #         print(i + "is non-tumorous")
        #     else:
        #         print(i + "is tumorous")

        test_datagen = ImageDataGenerator(rescale=1/255)
        test_generator = test_datagen.flow_from_directory(
            prediction_dir,
            target_size=(150, 150),
            batch_size=1,
            shuffle=False,
            class_mode=None
        )

        # probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
        # new_predictions = probability_model.predict(test_generator)
        # print("First element of prediction (before argmax):", new_predictions[0])
        # np.argmax(new_predictions)
        # print("New predictions: ", new_predictions)

        # STEP_SIZE_TEST = test_generator.n // test_generator.batch_size
        test_generator.reset()
        pred = model.predict(test_generator)
        # pred = model.predict_generator(test_generator,
        #                                steps=1,
        #                                verbose=1)

        print("pred:", pred)
        predicted_class_indices = np.argmax(pred, axis=1)

        labels = train_generator.class_indices
        print("Labels: ", labels)
        labels = dict((v, k) for k, v in labels.items())
        print("Second labels:", labels)
        predictions = [labels[k] for k in predicted_class_indices]
        #
        filenames = test_generator.filenames
        #
        print(filenames, predictions, "\n")

        # # Plotting the model accuracy
        # epochs = range(len(acc))  # Get number of epochs
        #
        # plt.plot(epochs, acc, 'r', "Training Accuracy")
        # plt.plot(epochs, val_acc, 'b', "Validation Accuracy")
        # plt.title('Training and validation accuracy')
        # plt.figure()
        #
        # plt.plot(epochs, loss, 'r', "Training Loss")
        # plt.plot(epochs, val_loss, 'b', "Validation Loss")
        #
        # plt.title('Training and validation loss')
        # plt.show()
