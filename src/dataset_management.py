# Dataset: https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection

import os
import random
from shutil import copyfile

from keras.preprocessing.image import ImageDataGenerator
from os import listdir
import cv2


class LoadDataset:

    def __init__(self) -> None:
        pass

    def manage_directories(self):
        # Create directories for augmented, training, and validation dataset
        try:
            os.mkdir('trial1')
            os.mkdir('trial1/augmented data1/')
            os.mkdir('trial1/augmented data1/yes_real')
            os.mkdir('trial1/augmented data1/no_real')
            os.mkdir('trial1/augmented data1/training')
            os.mkdir('trial1/augmented data1/training/yes')
            os.mkdir('trial1/augmented data1/training/no')
            os.mkdir('trial1/augmented data1/testing')
            os.mkdir('trial1/augmented data1/testing/yes')
            os.mkdir('trial1/augmented data1/testing/no')
        except OSError:
            pass

    def augment_data(self, file_dir, no_of_samples, output_dir):

        # Using ImageDataGenerator for data augmentation
        data_gen = ImageDataGenerator(rotation_range=10,
                                      width_shift_range=0.1,
                                      height_shift_range=0.1,
                                      shear_range=0.1,
                                      brightness_range=(0.3, 1.0),
                                      horizontal_flip=True,
                                      vertical_flip=True,
                                      fill_mode='nearest'
                                      )

        for filename in listdir(file_dir):
            # Load and reshape the image
            image = cv2.imread(file_dir + '\\' + filename)
            image = image.reshape((1,) + image.shape)
            save_prefix = 'aug_' + filename[:-4]

            # Generate 'n_generated_samples' sample images
            i = 0
            for batch in data_gen.flow(x=image, batch_size=1, save_to_dir=output_dir,
                                       save_prefix=save_prefix, save_format='jpg'):
                i += 1
                if i > no_of_samples:
                    break

    def store_augmented_data(self):
        augmented_data_path = 'trial1/augmented data1/'

        # augment data for the examples with label equal to 'yes' representing tumorous examples
        self.augment_data(file_dir='data/yes', no_of_samples=6, output_dir=augmented_data_path + 'yes_real')
        # augment data for the examples with label equal to 'no' representing non-tumorous examples
        self.augment_data(file_dir='data/no', no_of_samples=9, output_dir=augmented_data_path + 'no_real')

        def split_data(source, training, testing, split_size):
            dataset = []

            for unitData in os.listdir(source):
                data = source + unitData
                if os.path.getsize(data) > 0:
                    dataset.append(unitData)
                else:
                    print('Skipped ' + unitData)
                    print('Invalid file i.e zero size')

            train_set_length = int(len(dataset) * split_size)
            test_set_length = int(len(dataset) - train_set_length)
            shuffled_set = random.sample(dataset, len(dataset))
            train_set = dataset[0:train_set_length]
            test_set = dataset[-test_set_length:]

            for unitData in train_set:
                temp_train_set = source + unitData
                final_train_set = training + unitData
                copyfile(temp_train_set, final_train_set)

            for unitData in test_set:
                temp_test_set = source + unitData
                final_test_set = testing + unitData
                copyfile(temp_test_set, final_test_set)

        yes_source_dir = "trial1/augmented data1/yes_real/"
        training_yes_dir = "trial1/augmented data1/training/yes/"
        testing_yes_dir = "trial1/augmented data1/testing/yes/"
        no_source_dir = "trial1/augmented data1/no_real/"
        training_no_dir = "trial1/augmented data1/training/no/"
        testing_no_dir = "trial1/augmented data1/testing/no/"
        training_size = .8
        split_data(yes_source_dir, training_yes_dir, testing_yes_dir, training_size)
        split_data(no_source_dir, training_no_dir, testing_no_dir, training_size)
