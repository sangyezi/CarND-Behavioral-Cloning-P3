from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.callbacks import EarlyStopping

import csv
import cv2
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import sys
import pickle
import glob

def build_model(dropout=0.2):
    model = Sequential()
    model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(160, 320, 3)))

    # normalized layer
    model.add(Lambda(lambda x: x / 255.0 - 0.5))

    # five convolution layers
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation="relu"))
    # model.add(Dropout(dropout))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation="relu"))
    # model.add(Dropout(dropout))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation="relu"))
    # model.add(Dropout(dropout))
    model.add(Convolution2D(64, 3, 3, activation="relu"))
    # model.add(Dropout(dropout))
    model.add(Convolution2D(64, 3, 3, activation="relu"))
    # model.add(Dropout(dropout))
    model.add(Flatten())
    model.add(Dense(1164))
    model.add(Dropout(dropout))
    model.add(Dense(100))
    model.add(Dropout(dropout))
    model.add(Dense(50))
    model.add(Dropout(dropout))
    model.add(Dense(10))
    model.add(Dropout(dropout))
    model.add(Dense(1))
    model.add(Dropout(dropout))

    model.compile(loss='mse', optimizer='adam')

    return model


def get_samples(path, file_name, correction=0.2):
    corrections = [0, correction, -correction]
    samples = []
    img_folder = 'IMG/'

    with open(path + file_name) as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            if img_folder in row[0]:
                center_angle = float(row[3])
                for i in range(0, 3):
                    name = path + img_folder + row[i].split('/')[-1]
                    # image = cv2.imread(name)
                    # if image is not None and image.shape == (160, 320, 3) and center_angle is not None:
                    angle = center_angle + corrections[i]
                    samples.append((name, angle))
    return samples


def generator(samples, batch_size=32):
    num_samples = len(samples)
    shuffle(samples)
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                image = cv2.imread(batch_sample[0])

                if image is not None and image.shape == (160, 320, 3):
                    b, g, r = cv2.split(image)  # get b,g,r
                    image = cv2.merge([r, g, b])  # switch it to rgb
                    angle = batch_sample[1]

                    images.append(image)
                    angles.append(angle)
                    images.append(cv2.flip(image, 1))
                    angles.append(angle * -1.0)

            X_train = np.array(images)
            y_train = np.array(angles)

            if len(images) > 0:
                yield shuffle(X_train, y_train)


def main(dropout=0.2, nb_epoch=3, batch_size=32, correction=0.2):
    pathes = glob.glob('./training_data/track*set*/')
    
    log_csv = 'driving_log_full.csv'

    samples = []

    for path in pathes:
        samples.extend(get_samples(path, log_csv, correction=correction))

    shuffle(samples)

    train_samples, valid_samples = train_test_split(samples, test_size=0.2)

    train_generator = generator(train_samples, batch_size=int(batch_size / 2))
    valid_generator = generator(valid_samples, batch_size=int(batch_size / 2))

    model = build_model(dropout=dropout)

    cb = EarlyStopping(monitor='val_loss', min_delta=.005, patience=1, verbose=1, mode='auto')

    history_object = model.fit_generator(train_generator, samples_per_epoch=len(train_samples) * 2,
                        validation_data=valid_generator, nb_val_samples=len(valid_samples) * 2,
                        nb_epoch=nb_epoch, callbacks=[cb])
    model.save('model.h5')

    with open('history_object.p', 'wb') as history_file:
        pickle.dump(history_object.history, history_file)

    # fix bug: https://github.com/tensorflow/tensorflow/issues/3388
    from keras import backend as K
    K.clear_session()

if __name__ == "__main__":
    dropout = 0.2
    nb_epoch = 10
    batch_size = 32
    correction = 0.2

    if len(sys.argv) > 1:
        for argv in sys.argv[1:]:
            name, val = argv.split('=')
            if "drop" in name:
                dropout = float(val)
            if "epoch" in name:
                nb_epoch = int(val)
            if "batch" in name:
                batch_size = int(val)
            if "correction" in name:
                correction = float(val)

    main(dropout=dropout, nb_epoch=nb_epoch, batch_size=batch_size, correction=correction)
