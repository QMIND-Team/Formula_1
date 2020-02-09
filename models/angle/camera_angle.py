import csv
import os
import cv2
import numpy as np
from PIL import ImageFile
from keras_preprocessing.image import ImageDataGenerator
from sklearn.utils import class_weight
from keras.applications import InceptionV3
from keras.models import Model
from keras import Sequential
from keras.activations import softmax
from keras.layers import Dense


def process_csv(csv_path):
    frame_dict = {}
    class_list = []
    with open(csv_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        headers = next(csv_reader)  # Records the first row of csv

        for row in csv_reader:
            if row[4] != "":
                frame_dict[row[1]] = row[4]
                class_list.append(row[4])

    csv_file.close()
    return frame_dict, class_list


def camera_angle(class_list, class_weights):
    original_model = InceptionV3()
    bottleneck_input = original_model.get_layer(index=0).input
    bottleneck_output = original_model.get_layer(index=-2).output
    bottleneck_model = Model(inputs=bottleneck_input, outputs=bottleneck_output)

    for layer in bottleneck_model.layers:
        layer.trainable = False

    new_model = Sequential()
    new_model.add(bottleneck_model)
    new_model.add(Dense(5, activation=softmax, input_dim=2048))

    # For a binary classification problem
    new_model.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

    datagen = ImageDataGenerator()
    train_it = datagen.flow_from_directory("/Users/jd/Documents/Formula_1/data/train", target_size=(299, 299),
                                           batch_size=64, class_mode="categorical", color_mode="rgb")
    test_it = datagen.flow_from_directory("/Users/jd/Documents/Formula_1/data/test", target_size=(299, 299),
                                          batch_size=64, class_mode="categorical", color_mode="rgb")

    ImageFile.LOAD_TRUNCATED_IMAGES = True
    new_model.fit_generator(train_it, class_weight=class_weights, epochs=3)

    print(new_model.evaluate_generator(test_it))


if __name__ == '__main__':
    frame_dict, class_list = process_csv("../../labels/Frame Tagger - Singapore.csv")

    # image_preprocessor("../../frames/Singapore Frames", frame_dict)

    print(np.unique(class_list))

    class_weights = class_weight.compute_class_weight('balanced', np.unique(class_list), class_list)

    camera_angle(class_list, class_weights)

    print(np.unique(class_list))
    print(class_weights)
    pass
