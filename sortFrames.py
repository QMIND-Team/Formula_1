import csv
import os
import numpy as np
from sklearn.model_selection import train_test_split

def main(csv_path, frame_path, new_path):

    img_class_arr = []
    with open(csv_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        headers = next(csv_reader)  # Records the first row of csv

        for row in csv_reader:
            if row[4] != "":
                img_name = row[1] + ".jpg"
                img_class_arr.append([img_name, row[4]])

        unique_classes = np.unique([item[1] for item in img_class_arr])

        for c in unique_classes:
            current_class = [item[0] for item in img_class_arr if item[1] == c]
            current_class_train, current_class_test = train_test_split(current_class, train_size=.8)


            #  Move images into folders
            #  Train
            for img_name in current_class_train:
                src_path = os.path.join(frame_path, img_name)
                if os.path.exists(src_path):
                    os.rename(src_path, os.path.join(new_path, "train", c, img_name))
            #  Test
            for img_name in current_class_test:
                src_path = os.path.join(frame_path, img_name)
                if os.path.exists(src_path):
                    os.rename(src_path, os.path.join(new_path, "test", c, img_name))

    csv_file.close()

if __name__ == '__main__':
    main("/Users/jd/Documents/Formula_1/labels/Frame Tagger - Singapore.csv", "/Users/jd/Documents/Formula_1/frames/Singapore Frames", "/Users/jd/Documents/Formula_1/data")