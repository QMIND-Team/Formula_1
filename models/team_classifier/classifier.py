from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import glob
import os

# GLOBALS
output_folder_path = ".." + os.path.sep + ".." + os.path.sep + "data" + os.path.sep + "team_on_screen" + os.path.sep
annotation_path = ".." + os.path.sep + ".." + os.path.sep + "data" + os.path.sep + "train" + os.path.sep + "isTrackView" + os.path.sep
percentage_test = 20

# generates histograms for images in ndarray form.
# data written out to csv, 256 x 3.
# 256 col corresponding to the pixel count in each of the
# 256 pixel intensity bins as a percentage of the total pixel count
# in that channel.
# 3 row for the corresponding values in each of the red green and blue
# components.
# input should be cropped images, without background to minimize noise.
def generate_histogram(imgs):
    for img in imgs:
        hists = []
        chans = cv2.split(img[0])
        colors = ("b", "g", "r")
        for(chan, color) in zip(chans, colors):
            hist = cv2.calcHist([chan], [0], None, [256], [0, 256]).flatten()
            hist = hist / chan.size
            hists.append(hist)
        hist_arr = np.array(hists)
        np.savetxt(output_folder_path + img[1].split(".jpg")[0] + ".csv", hist_arr, delimiter=',')


# read .jpg files and encode as (ndarray form, file name) tuple
def encode_images(folder):
    images = []
    for filename in os.listdir(folder):
        img = mpimg.imread(os.path.join(folder, filename))
        if img is not None:
            images.append((img, filename))
    return images


def split_dataset(img_path):
    file_train = open(img_path + '..' + os.path.sep + 'train.txt', 'w')
    file_test = open(img_path + '..' + os.path.sep + 'valid.txt', 'w')
    counter = 1
    index_test = round(100 / percentage_test)
    for file in glob.iglob(os.path.join(img_path, '*.jpg')):
        title, ext = os.path.splitext(os.path.basename(file))
        if counter == index_test:
            counter = 1
            file_test.write(img_path + title + ".jpg" + "\n")
        else:
            file_train.write(img_path + title + ".jpg" + "\n")
            counter = counter + 1


def generate_bounding_boxes(img):
    pass


def main():
    split_dataset(annotation_path)

    #imgs = encode_images("test_imgs")
    #generate_histogram(imgs)


if __name__ == "__main__":
    main()

'''
cv2.imshow('image',test_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''

