from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os

# GLOBALS
output_folder_path = ".." + os.path.sep + ".." + os.path.sep + "data" + os.path.sep + "team_on_screen" + os.path.sep

# generates histograms for images in ndarray form.
# data written out to csv, 256 x 3.
# 256 col corresponding to the pixel count in each of the
# 256 pixel intensity bins as a percentage of the total pixel count
# in that channel.
# 3 row for the corresponding values in each of the red green and blue
# components.
# input should be cropped images, without background to minimize noise.
def generateHistogram(imgs):
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

# read in images into (ndarray form, file name) tuple
def load_images(folder):
    images = []
    for filename in os.listdir(folder):
        img = mpimg.imread(os.path.join(folder, filename))
        if img is not None:
            images.append((img, filename))
    return images


def main():
    imgs = load_images("test_imgs")
    generateHistogram(imgs)


if __name__ == "__main__":
    main()

'''
cv2.imshow('image',test_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''

