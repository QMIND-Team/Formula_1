from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import cv2
import os

def generate1DimHistogram (imgs):
    for img in imgs:
        chans = cv2.split(img[0])
        colors = ("b", "g", "r")

        plt.figure()
        plt.title(img[1])
        plt.xlabel("bins")
        plt.ylabel("# of pixels")
        features = []

        # loop through image channels
        for(chan, color) in zip(chans, colors):
            # create histogram for current channel and concat resulting hist
            # for each channel
            hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
            features.extend(hist)

            # plot the histogram
            plt.plot(hist, color=color)
            plt.xlim([0,256])
        plt.show()

def load_images(folder):
    images = []
    for filename in os.listdir(folder):
        img = mpimg.imread(os.path.join(folder, filename))
        if img is not None:
            images.append((img, filename))
    return images

def main():
    imgs = load_images("test_imgs")
    generate1DimHistogram(imgs)

if __name__ == "__main__":
    main()

'''
cv2.imshow('image',test_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''

