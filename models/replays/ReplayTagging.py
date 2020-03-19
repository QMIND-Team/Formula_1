# Multi Scale Template Code adapted from https://www.pyimagesearch.com/2015/01/26/multi-scale-template-matching-using
# -python-opencv/
import cv2
import glob
import os
import csv

import imutils
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))  # Sets cwd to this file's directory
template = cv2.imread('replay.jpg', cv2.IMREAD_GRAYSCALE)


def main():
    actual_pos = 0
    actual_neg = 0
    false_pos = 0
    false_neg = 0
    correct = 0
    wrong = 0

    with open("../../labels/Frame Tagger - Singapore.csv") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        headers = next(csv_reader)

        for row in csv_reader:
            if row[2] != "TRUE" and row[2] != "FALSE":
                print(row[3])
                print("Ignoring {}, not T or F".format(row[1]))
                continue
            elif row[3] == "":
                continue

            filename = "../../frames/Singapore Frames/" + row[1] + ".jpg"
            print(filename)
            if os.path.isfile(filename):
                image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
                match = cv2.matchTemplate(image, template, cv2.TM_SQDIFF)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(match)

                print(is_replay(image))
                # 99+-1 and 55+-1
                if min_loc[0] >= 98 and min_loc[0] <= 100 and min_loc[1] >= 54 and min_loc[
                    1] <= 56 and min_val < 150000:
                    if row[2] == "TRUE":
                        actual_pos += 1
                    else:
                        false_pos += 1
                else:
                    if row[2] == "FALSE":
                        actual_neg += 1
                        correct += 1
                    else:
                        false_neg += 1
        print("[[{}\t{}]\n [{}\t{}]]".format(actual_pos, false_pos, false_neg, actual_neg))
    csv_file.close()


def is_replay(img, debug=False):
    w, h = template.shape[::-1]
    found = None

    # loop over the scales of the image
    for scale in np.linspace(0.8, 1.5, 2)[::-1]:
        # resize the image according to the scale, and keep track
        # of the ratio of the resizing
        resized = imutils.resize(img, width=int(img.shape[1] * scale))
        r = img.shape[1] / float(resized.shape[1])

        # if the resized image is smaller than the template, then break
        # from the loop
        if resized.shape[0] < h or resized.shape[1] < w:
            break

        # detect edges in the resized, grayscale image and apply template
        # matching to find the template in the image

        edged = cv2.Canny(resized, 50, 200)
        result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF)
        (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

        if debug:
            # check to see if the iteration should be visualized
            # draw a bounding box around the detected region
            clone = np.dstack([edged, edged, edged])
            cv2.rectangle(clone, (maxLoc[0], maxLoc[1]),
                          (maxLoc[0] + w, maxLoc[1] + h), (0, 0, 255), 2)
            cv2.imshow("Visualize", clone)
            cv2.waitKey(0)

        # if we have found a new maximum correlation value, then update
        # the bookkeeping variable
        if found is None or maxVal > found[0]:
            found = (maxVal, maxLoc, r)

    # unpack the bookkeeping variable and compute the (x, y) coordinates
    # of the bounding box based on the resized ratio
    (_, maxLoc, r) = found
    (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
    (endX, endY) = (int((maxLoc[0] + w) * r), int((maxLoc[1] + h) * r))

    if debug:
        # draw a bounding box around the detected result and display the image
        cv2.rectangle(img, (startX, startY), (endX, endY), (255, 255, 0), 5)
        cv2.imshow("Debug", img)
        print("{} {} : {} {}".format(startX, endX, startY, endY))

    h, w = img.shape
    if w*.02 <= startX <= w*.07 and h*.01 <= startY <= h*.07:
        return True

    return False

if __name__ == "__main__":
    main()
