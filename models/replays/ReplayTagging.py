import cv2
import glob
import os
import csv

def match_frame():
    a = 0

def main():
    replayImage = cv2.imread("replay.jpg", cv2.IMREAD_GRAYSCALE)
    correct = 0
    wrong = 0

    with open("../../labels/Frame Tagger - US.csv") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        headers = next(csv_reader)

        for row in csv_reader:
            if row[2] != "TRUE" and row[2] != "FALSE":
                print(row[3])
                print("Ignoring {}, not T or F".format(row[1]))
                continue
            elif row[3] == "":
                continue

            filename = "../../frames/US Frames/" + row[1] + ".jpg"
            print(filename)
            if os.path.isfile(filename):
                image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
                match = cv2.matchTemplate(image, replayImage, cv2.TM_SQDIFF)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(match)

                if min_loc[0] == 99 and min_loc[1] == 55:
                    if row[2] == "TRUE":
                        correct += 1
                    else:
                        wrong += 1
                        print("FALSE NEGATIVE: " + row[1])
                else:
                    if row[2] == "FALSE":
                        correct +=1
                    else:
                        wrong += 1
                        print("FALSE POSITIVE: " + row[1])
        print("CORRECT: {}    WRONG: {}".format(correct, wrong))
    csv_file.close()

# def main():
#     replayImage = cv2.imread("replay.jpg", cv2.IMREAD_GRAYSCALE)
#     for filename in glob.glob("frames/*.jpg"):
#         image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
#         cv2.imshow('image', image)
#         cv2.waitKey()
#         methods = ['cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
#         for meth in methods:
#             method = eval(meth)
#             match = cv2.matchTemplate(image, replayImage, method)
#             min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(match)
#             match_color = cv2.cvtColor(match, cv2.COLOR_GRAY2RGB)
#             if meth == 'cv2.TM_CCOEFF_NORMED' or meth =='cv2.TM_CCORR_NORMED':
#                 loc = max_loc
#             else: 
#                 loc = min_loc
#             cv2.circle(match_color, loc, 20, (0, 0, 255), 5)
#             cv2.imshow('image', match_color)
#             print(loc)
#             cv2.waitKey()

if __name__ == "__main__":
    main()