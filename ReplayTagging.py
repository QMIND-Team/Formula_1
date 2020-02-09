import cv2
import glob
import os
import csv

def match_frame():
    a = 0

def main():
    replayImage = cv2.imread("replay.jpg", cv2.IMREAD_GRAYSCALE)
    actual_pos = 0
    actual_neg = 0
    false_pos = 0
    false_neg = 0
    with open("TaggedFrames.csv") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        headers = next(csv_reader)
        
        for row in csv_reader:
            filename = "frames/" + row[1] + ".jpg"
            if os.path.isfile(filename):
                image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
                match = cv2.matchTemplate(image, replayImage, cv2.TM_SQDIFF)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(match)

                # 99+-1 and 55+-1
                if min_loc[0] >= 98 and min_loc[0] <= 100 and min_loc[1] >= 54 and min_loc[1] <= 56 and min_val < 150000:
                    if row[2] == "TRUE":
                        actual_pos += 1
                    else:
                        false_pos += 1m
                else:
                    if row[2] == "FALSE":
                        actual_neg += 1
                    else:
                        false_neg += 1
        print("[[{}\t{}]\n [{}\t{}]]".format(actual_pos, false_pos, false_neg, actual_neg))
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