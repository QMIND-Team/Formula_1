import os
import fileinput
import glob
from shutil import copyfile

train_path = os.path.abspath("..")
percentage_test = 50


# helper functions for processing labels and gathering labeled data


def main():
    pass
    #fix_annotation('annotations')
    #copy_annotated_frames('annotations', train_path)

def fix_annotation(path):
    for filename in os.listdir(path):
        for line in fileinput.input(path + os.path.sep + filename, True):
            split = line.split(" ")
            if split[0] == '4':
                split[0] = "0"
                print(" ".join(split), end='')
            else:
                print(line, end='')

# copies all annotated frames into local directory
def copy_annotated_frames(annotation_path, frame_folder_path):
    dir = os.path.dirname(__file__)
    annotated_filenames = []
    for filename in os.listdir(annotation_path):
        if filename != "classes.txt":
            annotated_filenames.append(filename.split(".")[0])
        for view_folder in os.listdir(frame_folder_path):
            for frame_filename in annotated_filenames:
                if os.path.exists(frame_folder_path + os.path.sep + view_folder + os.path.sep + frame_filename + ".jpg"):
                    src = frame_folder_path + os.path.sep + view_folder + os.path.sep + frame_filename + ".jpg"
                    dst = (annotation_path + os.path.sep + frame_filename + ".jpg")
                    print(copyfile(src, dst))


if __name__ == "__main__":
    main()
