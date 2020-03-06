from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import glob
import os


# GLOBALS
output_folder_path = ".." + os.path.sep + ".." + os.path.sep + "data" + os.path.sep + "team_on_screen" + os.path.sep
annotation_path = ".." + os.path.sep + ".." + os.path.sep + "data" + os.path.sep + "train" + os.path.sep + "isTrackView" + os.path.sep
train_path = ".." + os.path.sep + ".." + os.path.sep + "data" + os.path.sep + "train" + os.path.sep + "carDetection" + os.path.sep
test_path = ".." + os.path.sep + ".." + os.path.sep + "data" + os.path.sep + "test" + os.path.sep + "isTrackView" + os.path.sep
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


def implement_model(img_path, conf_path, weights_path, class_path):

    image = cv2.imread(img_path)
    width = image.shape[1]
    height = image.shape[0]
    scale = 0.00392

    classes = None
    with open(class_path, 'r') as f:
        classes = [line.strip() for line in f.readlines()]


    net = cv2.dnn.readNet(weights_path, conf_path)
    blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)
    net.setInput(blob)

    def get_output_layers():
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        return output_layers

    def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
        label = str(classes[class_id])
        color = (255,255,255)
        cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)
        cv2.putText(img, "%s %.2f" % (label, confidence * 100), (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    outs = net.forward(get_output_layers())

    #init
    class_ids = []
    confidences = []
    boxes = []
    conf_thres = 0.5
    nms_thres = 0.4

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_thres:
                centre_x = int(detection[0] * width)
                centre_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = centre_x - w/2
                y = centre_y - h/2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    # apply non-max suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_thres, nms_thres)

    # go through the detections remaining
    # after nms and draw bounding box
    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]

        draw_bounding_box(image, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h))

    # display output image
    cv2.imshow("object detection", image)

    # wait until any key is pressed
    cv2.waitKey()

    # save output image to disk
    cv2.imwrite("object-detection.jpg", image)

    # release resources
    cv2.destroyAllWindows()

def generate_bounding_boxes(img):
    pass

def main():

    image = os.path.abspath(test_path + 'singapore-frame_70550.jpg')

    implement_model(image, "yolov3_obj.cfg", "yolov3_final.weights", "obj.names")

    #imgs = encode_images("test_imgs")
    #generate_histogram(imgs)


if __name__ == "__main__":
    main()

'''
cv2.imshow('image',test_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''

