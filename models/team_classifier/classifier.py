import numpy as np
import cv2
import os
import time
import operator
import pprint
from matplotlib import pyplot as plt

# GLOBALS
control_path = ".." + os.path.sep + ".." + os.path.sep + "data" + os.path.sep + "team_on_screen" + os.path.sep + 'control_hist' + os.path.sep
annotation_path = ".." + os.path.sep + ".." + os.path.sep + "data" + os.path.sep + "train" + os.path.sep + "isTrackView" + os.path.sep
train_path = ".." + os.path.sep + ".." + os.path.sep + "data" + os.path.sep + "train" + os.path.sep
test_path = ".." + os.path.sep + ".." + os.path.sep + "data" + os.path.sep + "test" + os.path.sep
percentage_test = 20


# generates histograms for images in ndarray form.
# data written out to csv, 256 x 3.
# 256 col corresponding to the pixel count in each of the
# 256 pixel intensity bins as a percentage of the total pixel count
# in that channel.
# 3 row for the corresponding values in each of the red green and blue
# components.
# input should be cropped images, without background to minimize noise.
def generate_histogram(img):
    hists = []
    chans = cv2.split(img)
    colors = ("b", "g", "r")
    for (chan, color) in zip(chans, colors):
        hist = cv2.calcHist([chan], [0], None, [256], [0, 256]).flatten()
        hist = hist / chan.size
        hists.append(hist)
    return np.array(hists)



def return_intersection(hist_1, hist_2):
    minima = np.minimum(hist_1,hist_2)
    return np.true_divide(np.sum(minima), np.sum(hist_2))




def classify_team(img):
    intersections = {}
    frame_hist = generate_histogram(img)
    for control in os.listdir(control_path):
        control_hist = np.genfromtxt(os.path.join(control_path, control), delimiter=',')
        intersections[control] = return_intersection(control_hist, frame_hist)
    # pprint.pprint(intersections)
    return max(intersections.items(), key=operator.itemgetter(1))[0]



'''
frame_bound_box
    places bounding boxes, class labels and confidences around each detected
    f1 car in the given image.

inputs:
    cvimg : image processed into ndarray by cv2
    classes : list of classes model has trained on
    net : loaded yolo model
    out_layers : output layers of yolo model
returns:
    image : cvimg with bounding boxes around f1 cars superimposed 
'''

def frame_bound_box(cvimg, classes, net, out_layers):
    image = cvimg
    width = image.shape[1]
    height = image.shape[0]
    scale = 0.00392
    class_ids = []
    confidences = []
    boxes = []
    conf_thres = 0.5
    nms_thres = 0.4

    def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
        #label = str(classes[class_id])
        label = class_id
        color = (255, 255, 255)
        cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
        cv2.putText(img, "%s %.2f" % (label, confidence * 100), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416,416), swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    outs = net.forward(out_layers)
    end = time.time()
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[0]
            if confidence > conf_thres:
                centre_x = int(detection[0] * width)
                centre_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(centre_x - w / 2)
                y = int(centre_y - h / 2)
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
        if (box and min(box) >= 0):
            crop = image[y:y + h, x:x + w]
            #cv2.imshow('crop', crop)
            class_id = classify_team(crop)
            class_ids[i] = class_id

        draw_bounding_box(image, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h))
    print("took %.2f seconds" % (end-start))
    return image

'''
video_bound_box 
    displays the input video as each frame passes though the yolo network
    placing bounding boxes around any detected f1 cars.

inputs: 
    in_video : path to video to be processed
    cfg : path to yolov3_obj.cfg
    weights : path to yolov3_final.weights
    classes : path to obj.names
    
returns: 
    None
'''

def video_bound_box(in_video, cfg, weights, classes):
    labels = open(classes).read().strip().split("\n")
    net = cv2.dnn.readNet(weights, cfg)
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    while True:
        vs = cv2.VideoCapture(in_video)
        m = 0
        while True:
            (ret, frame) = vs.read()
            if not ret:
                break
            m += 1
            # if m % 60 != 0 :
            #     continue
            frame = frame_bound_box(frame, labels, net, ln)
            frame = cv2.resize(frame, (1920, 1080))
            cv2.putText(frame, '%d' % m, (1880, 1060), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
            cv2.imshow('frame', frame)
            cv2.resizeWindow('frame', 1920, 1080)
            print("frame %d " % m)
            if cv2.waitKey(1) == ord('q'):
                break
        vs.release()
        cv2.destroyAllWindows()

def main():
    #image = os.path.abspath(test_path + 'germany_test720.mp4')
    image = os.path.abspath(test_path + 'singapore_test.mp4')
    # uncomment below to run 720p clip on yolov3-tiny network -> better performance
    video_bound_box(image, "yolov3-tiny/yolov3-tiny_obj.cfg", "yolov3-tiny/yolov3-tiny_obj_8000.weights", "obj.names")
    # uncomment below to run 720p clip on full yolov3 network -> more accurate boxing
    #video_bound_box(image, "yolov3/yolov3_obj.cfg", "yolov3/yolov3_final.weights", "obj.names")


if __name__ == "__main__":
    main()

