import csv
import os
import cv2
import numpy as np
from PIL import ImageFile
import keras as K
import cv2
import os

class CameraAngleClassifier():
    def __init__(self, trackViewModelFilename, otherViewModelFilename):
        super().__init__()
        self.isTrackViewModel = K.models.load_model(trackViewModelFilename)
        self.otherViewModel = K.models.load_model(otherViewModelFilename)

    def getCameraAngle(self, image):
        '''
        image should be a numpy array of shape (x, y, 3)
        '''
        x = K.backend.reshape(K.backend.constant(cv2.resize(image, (299, 299))), shape=(1, 299, 299, 3))
        isTrackView = np.argmax(self.isTrackViewModel.predict(x))
        if (isTrackView == 1):
            return 'Track View'
        otherViewNames = ['Driver View' 'Other View' 'Pit View' 'Spectator View']
        otherView = np.argmax(self.otherViewModel.predict(x))
        return otherViewNames[otherView]


if __name__ == '__main__':
    cameraAngleClassifier = CameraAngleClassifier("IsTrackViewModel.h5", "OtherViewModel.h5")
    filedir = '../../data/test/isOtherView'
    for filename in os.listdir(filedir):
        image = cv2.imread(os.path.join(filedir, filename), cv2.IMREAD_COLOR)
        result = cameraAngleClassifier.getCameraAngle(image)
        print(result)
    pass
