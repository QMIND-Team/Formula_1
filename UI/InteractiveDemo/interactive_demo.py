import os
import sys

from PyQt5 import QtWidgets, QtMultimedia, uic, QtCore
from PyQt5.QtWidgets import (QApplication, QFileDialog, QHBoxLayout, QLineEdit,
                             QPushButton, QSizePolicy, QSlider, QMessageBox, QStyle, QVBoxLayout,
                             QWidget, QShortcut, QMenu)
from PyQt5.QtGui import QPalette, QKeySequence, QIcon
from PyQt5.QtCore import QDir, Qt, QUrl, QSize, QPoint, QTime, QMimeData, QProcess, QEvent
from PyQt5.QtGui import QFont, QPainter, QImage, QTextCursor
import sys, time, threading, cv2
from PyQt5.QtCore import QTimer, QPoint, pyqtSignal
from PyQt5.QtWidgets import QApplication, QMainWindow, QTextEdit, QLabel
from PyQt5.QtWidgets import QWidget, QAction, QVBoxLayout, QHBoxLayout
from PyQt5.QtGui import QFont, QPainter, QImage, QTextCursor
import queue as Queue

WINDOW_TITLE = "Interactive Classification"
IMG_SIZE = 240, 120  # 640,480 or 1280,720 or 1920,1080
IMG_FORMAT = QImage.Format_RGB888
DISP_SCALE = 4  # Scaling factor for display image
DISP_MSEC = 50  # Delay between display cycles
CAP_API = cv2.CAP_ANY  # API: CAP_ANY or CAP_DSHOW etc...
EXPOSURE = 0  # Zero for automatic exposure
TEXT_FONT = QFont("Courier", 10)

camera_num = 1  # Default camera (first in list)
image_queue = Queue.Queue()  # Queue to hold images
capturing = True  # Flag to indicate capturing


# Grab images from the camera (separate thread)
def grab_images(cam_num, queue):
    cap = cv2.VideoCapture(cam_num - 1 + CAP_API)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMG_SIZE[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMG_SIZE[1])
    if EXPOSURE:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
        cap.set(cv2.CAP_PROP_EXPOSURE, EXPOSURE)
    else:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
    while capturing:
        if cap.grab():
            retval, image = cap.retrieve(0)
            if image is not None and queue.qsize() < 2:
                queue.put(image)
            else:
                time.sleep(DISP_MSEC / 1000.0)
        else:
            print("Error: can't grab camera image")
            break
    cap.release()

class Form(QtWidgets.QDialog):
    def __init__(self, parent=None):
        QtWidgets.QDialog.__init__(self, parent)
        self.ui = uic.loadUi(os.path.join(os.path.dirname(__file__), "interactive_layout.ui"), self)
        self.player = QtMultimedia.QMediaPlayer(None, QtMultimedia.QMediaPlayer.VideoSurface)
        file = os.path.join(os.path.dirname(__file__), "/Users/jd/Downloads/BritishGP.m4v")
        self.player.setMedia(QtMultimedia.QMediaContent(QtCore.QUrl.fromLocalFile(file)))
        self.player.setVideoOutput(self.ui.video)


    # Key Shortcuts
        self.shortcut = QShortcut(QKeySequence("q"), self)  # Quit
        self.shortcut.activated.connect(self.handleQuit)
        self.shortcut = QShortcut(QKeySequence("f"), self)  # Fullscreen
        self.shortcut.activated.connect(self.handleFullscreen)

    # Buttons
        self.play_button.clicked.connect(self.play_button_clicked)
        self.reset_button.clicked.connect(self.reset_button_clicked)

    def play_button_clicked(self):
        self.player.play()

    def reset_button_clicked(self):
        self.player.setPosition(0)
        self.player.stop()

    def handleQuit(self):
        global capturing
        capturing = False
        self.capture_thread.join()

        app.quit()

    def handleFullscreen(self):
        if self.windowState() & Qt.WindowFullScreen:
            self.showNormal()
        else:
            self.showFullScreen()

    # Start image capture & display
    def start(self):
        self.timer = QTimer(self)  # Timer to trigger display
        self.timer.timeout.connect(lambda:
                                   self.show_image(image_queue, self.camera, DISP_SCALE))
        self.timer.start(DISP_MSEC)
        self.capture_thread = threading.Thread(target=grab_images,
                                               args=(camera_num, image_queue))
        self.capture_thread.start()  # Thread to grab images

    # Fetch camera image from queue, and display it
    def show_image(self, imageq, display, scale):
        if not imageq.empty():
            image = imageq.get()
            if image is not None and len(image) > 0:
                img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                self.display_image(img, display, scale)

    # Display an image, reduce size if required
    def display_image(self, img, display, scale=1):
        disp_size = IMG_SIZE
        disp_bpl = disp_size[0] * 3
        if scale > 1:
            img = cv2.resize(img, disp_size,
                             interpolation=cv2.INTER_CUBIC)
        qimg = QImage(img.data, disp_size[0], disp_size[1],
                      disp_bpl, IMG_FORMAT)
        display.setImage(qimg)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    w = Form()
    w.show()
    w.start()
    sys.exit(app.exec())
