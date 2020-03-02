from PyQt5 import QtWidgets, QtMultimedia, uic, QtCore
from PyQt5.QtWidgets import (QApplication, QFileDialog, QHBoxLayout, QLineEdit,
                             QPushButton, QSizePolicy, QSlider, QMessageBox, QStyle, QVBoxLayout,
                             QWidget, QShortcut, QMenu)
from PyQt5.QtGui import QPalette, QKeySequence, QIcon
from PyQt5.QtCore import QDir, Qt, QUrl, QSize, QPoint, QTime, QMimeData, QProcess, QEvent
from PyQt5.QtGui import QFont, QPainter, QImage, QTextCursor

# Image widget
class ImageWidget(QWidget):
    def __init__(self, parent=None):
        super(ImageWidget, self).__init__(parent)
        self.image = None

    def setImage(self, image):
        self.image = image
        self.setMinimumSize(image.size())
        self.update()

    def paintEvent(self, event):
        qp = QPainter()
        qp.begin(self)
        if self.image:
            qp.drawImage(QPoint(0, 0), self.image)
        qp.end()
