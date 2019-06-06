from PyQt5 import QtGui
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget
from PyQt5.QtGui import QPainter, QBrush, QPen
from PyQt5.QtCore import Qt
import sys
import time
 
 
class Main(QMainWindow):
    def __init__(self, parent=None):
        super(Main, self).__init__(parent)

        self.title = "Green NeopJukEE"
        self.top = 0
        self.left = 0
        self.width = 1450
        self.height = 800
     
        self.InitWindow()
 
 
    def InitWindow(self):
        self.setWindowIcon(QtGui.QIcon("icon.png"))
        self.setWindowTitle(self.title)
        self.setGeometry(self.top, self.left, self.width, self.height)
        self.show()
 
    def paintEvent(self, e):
        # default mode is red
        painter = QPainter(self)
        painter.setPen(QPen(Qt.darkGray, 5, Qt.SolidLine))
        painter.setBrush(QBrush(Qt.black, Qt.SolidPattern))
        painter.drawRect(550, 50, 350, 700)

        painter.setPen(QPen(Qt.darkRed, 5, Qt.SolidLine))
        painter.setBrush(QBrush(Qt.red, Qt.SolidPattern))
        painter.drawEllipse(611.25, 131.25, 227.5, 227.5)

        painter.setPen(QPen(Qt.darkGray, 5, Qt.SolidLine))
        painter.setBrush(QBrush(Qt.gray, Qt.SolidPattern))
        painter.drawEllipse(611.25, 441.25, 227.5, 227.5)

class Green(QWidget):
    def __init__(self, parent):
        super(Green, self).__init__(parent)

    def paintEvent(self, e):
        painter = QPainter(self)
        painter.setPen(QPen(Qt.darkGray, 5, Qt.SolidLine))
        painter.setBrush(QBrush(Qt.gray, Qt.SolidPattern))
        painter.drawEllipse(611.25, 131.25, 227.5, 227.5)

        painter.setPen(QPen(Qt.darkGreen, 5, Qt.SolidLine))
        painter.setBrush(QBrush(Qt.green, Qt.SolidPattern))
        painter.drawEllipse(611.25, 441.25, 227.5, 227.5)
 
 
App = QApplication(sys.argv)
window = Main()
green = Green(window)

# signal handling
if signal == 1:
    # green
    window.setCentralWidget(green)
else:
    # red
    window.setCentralWidget(None)

sys.exit(App.exec())

