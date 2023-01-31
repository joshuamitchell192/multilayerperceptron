import sys
from PyQt6 import QtCore, QtGui, QtWidgets, uic
from PyQt6.QtCore import Qt

from neural_network import MultilayerPerceptron

class Canvas(QtWidgets.QLabel):

    def __init__(self):
        super().__init__()
        # self.label = QtWidgets.QLabel()
        pixmap = QtGui.QPixmap(28, 28).scaled(28*15, 28*15)
        pixmap.fill(Qt.GlobalColor.black)
        self.setPixmap(pixmap)

        self.last_x, self.last_y = None, None
        # self.pen_color = QtGui.QColor('#000000')

    def set_pen_color(self, c):
        self.pen_color = QtGui.QColor(c)

    def mouseMoveEvent(self, e):
        if self.last_x is None: # First event.
            self.last_x = e.position().x()
            self.last_y = e.position().y()
            return # Ignore the first time.

        canvas = self.pixmap()
        painter = QtGui.QPainter(canvas)
        p = QtGui.QPen()
        p.setWidth(15)
        p.setColor(QtGui.QColor('white'))
        painter.setPen(p)
        # point1 = 
        painter.drawLine(int(self.last_x), int(self.last_y), int(e.position().x()), int(e.position().y()))
        painter.end()
        self.setPixmap(canvas)

        image = QtGui.QImage(canvas.toImage())
        image.scaled(28, 28)#.toImageFormat(QtGui.QImage.Format.Format_Grayscale8)
        image.save('./newImage.png', "PNG")

        # Update the origin for next time.
        self.last_x = e.position().x()
        self.last_y = e.position().y()

    def mouseReleaseEvent(self, e):
        self.last_x = None
        self.last_y = None


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()

        self.canvas = Canvas()
        self.count = 0

        w = QtWidgets.QWidget()
        l = QtWidgets.QVBoxLayout()
        w.setLayout(l)
        l.addWidget(self.canvas)

        control_buttons_layout = QtWidgets.QHBoxLayout()
        self.add_buttons(control_buttons_layout)
        l.addLayout(control_buttons_layout)

        self.setCentralWidget(w)

    def add_buttons(self, layout):
        # for c in COLORS:
        save_button = QButton("Save")
        clear_button = QButton("Clear")
        # b.pressed.connect(lambda c=c: self.canvas.set_pen_color(c))

        save_button.pressed.connect(self.save)
        layout.addWidget(save_button)
        layout.addWidget(clear_button)

    def save(self):
        self.count = self.count + 1
        image = QtGui.QImage(self.canvas.pixmap().toImage()).scaledToHeight(28)
        image.save(f"./digit_{self.count}.png", "PNG")

        self.clear()

    def clear(self):
        pixmap = QtGui.QPixmap(28, 28).scaled(28*15, 28*15)
        pixmap.fill(Qt.GlobalColor.black)
        self.canvas.setPixmap(pixmap)

class QButton(QtWidgets.QPushButton):
    def __init__(self, text):
        super().__init__()
        self.setFixedSize(QtCore.QSize(45,30))
        self.setText(text)
        # self.color = color
        # self.setStyleSheet("background-color: %s;" % color)



if __name__ == "__main__":

    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()

    window.show()
    app.exec()