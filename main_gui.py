import sys
# import os
# import cv2
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMessageBox, QGraphicsPixmapItem, QGraphicsScene, QStatusBar
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.uic import loadUiType
from PyQt5.QtCore import QStringListModel
from symmetry_eval import *

MainWindowForm, MainWindowBase = loadUiType('GUI/gui.ui')


def w_h_resize(img_path, w_re, h_re):
    img_path = str(img_path)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[0:2]
    if h < w:
        ratio = float(w / h)
        h_re = int(w_re / ratio)
        img = cv2.resize(img, (w_re, h_re))
    else:
        ratio = float(h / w)
        w_re = int(h_re / ratio)
        img = cv2.resize(img, (w_re, h_re))
    return img, w_re, h_re


class MainWindow(MainWindowBase, MainWindowForm):
    def __init__(self):
        # Initialization
        super(MainWindow, self).__init__()
        self.setupUi(self)
        self.setWindowTitle('Facial Symmetry Evaluation')
        self.paras = parameters()
        self.yolo_dir, self.crop_dir, self.san_dir, self.results_path = '', '', '', ''
        self.landmarks, self.landmarks_normal = [], []
        self.SI_Eyebrows, self.SI_Eyes, self.SI_Mouth = 0, 0, 0
        # StatusBar initialization
        self.statusBar = QStatusBar()
        self.statusBar.setStyleSheet('QStatusBar::item {border: none;}')
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage('Individual Project: Face symmetry evaluation')
        # Set events for buttons
        self.display_page1()
        self.pushButton_p11.clicked.connect(self.open_file)  # Right Widget, Page1, 'Select File'
        self.pushButton_p12.clicked.connect(self.detect_faces)  # Right Widget, Page1, 'Detext Face(s)'
        self.pushButton_p13.clicked.connect(self.display_page2)  # Right Widget, Page1, 'Next'
        self.pushButton_p21.clicked.connect(self.display_page1)  # Right Widget, Page2, 'Previous'
        self.pushButton_p22.clicked.connect(self.display_page3)  # Right Widget, Page2, 'Start Evaluation'
        self.pushButton_p31.clicked.connect(self.display_page2)  # Right Widget, Page3, 'Previous'
        self.pushButton_p32.clicked.connect(self.display_page1)  # Right Widget, Page3, 'Retry'
        self.pushButton_Re.clicked.connect(self.page3_graphic_change)
        self.pushButton_BH.clicked.connect(self.page3_graphic_change)
        self.pushButton_EH.clicked.connect(self.page3_graphic_change)
        self.pushButton_EW.clicked.connect(self.page3_graphic_change)
        self.pushButton_EA.clicked.connect(self.page3_graphic_change)
        self.pushButton_MC.clicked.connect(self.page3_graphic_change)
        self.pushButton_MA.clicked.connect(self.page3_graphic_change)
        self.pushButton_MTA.clicked.connect(self.page3_graphic_change)
        self.listView.clicked.connect(self.face_list)  # Right Widget, Page2, list

    def display_page1(self):
        self.stackedWidget.setCurrentIndex(0)
        self.label_1.setStyleSheet("background-color: rgb(255, 85, 0)")
        self.label_2.setStyleSheet("background-color: rgb(100, 30, 0)")
        self.label_3.setStyleSheet("background-color: rgb(100, 30, 0)")

    def display_page2(self):
        if self.stackedWidget.currentIndex() == 0:
            if self.yolo_dir == '':
                QMessageBox.warning(self, 'Warning', 'Please select a picture and perform face detection!')
            else:
                self.stackedWidget.setCurrentIndex(1)
                self.label_1.setStyleSheet("background-color: rgb(100, 30, 0)")
                self.label_2.setStyleSheet("background-color: rgb(255, 85, 0)")
                self.label_3.setStyleSheet("background-color: rgb(100, 30, 0)")
                self.yolo_dir = get_file(self.yolo_dir)
                slm = QStringListModel()
                slm.setStringList(self.yolo_dir)
                self.listView.setModel(slm)
                labeltext = f'Select 1 from {num_detect} face(s)'
                self.label_p21.setText(labeltext)
        else:
            self.stackedWidget.setCurrentIndex(1)
            self.label_1.setStyleSheet("background-color: rgb(100, 30, 0)")
            self.label_2.setStyleSheet("background-color: rgb(255, 85, 0)")
            self.label_3.setStyleSheet("background-color: rgb(100, 30, 0)")

    def display_page3(self):
        self.stackedWidget.setCurrentIndex(2)
        self.label_1.setStyleSheet("background-color: rgb(100, 30, 0)")
        self.label_2.setStyleSheet("background-color: rgb(100, 30, 0)")
        self.label_3.setStyleSheet("background-color: rgb(255, 85, 0)")
        Symmetry_Index = self.detect_landmarks()
        self.lineEdit_BH_L.setText(f'{Symmetry_Index[0]} px')
        self.lineEdit_BH_R.setText(f'{Symmetry_Index[1]} px')
        self.lineEdit_BH_D.setText(f'{Symmetry_Index[2]} px')
        self.lineEdit_EH_L.setText(f'{Symmetry_Index[3]} px')
        self.lineEdit_EH_R.setText(f'{Symmetry_Index[4]} px')
        self.lineEdit_EH_D.setText(f'{Symmetry_Index[5]} px')
        self.lineEdit_EW_L.setText(f'{Symmetry_Index[6]} px')
        self.lineEdit_EW_R.setText(f'{Symmetry_Index[7]} px')
        self.lineEdit_EW_D.setText(f'{Symmetry_Index[8]} px')
        self.lineEdit_EA_L.setText(f'{Symmetry_Index[9]} px²')
        self.lineEdit_EA_R.setText(f'{Symmetry_Index[10]} px²')
        self.lineEdit_EA_D.setText(f'{Symmetry_Index[11]} px²')
        self.lineEdit_MC_L.setText(f'{Symmetry_Index[12]} px')
        self.lineEdit_MC_R.setText(f'{Symmetry_Index[13]} px')
        self.lineEdit_MC_D.setText(f'{Symmetry_Index[14]} px')
        self.lineEdit_MA_L.setText(f'{Symmetry_Index[15]} px²')
        self.lineEdit_MA_R.setText(f'{Symmetry_Index[16]} px²')
        self.lineEdit_MA_D.setText(f'{Symmetry_Index[17]} px²')
        self.lineEdit_MTA_D.setText(f'{Symmetry_Index[18]} °')
        self.lineEdit_Eva_EB.setText(f'{Symmetry_Index[19]} %')
        self.lineEdit_Eva_E.setText(f'{Symmetry_Index[20]} %')
        self.lineEdit_Eva_M.setText(f'{Symmetry_Index[21]} %')
        labelText = 'The face seems SYMMETRICAL !'
        if Symmetry_Index[19] < 90 or Symmetry_Index[20] < 90 or Symmetry_Index[21] < 90:
            labelText = 'The face seems ASYMMETRICAL ! '
        self.label_p31.setText(labelText)
        self.page3_graphic(self.results_path)

    def open_file(self):
        fileName, fileType = QtWidgets.QFileDialog.getOpenFileName(self, "Select File", os.getcwd(),
                                                                   "Image Files(*.jpg *.jpeg)")
        self.lineEdit.setText(fileName)
        self.paras.image = self.lineEdit.text()
        self.page1_graphic(self.paras.image)

    def page1_graphic(self, img_path):
        img, w_re, h_re = w_h_resize(img_path, 655, 545)
        frame = QImage(img, w_re, h_re, w_re * 3, QImage.Format_RGB888)
        pix = QPixmap.fromImage(frame)
        item = QGraphicsPixmapItem(pix)
        scene = QGraphicsScene()
        scene.addItem(item)
        self.graphicsView_p1.setScene(scene)

    def page2_graphic(self, img_path):
        img, w_re, h_re = w_h_resize(img_path, 495, 495)
        frame = QImage(img, w_re, h_re, w_re * 3, QImage.Format_RGB888)
        pix = QPixmap.fromImage(frame)
        item = QGraphicsPixmapItem(pix)
        scene = QGraphicsScene()
        scene.addItem(item)
        self.graphicsView_p2.setScene(scene)

    def page3_graphic(self, img_path):
        sender = self.sender()
        clickEvent = sender.text()
        img, w_re, h_re = w_h_resize(img_path, 390, 390)
        frame = QImage(img, w_re, h_re, w_re * 3, QImage.Format_RGB888)
        pix = QPixmap.fromImage(frame)
        item = QGraphicsPixmapItem(pix)
        scene = QGraphicsScene()
        scene.addItem(item)
        self.graphicsView_p3.setScene(scene)

    def page3_graphic_change(self):
        sender = self.sender()
        clickEvent = sender.text()
        if clickEvent == u'Brows Height:':
            img_path = self.results_path.split('.')[0] + '_Eyebrows.jpg'
        elif clickEvent == u'Eyes Height:':
            img_path = self.results_path.split('.')[0] + '_EyeHeight.jpg'
        elif clickEvent == u'Eyes Width:':
            img_path = self.results_path.split('.')[0] + '_EyeWidth.jpg'
        elif clickEvent == u'Eyes Area:':
            img_path = self.results_path.split('.')[0] + '_EyeArea.jpg'
        elif clickEvent == u'Mouth Corner:':
            img_path = self.results_path.split('.')[0] + '_MouthCorner.jpg'
        elif clickEvent == u'Mouth Area:':
            img_path = self.results_path.split('.')[0] + '_MouthArea.jpg'
        elif clickEvent == u'Mouth Angle:':
            img_path = self.results_path.split('.')[0] + '_MouthAngle.jpg'
        else:
            img_path = self.results_path
        img, w_re, h_re = w_h_resize(img_path, 390, 390)
        frame = QImage(img, w_re, h_re, w_re * 3, QImage.Format_RGB888)
        pix = QPixmap.fromImage(frame)
        item = QGraphicsPixmapItem(pix)
        scene = QGraphicsScene()
        scene.addItem(item)
        self.graphicsView_p3.setScene(scene)

    def face_list(self, qModelIndex):
        self.page2_graphic(self.yolo_dir[qModelIndex.row()])
        self.crop_dir = self.yolo_dir[qModelIndex.row()]

    def detect_faces(self):
        global num_detect
        if not os.path.isfile(self.lineEdit.text()):
            QMessageBox.warning(self, 'Warning', 'Please select an image file')
        else:
            self.paras.image = self.lineEdit.text()
            self.paras.image = self.paras.image.replace('\\', '/').replace('\\\\', '/').replace('//', '/')
            results = yolo_detect(self.paras)
            if len(results) == 2:
                self.yolo_dir, num_detect = results[:]
            else:
                num_detect = 0
                QMessageBox.warning(self, 'Warning', 'There is no face detected, try other images please!')
            img_path = str(self.yolo_dir) + '/' + self.paras.image.split('/')[-1]
            face = 'faces' if num_detect > 1 else 'face'
            labeltext = f'{num_detect} {face} detected!'
            self.label_p11.setText(labeltext)
            self.page1_graphic(img_path)

    def detect_landmarks(self):
        img = cv2.imread(self.crop_dir)  # read face image
        h, w = img.shape[0:2]
        h_re, w_re = int(h * 720 / h), int(w * 720 / h)  # x=>640, scale y correspondingly
        img_resize = cv2.resize(img, dsize=(w_re, h_re), fx=1, fy=1,
                                interpolation=cv2.INTER_LINEAR)  # scale image to x=640
        cv2.imwrite(self.crop_dir, img_resize)  # save scaled image for SAN
        self.san_dir, self.landmarks = SAN_detect(self.crop_dir, self.paras)
        self.results_path, self.landmarks_normal = landmark_normalize(self.san_dir, self.landmarks)
        SI_data = SI_calculate(self.results_path, self.landmarks_normal)
        return SI_data

    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'Exit', 'Are you sure to quit?', QMessageBox.Yes | QMessageBox.Cancel)
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
