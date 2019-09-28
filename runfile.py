from PyQt5.QtWidgets import*
from PyQt5.QtPrintSupport import *
from PyQt5 import QtCore, QtGui, uic
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import time
import cv2
import os
import sys
from PIL import Image 
import threading
import inceptiontest

ui_MainWindow = uic.loadUiType("mainwindow.ui")[0]

class Fruitdetection(ui_MainWindow ,QMainWindow):
    
    def __init__(self, *args, **kwargs):
        
        ui_MainWindow.__init__(self, *args, **kwargs)
        QMainWindow.__init__(self)
        
        self.path = None
        self.inceppath= None
        self.calory = None
        self.category =None
        self.title = 'Fruit detection'
        self.left = 10
        self.top = 10
        self.width = 600
        self.height = 500
        
        self.status = QStatusBar()
        self.setStatusBar(self.status)
    
        self.setupUi(self)
        self.setinit()
        
    def setinit(self):
        
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        
        self.startButton.clicked.connect(self.gotomainpro)
        # status bar config
        file_toolbar = QToolBar("File")
        file_toolbar.setIconSize(QSize(14, 14))
        self.addToolBar(file_toolbar)
        file_menu = self.menuBar().addMenu("&File")

        open_file_action = QAction("Open...", self)
        open_file_action.setStatusTip("Open file")
        open_file_action.triggered.connect(self.file_open)
        file_menu.addAction(open_file_action)
        file_toolbar.addAction(open_file_action)

        saveas_file_action = QAction("Save...", self)
        saveas_file_action.setStatusTip("Save current page to specified file")
        saveas_file_action.triggered.connect(self.file_save)
        file_menu.addAction(saveas_file_action)
        file_toolbar.addAction(saveas_file_action)
        
        self.show()
    
    
    def gotomainpro(self):
        
        
        if not self.path :
            self.s= "Oh no!\n Please choose your fruit :)"
            self.dialog_critical(self.s)
                 
        else:
            self.mprogram(self.path)
    
            if self.inceppath :
                self.outputimage(self.inceppath)
            else:
                return
            
            
            if self.calory:
                self.setcalory(self.calory)
            else:
                return             
    
    def outputimage(self,mnpath):
        if mnpath:
            try:
                image1 = QImage(mnpath)
                if image1.isNull():
                    self.dialog_critical(str("Cannot load"))
                    return

                self.label_2.setPixmap(QPixmap.fromImage(image1))
                  
            except Exception as e:
                self.dialog_critical(str(e))
                                             
        else:
            return
        
    def setcalory(self,ce):
        if ce:
            self.textEdit.setText(str(ce))
        if self.category:
            self.textEdit_2.setText("Fruit category :\n {}".format(str(self.category)))
        else:
            return    
            
    def dialog_critical(self, s):
    
        dlg = QMessageBox(self)
        dlg.setText(s)
        dlg.setIcon(QMessageBox.Critical)
        dlg.show()
        
    def file_open(self):
        self.textEdit_2.setText("Processing ...")
        options = QFileDialog.Options()
        self.path, _ = QFileDialog.getOpenFileName(self, "Open file", "", 
                                              'Images (*.png *.jpeg *.jpg *.bmp *.gif *.JPG)',options=options)
        
        if self.path:
            try: 
                
                image = QImage(self.path)
                if image.isNull():
                    self.dialog_critical(str("Cannot load"))
                    return
                image1 = Image.open(self.path).convert("RGB")
                h,w=image1.size
                print(h,w)
                if h >= 120 or w >= 120:
                    image1 = image1.resize((120,120))
                    image1.save("MY_new_resize.jpg")
                    self.path = "MY_new_resize.jpg"
                    image = QImage(self.path)
                 
                self.label.setPixmap(QPixmap.fromImage(image))
                #self.gotomainpro()
                
            except Exception as e:
                self.dialog_critical(str(e))
                                             
        else:
            self.dialog_critical(str("select image"))
            #self.path = path
            
            return
        

    def file_save(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "Images (*.png *.jpeg *.jpg *.bmp *.gif *.JPG)")

        if not self.path:
            # If dialog is cancelled, will return ''
            return
    
    
    def mprogram(self,path):
        if path :
            self.calory,self.inceppath,self.category = inceptiontest.inceptionfunc(path) 
            
        else:
            self.dialog_critical(str("We have problem to run inception.py"))
                

                
if __name__ == '__main__':

    app = QApplication(sys.argv)
    window = Fruitdetection()
    app.exec_()             