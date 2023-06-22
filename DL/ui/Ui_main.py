# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '/home/lizijian/lincode/ember/DL/ui/main.ui',
# licensing of '/home/lizijian/lincode/ember/DL/ui/main.ui' applies.
#
# Created: Thu Jun 22 00:59:52 2023
#      by: pyside2-uic  running on PySide2 5.13.0
#
# WARNING! All changes made in this file will be lost!

from PySide2 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1050, 773)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.pushButton_scan = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_scan.setGeometry(QtCore.QRect(350, 220, 91, 41))
        self.pushButton_scan.setObjectName("pushButton_scan")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(330, 50, 55, 18))
        self.label.setText("")
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(270, 50, 501, 51))
        font = QtGui.QFont()
        font.setFamily("Yrsa Medium")
        font.setPointSize(18)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.lineEdit_dir = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_dir.setGeometry(QtCore.QRect(290, 150, 471, 26))
        self.lineEdit_dir.setObjectName("lineEdit_dir")
        self.pushButton_kill = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_kill.setGeometry(QtCore.QRect(560, 220, 91, 41))
        self.pushButton_kill.setObjectName("pushButton_kill")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(110, 140, 151, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.tableWidget = QtWidgets.QTableWidget(self.centralwidget)
        self.tableWidget.setGeometry(QtCore.QRect(60, 310, 911, 381))
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setColumnCount(0)
        self.tableWidget.setRowCount(0)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1050, 23))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QtWidgets.QApplication.translate("MainWindow", "MainWindow", None, -1))
        self.pushButton_scan.setText(QtWidgets.QApplication.translate("MainWindow", "检测 ", None, -1))
        self.label_2.setText(QtWidgets.QApplication.translate("MainWindow", "基于ember数据集的融合模型恶意软件查杀系统", None, -1))
        self.pushButton_kill.setText(QtWidgets.QApplication.translate("MainWindow", "查杀", None, -1))
        self.label_3.setText(QtWidgets.QApplication.translate("MainWindow", " 请输入要扫描的文件夹：", None, -1))

