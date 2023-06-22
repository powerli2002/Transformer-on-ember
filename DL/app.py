# from ui.Ui_main import Ui_MainWindow

# from PySide2 import QtWidgets
# # import qtawesome as qta
# from PySide2.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
# from PySide2.QtCharts import QtCharts
# from PySide2.QtCore import QStringListModel
# from Ensemble_app import evaluate_detection

# class UI_mainwindow(QtWidgets.QMainWindow):
#     def __init__(self, parent=None):
#         super(UI_mainwindow, self).__init__(parent)
#         self.ui = Ui_MainWindow()
#         self.ui.setupUi(self)

#         self.init_slots()  # 定义控件绑定相关操作输入
  
#         self.dirpath = ""


#     def init_slots(self):
#         self.ui.pushButton_kill.clicked.connect(self.kill_dir)
#         self.ui.pushButton_scan.clicked.connect(self.scan_dir)

#     def kill_dir(self):


#         pass

#     def scan_dir(self):
#         self.dirpath = self.ui.lineEdit_dir.text()

#         folder_path = "/path/to/folder"
#         results, statistics = evaluate_detection(folder_path)
#         file_names = [result["file_name"] for result in results]
#         DL_outputs = [result["DL_output"] for result in results]
#         ember_scores = [result["ember_score"] for result in results]
#         ensemble_answers = [result["ensemble_ans"] for result in results]

#         total = statistics["total"]
#         DL_malicious = statistics["DL_malicious"]
#         DL_detection_rate = statistics["DL_detection_rate"]
#         ember_malicious = statistics["ember_malicious"]
#         ember_detection_rate = statistics["ember_detection_rate"]
#         ensemble_malicious = statistics["ensemble_malicious"]
#         ensemble_detection_rate = statistics["ensemble_detection_rate"]


#         self.stringlistmodel = QStringListModel()  # 创建stringlistmodel对象
#         self.ui.listView.setModel(self.stringlistmodel)  # 把view和model关联
#         # self.stringlistmodel.dataChanged.connect(self.save)  # 存储所有行的数据

#         chart = QtCharts.QChart()
#         series = QtCharts.QBarSeries()

#         # Add data to the bar chart
#         data = [
#             ("DL Malicious", statistics["DL_malicious"]),
#             ("Ember Malicious", statistics["ember_malicious"]),
#             ("Ensemble Malicious", statistics["ensemble_malicious"])
#         ]

#         for label, value in data:
#             bar_set = QtCharts.QBarSet(label)
#             bar_set.append(value)
#             series.append(bar_set)

#         chart.addSeries(series)

#         # Customize the chart appearance
#         chart.setTitle("Detection Results")
#         chart.setAnimationOptions(QtCharts.QChart.SeriesAnimations)
#         chart.createDefaultAxes()
#         chart.setDropShadowEnabled(True)

#         # Create a chart view
#         chart_view = QtCharts.QChartView(chart, self.ui.frame_chart)

#         # Set the chart view as the layout's widget
#         layout = QtWidgets.QVBoxLayout(self.ui.frame_chart)
#         layout.addWidget(chart_view)

#         # Display statistics in QLabel
#         self.ui.label_statistics.setText(
#             f"Total: {statistics['total']}\n"
#             f"DL Malicious: {statistics['DL_malicious']}\n"
#             f"DL Detection Rate: {statistics['DL_detection_rate']}\n"
#             f"Ember Malicious: {statistics['ember_malicious']}\n"
#             f"Ember Detection Rate: {statistics['ember_detection_rate']}\n"
#             f"Ensemble Malicious: {statistics['ensemble_malicious']}\n"
#             f"Ensemble Detection Rate: {statistics['ensemble_detection_rate']}"
#         )

# app = QtWidgets.QApplication([])
# window = UI_mainwindow()
# window.show()
# app.exec_()




from ui.Ui_main import Ui_MainWindow
from PySide2 import QtWidgets
from PySide2.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox,QTableWidgetItem
from PySide2.QtCharts import QtCharts
from PySide2.QtCore import QStringListModel
from Ensemble_app import evaluate_detection
import os

class UI_mainwindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(UI_mainwindow, self).__init__(parent)
        self.ui = Ui_MainWindow()
        
        self.ui.setupUi(self)
        self.init_column_widths()
        # self.init_column_widths()
        

        self.init_slots()  # Define control-related operations
        self.dirpath = ""

        

        


    def init_column_widths(self):
        self.ui.tableWidget.setColumnCount(4)
        self.ui.tableWidget.setColumnWidth(0, 300)  # Set the width of column 0 (File Name) to 150 pixels
        self.ui.tableWidget.setColumnWidth(1, 200)  # Set the width of column 1 (Ember Detection Result) to 150 pixels
        self.ui.tableWidget.setColumnWidth(2, 200)  # Set the width of column 2 (Transformer Detection Result) to 150 pixels
        self.ui.tableWidget.setColumnWidth(3, 200)  # Set the width of column 3 (Ensemble Detection Result) to 150 pixels
        headers = ["File Name", "Ember", "Transformer", "Ensemble"]
        self.ui.tableWidget.setHorizontalHeaderLabels(headers)

    def init_slots(self):
        self.ui.pushButton_kill.clicked.connect(self.kill_dir)
        self.ui.pushButton_scan.clicked.connect(self.scan_dir)
        

    def kill_dir(self):
        rows_to_remove = []
        for row in range(self.ui.tableWidget.rowCount()):
            ensemble_malicious = self.ui.tableWidget.item(row, 3).text()
            if ensemble_malicious == "dangerous":
                file_name = self.ui.tableWidget.item(row, 0).text()
                rows_to_remove.append(row)
                os.remove(file_name)
                # Delete the file or perform any other desired action here based on the file_name

        # Remove the selected rows from the tableWidget
        for row in reversed(rows_to_remove):
            self.ui.tableWidget.removeRow(row)

    def scan_dir(self):
        self.dirpath = self.ui.lineEdit_dir.text()

        results, statistics = evaluate_detection(self.dirpath)

        # Set up the column headers for the QTableWidget
        
        

        # Populate the QTableWidget with the detection results
        self.ui.tableWidget.setRowCount(len(results))
        for row, result in enumerate(results):
            file_name = result["file_name"]
            ember_malicious = "dangerous" if result["ember_score"]==1 else "good"
            dl_malicious = "dangerous" if result["DL_ans"]==1 else "good"
            ensemble_malicious = "dangerous" if result["ensemble_ans"]==1 else "good"

            # Set the data in the QTableWidgetItem for each column
            self.ui.tableWidget.setItem(row, 0, QTableWidgetItem(file_name))
            self.ui.tableWidget.setItem(row, 1, QTableWidgetItem(ember_malicious))
            self.ui.tableWidget.setItem(row, 2, QTableWidgetItem(dl_malicious))
            self.ui.tableWidget.setItem(row, 3, QTableWidgetItem(ensemble_malicious))



    

app = QtWidgets.QApplication([])
window = UI_mainwindow()
window.show()
app.exec_()
