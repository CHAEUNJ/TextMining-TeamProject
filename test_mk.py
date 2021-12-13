import functools
from PIL import Image

from PyQt5 import QtCore, QtGui, QtWidgets
import test_rc
import threading

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        self.rank = ["도치피자", 157, 4.5, 4.8, 4.8, 4.9, 4.8]
        self.keyword = ["피자", "맛집", "파스타", "화덕", "치즈", "기타", "등등"]

        MainWindow.setWindowTitle("TextMining TeamProject")
        MainWindow.resize(360, 640)
        MainWindow.setStyleSheet("color: rgb(229, 229, 229);")

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setGeometry(QtCore.QRect(0, 0, 361, 81))
        self.frame.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)

        self.widget = QtWidgets.QWidget(self.frame)
        self.widget.setGeometry(QtCore.QRect(10, 0, 341, 80))
        self.widget.setStyleSheet("border-image: url(:/newPrefix/gui_image/search_img.png);")

        self.widget_2 = QtWidgets.QWidget(self.widget)
        self.widget_2.setGeometry(QtCore.QRect(30, 23, 31, 31))
        self.widget_2.setStyleSheet("border-image: url(:/newPrefix/gui_image/search_icon.png);")

        self.lineEdit = QtWidgets.QLineEdit(self.widget)
        self.lineEdit.setGeometry(QtCore.QRect(70, 18, 231, 41))
        self.lineEdit.setStyleSheet("border-image: url(:/newPrefix/gui_image/search_window.png);\n"
"color: rgb(39, 174, 96);\n""font: 18pt;")
        self.lineEdit.returnPressed.connect(self.enterInputFunction)

        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(10, 95, 56, 12))
        self.label.setStyleSheet("color: rgb(75, 75, 75);")
        self.label.setText("맛집 랭킹")
        self.label.hide()

        self.pbar = QtWidgets.QProgressBar(self.centralwidget)
        self.pbar.setGeometry(QtCore.QRect(55, 250, 250, 30))

        self.lbl_pbar = QtWidgets.QLabel(self.centralwidget)
        self.lbl_pbar.setGeometry(QtCore.QRect(100, 280, 180, 30))
        self.lbl_pbar.setStyleSheet("font: 13pt;\n""color: rgb(68, 68, 68);")
        self.lbl_pbar.setText("맛집 랭킹을 가져오고 있어요!")
        self.lbl_pbar.hide()

        self.scrollArea = QtWidgets.QScrollArea(self.centralwidget)
        self.scrollArea.setGeometry(QtCore.QRect(0, 120, 361, 471))
        self.scrollArea.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustToContents)
        self.scrollArea.setWidgetResizable(True)

        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 346, 469))

        self.restaurant_list_layout = QtWidgets.QVBoxLayout()

        self.widget_list = []
        self.label_list = []
        self.score_list = []

        for idx in range(10):
                self.widget_list.append(QtWidgets.QWidget(self.scrollAreaWidgetContents))
                self.widget_list[idx].setGeometry(QtCore.QRect(10, 10 + (130 * idx), 321, 111))
                self.widget_list[idx].setFixedHeight(111)
                self.widget_list[idx].setFixedWidth(321)
                self.widget_list[idx].setStyleSheet("border-image: url(:/newPrefix/gui_image/list_bg.png);")

                self.lbl_ranking = QtWidgets.QLabel(self.widget_list[idx])
                self.lbl_ranking.setGeometry(QtCore.QRect(15, 40, 31, 31))
                self.lbl_ranking.setStyleSheet("font: 20pt;\n""color: rgb(169, 169, 169);")
                self.lbl_ranking.setText(str(idx+1))

                self.lbl_taste = QtWidgets.QLabel(self.widget_list[idx])
                self.lbl_taste.setGeometry(QtCore.QRect(50, 70, 21, 16))
                self.lbl_taste.setStyleSheet("font: 12pt;\n""color: rgb(68, 68, 68);")
                self.lbl_taste.setText("맛")

                self.lbl_service = QtWidgets.QLabel(self.widget_list[idx])
                self.lbl_service.setGeometry(QtCore.QRect(130, 70, 41, 16))
                self.lbl_service.setStyleSheet("font: 12pt;\n""color: rgb(68, 68, 68);")
                self.lbl_service.setText("서비스")

                self.lbl_vibes = QtWidgets.QLabel(self.widget_list[idx])
                self.lbl_vibes.setGeometry(QtCore.QRect(230, 70, 41, 16))
                self.lbl_vibes.setStyleSheet("font: 12pt;\n""color: rgb(68, 68, 68);")
                self.lbl_vibes.setText("분위기")

                self.lbl_price = QtWidgets.QLabel(self.widget_list[idx])
                self.lbl_price.setGeometry(QtCore.QRect(50, 90, 31, 16))
                self.lbl_price.setStyleSheet("font: 12pt;\n""color: rgb(68, 68, 68);")
                self.lbl_price.setText("가격")

                self.lbl_visit_again = QtWidgets.QLabel(self.widget_list[idx])
                self.lbl_visit_again.setGeometry(QtCore.QRect(130, 90, 41, 16))
                self.lbl_visit_again.setStyleSheet("font: 12pt;\n""color: rgb(68, 68, 68);")
                self.lbl_visit_again.setText("재방문")

                self.lbl_star_1 = QtWidgets.QLabel(self.widget_list[idx])
                self.lbl_star_1.setGeometry(QtCore.QRect(65, 70, 16, 16))
                self.lbl_star_1.setText(
                        "<html><head/><body><p><img src=\":/newPrefix/gui_image/star_icon.png\"/></p></body></html>")

                self.lbl_star_2 = QtWidgets.QLabel(self.widget_list[idx])
                self.lbl_star_2.setGeometry(QtCore.QRect(165, 70, 16, 16))
                self.lbl_star_2.setText(
                        "<html><head/><body><p><img src=\":/newPrefix/gui_image/star_icon.png\"/></p></body></html>")

                self.lbl_star_3 = QtWidgets.QLabel(self.widget_list[idx])
                self.lbl_star_3.setGeometry(QtCore.QRect(265, 70, 16, 16))
                self.lbl_star_3.setText(
                        "<html><head/><body><p><img src=\":/newPrefix/gui_image/star_icon.png\"/></p></body></html>")

                self.lbl_star_4 = QtWidgets.QLabel(self.widget_list[idx])
                self.lbl_star_4.setGeometry(QtCore.QRect(75, 90, 16, 16))
                self.lbl_star_4.setText(
                        "<html><head/><body><p><img src=\":/newPrefix/gui_image/star_icon.png\"/></p></body></html>")

                self.lbl_star_5 = QtWidgets.QLabel(self.widget_list[idx])
                self.lbl_star_5.setGeometry(QtCore.QRect(165, 90, 16, 16))
                self.lbl_star_5.setText(
                        "<html><head/><body><p><img src=\":/newPrefix/gui_image/star_icon.png\"/></p></body></html>")
                # 음식점 이름
                self.label_list.append([QtWidgets.QLabel(self.widget_list[idx]),
                                        QtWidgets.QLabel(self.widget_list[idx]), QtWidgets.QLabel(self.widget_list[idx])])
                self.label_list[idx][0].setGeometry(QtCore.QRect(50, 10, 111, 31))
                self.label_list[idx][0].setStyleSheet("font: 17pt;\n""color: rgb(5, 118, 23);")
                # 블로그 리뷰 수
                self.label_list[idx][1].setGeometry(QtCore.QRect(160, 18, 131, 16))
                self.label_list[idx][1].setStyleSheet("font: 14pt;\n""color: rgb(169, 169, 169);")
                # 키워드
                self.label_list[idx][2].setGeometry(QtCore.QRect(50, 45, 251, 16))
                self.label_list[idx][2].setStyleSheet("font: 11pt;\n""color: rgb(39, 174, 96);\n"
                                                      "border-image: url(:/newPrefix/gui_image/highlight.png);")
                self.label_list[idx][2].mousePressEvent = functools.partial(self.showWordCloud, self.label_list[idx][2])
                self.label_list[idx][2].setObjectName(str(idx))

                # 맛 점수
                self.score_list.append([QtWidgets.QLabel(self.widget_list[idx]), QtWidgets.QLabel(self.widget_list[idx]),
                                       QtWidgets.QLabel(self.widget_list[idx]), QtWidgets.QLabel(self.widget_list[idx]),
                                       QtWidgets.QLabel(self.widget_list[idx])])
                self.score_list[idx][0].setGeometry(QtCore.QRect(80, 70, 21, 16))
                self.score_list[idx][0].setStyleSheet("font: 12pt;\n""color: rgb(68, 68, 68);")
                # 서비스 점수
                self.score_list[idx][1].setGeometry(QtCore.QRect(180, 70, 21, 16))
                self.score_list[idx][1].setStyleSheet("font: 12pt;\n""color: rgb(68, 68, 68);")
                # 분위기 점수
                self.score_list[idx][2].setGeometry(QtCore.QRect(280, 70, 21, 16))
                self.score_list[idx][2].setStyleSheet("font: 12pt;\n""color: rgb(68, 68, 68);")
                # 음식 점수
                self.score_list[idx][3].setGeometry(QtCore.QRect(90, 90, 21, 16))
                self.score_list[idx][3].setStyleSheet("font: 12pt;\n""color: rgb(68, 68, 68);")
                # 재방문 점수
                self.score_list[idx][4].setGeometry(QtCore.QRect(180, 90, 21, 16))
                self.score_list[idx][4].setStyleSheet("font: 12pt;\n""color: rgb(68, 68, 68);")

        for idx in range(10):
                self.restaurant_list_layout.addWidget(self.widget_list[idx])

        self.scrollAreaWidgetContents.setLayout(self.restaurant_list_layout)

        self.scrollArea.setWidget(self.scrollAreaWidgetContents)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 358, 22))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        MainWindow.setStatusBar(self.statusbar)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.scrollArea.hide()

        self.step = 0

        self.timer = QtCore.QTimer()

    def timerEvent(self):
        if self.step >= 100:
            self.timer.stop()

            self.lbl_pbar.hide()

            # label update
            self.label_list[0][0].setText(self.rank[0])
            self.label_list[0][1].setText("블로그 리뷰 " + str(self.rank[1]))
            self.label_list[0][2].setText((" " + " ".join(self.keyword[:5])).replace(" ", " #"))
            self.label_list[0][2].setFixedWidth(len(self.label_list[0][2].text()) * 8)
            self.score_list[0][0].setText(str(self.rank[2]))
            self.score_list[0][1].setText(str(self.rank[3]))
            self.score_list[0][2].setText(str(self.rank[4]))
            self.score_list[0][3].setText(str(self.rank[5]))
            self.score_list[0][4].setText(str(self.rank[6]))

            self.label.show()
            self.scrollArea.show()

            self.step = 0
            return

        self.step = self.step + 1
        self.pbar.setValue(self.step)

    def enterInputFunction(self):
        self.scrollArea.hide()

        location, food = self.lineEdit.text().split()
        print(location, food)

        t = threading.Thread(target=code_main, args=(location, food))
        t.start()

        self.timer.setInterval(25)
        self.timer.timeout.connect(self.timerEvent)
        self.timer.start()
        self.lbl_pbar.show()

    def showWordCloud(self, key_word_idx, event):

        image = Image.open(self.label_list[int(key_word_idx.objectName())][0].text() + ".jpeg")
        image.show()

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.setFont(QtGui.QFont("Adobe Heiti Std"))
    MainWindow.show()
    sys.exit(app.exec_())
