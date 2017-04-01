# -*- coding: utf-8 -*-
# Source : Werkov, Github, PyQt4 examples
# URL : https://github.com/Werkov/PyQt4/tree/master/examples/dbus/chat

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    _fromUtf8 = lambda s: s


class Ui_ChatMainWindow(object):
    def setupUi(self, ChatMainWindow):
        # Main Window
        ChatMainWindow.setObjectName(_fromUtf8("ChatMainWindow"))
        ChatMainWindow.resize(800, 600)
        ChatMainWindow.setWindowIcon(QtGui.QIcon('GUI/ChatBot.png'))
        ChatMainWindow.setWindowTitle(QtGui.QApplication.translate(
            "ChatMainWindow", "Racoon's ChatBot", None, QtGui.QApplication.UnicodeUTF8))

        self.centralwidget = QtGui.QWidget(ChatMainWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))

        ChatMainWindow.setCentralWidget(self.centralwidget)

        # Chat History Box
        self.chatHistory = QtGui.QTextBrowser(self.centralwidget)
        self.chatHistory.setAcceptDrops(False)
        self.chatHistory.setToolTip(QtGui.QApplication.translate("ChatMainWindow", "Messages sent and received", None,
                                                                 QtGui.QApplication.UnicodeUTF8))
        self.chatHistory.setAcceptRichText(True)
        self.chatHistory.setObjectName(_fromUtf8("chatHistory"))

        # Message Line Edit
        self.messageLineEdit = QtGui.QLineEdit(self.centralwidget)
        self.messageLineEdit.setObjectName(_fromUtf8("messageLineEdit"))

        # Message Label
        self.label = QtGui.QLabel(self.centralwidget)
        self.label.setText(
            QtGui.QApplication.translate("ChatMainWindow", "Message:", None, QtGui.QApplication.UnicodeUTF8))
        self.label.setObjectName(_fromUtf8("label"))
        self.label.setBuddy(self.messageLineEdit)

        # Send button
        self.sendButton = QtGui.QPushButton(self.centralwidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Policy(1), QtGui.QSizePolicy.Policy(0))
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.sendButton.sizePolicy().hasHeightForWidth())
        self.sendButton.setSizePolicy(sizePolicy)
        self.sendButton.setToolTip(
            QtGui.QApplication.translate("ChatMainWindow", "Sends a message", None, QtGui.QApplication.UnicodeUTF8))
        self.sendButton.setWhatsThis(_fromUtf8(""))
        self.sendButton.setText(
            QtGui.QApplication.translate("ChatMainWindow", "Send", None, QtGui.QApplication.UnicodeUTF8))
        self.sendButton.setObjectName(_fromUtf8("sendButton"))

        # Menu bar
        self.menubar = QtGui.QMenuBar(ChatMainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 31))
        self.menubar.setObjectName(_fromUtf8("menubar"))

        self.menuHelp = QtGui.QMenu(self.menubar)
        self.menuHelp.setTitle(
            QtGui.QApplication.translate("ChatMainWindow", "Help", None, QtGui.QApplication.UnicodeUTF8))
        self.menuHelp.setObjectName(_fromUtf8("menuHelp"))

        self.menuFile = QtGui.QMenu(self.menubar)
        self.menuFile.setTitle(
            QtGui.QApplication.translate("ChatMainWindow", "File", None, QtGui.QApplication.UnicodeUTF8))
        self.menuFile.setObjectName(_fromUtf8("menuFile"))

        ChatMainWindow.setMenuBar(self.menubar)

        # Actions
        self.actionQuit = QtGui.QAction(ChatMainWindow)
        self.actionQuit.setText(
            QtGui.QApplication.translate("ChatMainWindow", "Quit", None, QtGui.QApplication.UnicodeUTF8))
        self.actionQuit.setShortcut(
            QtGui.QApplication.translate("ChatMainWindow", "Ctrl+Q", None, QtGui.QApplication.UnicodeUTF8))
        self.actionQuit.setObjectName(_fromUtf8("actionQuit"))

        self.actionAboutQt = QtGui.QAction(ChatMainWindow)
        self.actionAboutQt.setText(
            QtGui.QApplication.translate("ChatMainWindow", "About Qt...", None, QtGui.QApplication.UnicodeUTF8))
        self.actionAboutQt.setObjectName(_fromUtf8("actionAboutQt"))

        self.actionChangeNickname = QtGui.QAction(ChatMainWindow)
        self.actionChangeNickname.setText(
            QtGui.QApplication.translate("ChatMainWindow", "Change nickname...", None, QtGui.QApplication.UnicodeUTF8))
        self.actionChangeNickname.setShortcut(
            QtGui.QApplication.translate("ChatMainWindow", "Ctrl+N", None, QtGui.QApplication.UnicodeUTF8))
        self.actionChangeNickname.setObjectName(_fromUtf8("actionChangeNickname"))

        self.menuHelp.addAction(self.actionAboutQt)
        self.menuFile.addAction(self.actionChangeNickname)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionQuit)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuHelp.menuAction())

        # Status bar
        self.statusbar = QtGui.QStatusBar(ChatMainWindow)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))

        ChatMainWindow.setStatusBar(self.statusbar)

        # VboxLayout
        self.vboxlayout = QtGui.QVBoxLayout()
        self.vboxlayout.setMargin(0)
        self.vboxlayout.setSpacing(6)
        self.vboxlayout.setObjectName(_fromUtf8("vboxlayout"))

        self.vboxlayout.addWidget(self.chatHistory)

        # HboxLayout
        self.hboxlayout1 = QtGui.QHBoxLayout()
        self.hboxlayout1.setMargin(0)
        self.hboxlayout1.setSpacing(6)
        self.hboxlayout1.setObjectName(_fromUtf8("hboxlayout1"))

        self.hboxlayout1.addWidget(self.label)
        self.hboxlayout1.addWidget(self.messageLineEdit)
        self.hboxlayout1.addWidget(self.sendButton)

        # Main Layout
        self.hboxlayout = QtGui.QHBoxLayout(self.centralwidget)
        self.hboxlayout.setMargin(9)
        self.hboxlayout.setSpacing(6)
        self.hboxlayout.setObjectName(_fromUtf8("hboxlayout"))

        self.vboxlayout.addLayout(self.hboxlayout1)
        self.hboxlayout.addLayout(self.vboxlayout)

        # Connects
        QtCore.QObject.connect(self.messageLineEdit, QtCore.SIGNAL(_fromUtf8("returnPressed()")),
                               self.sendButton.animateClick)
        QtCore.QObject.connect(self.actionQuit, QtCore.SIGNAL(_fromUtf8("triggered(bool)")), ChatMainWindow.close)
        QtCore.QMetaObject.connectSlotsByName(ChatMainWindow)

        # Tab Order
        ChatMainWindow.setTabOrder(self.chatHistory, self.messageLineEdit)
        ChatMainWindow.setTabOrder(self.messageLineEdit, self.sendButton)
