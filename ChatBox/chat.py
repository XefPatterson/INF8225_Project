#!/usr/bin/env python
# Source : Werkov, Github, PyQt4 examples
# URL : https://github.com/Werkov/PyQt4/tree/master/examples/dbus/chat


# This is only needed for Python v2 but is harmless for Python v3.
import sip

sip.setapi('QString', 2)

from PyQt4 import QtCore, QtGui, QtDBus
import time
import tensorflow as tf

import os
import sys

sys.path.append(os.path.join('.', 'GUI'))
sys.path.append(os.path.join('.', 'Model_ForChat'))
sys.path.append(os.path.join('..', 'Model'))
from ui_chatmainwindow import Ui_ChatMainWindow
from ui_chatsetnickname import Ui_NicknameDialog
from model_forChatBox import seq2seq_chat
import utils


class ChatAdaptor(QtDBus.QDBusAbstractAdaptor):
    QtCore.Q_CLASSINFO("D-Bus Interface", 'com.trolltech.chat')

    QtCore.Q_CLASSINFO("D-Bus Introspection", ''
                                              '  <interface name="com.trolltech.chat">\n'
                                              '    <signal name="message">\n'
                                              '      <arg direction="out" type="s" name="nickname"/>\n'
                                              '      <arg direction="out" type="s" name="text"/>\n'
                                              '    </signal>\n'
                                              '    <signal name="action">\n'
                                              '      <arg direction="out" type="s" name="nickname"/>\n'
                                              '      <arg direction="out" type="s" name="text"/>\n'
                                              '    </signal>\n'
                                              '  </interface>\n'
                                              '')

    action = QtCore.pyqtSignal(str, str)

    message = QtCore.pyqtSignal(str, str)

    def __init__(self, parent):
        super(ChatAdaptor, self).__init__(parent)

        self.setAutoRelaySignals(True)


class ChatInterface(QtDBus.QDBusAbstractInterface):
    action = QtCore.pyqtSignal(str, str)

    message = QtCore.pyqtSignal(str, str)

    def __init__(self, service, path, connection, parent=None):
        super(ChatInterface, self).__init__(service, path,
                                            'com.trolltech.chat', connection, parent)


class ChatMainWindow(QtGui.QMainWindow, Ui_ChatMainWindow):
    action = QtCore.pyqtSignal(str, str)

    message = QtCore.pyqtSignal(str, str)

    def __init__(self):
        super(ChatMainWindow, self).__init__()

        self.m_nickname = "nickname"
        self.m_messages = []

        self.setupUi(self)
        self.sendButton.setEnabled(False)

        # Connects
        self.messageLineEdit.textChanged.connect(self.textChangedSlot)
        self.sendButton.clicked.connect(self.sendClickedSlot)
        self.actionChangeNickname.triggered.connect(self.changeNickname)
        self.actionAboutQt.triggered.connect(self.aboutQt)
        QtGui.qApp.lastWindowClosed.connect(self.exiting)

        # Add our D-Bus interface and connect to D-Bus.
        ChatAdaptor(self)
        QtDBus.QDBusConnection.sessionBus().registerObject('/', self)

        iface = ChatInterface('', '', QtDBus.QDBusConnection.sessionBus(),
                              self)
        QtDBus.QDBusConnection.sessionBus().connect('', '',
                                                    'com.trolltech.chat', 'message', self.messageSlot)
        iface.action.connect(self.actionSlot)

        dialog = NicknameDialog()
        dialog.cancelButton.setVisible(False)
        dialog.exec_()
        self.m_nickname = dialog.nickname.text().strip()
        self.action.emit(self.m_nickname, "joins the chat")

        # --------------------------------
        # CHAT-BOT INITIALIZATION
        # --------------------------------
        # Instanciates and build the model for feedforward only
        self.seq2seq_bot = seq2seq_chat(buckets=[(50, 50)], forward_only=True)
        self.seq2seq_bot.build()

        # Restore the trained model's parameters from checkpoint file
        self.sess = tf.Session()
        saver, summary_writer = utils.restore(self.seq2seq_bot, self.sess,
                                              save_name=os.path.join('..', 'Model', 'model_saved'))

    def rebuildHistory(self):
        history = '\n'.join(self.m_messages)
        self.chatHistory.setPlainText(history)

    @QtCore.pyqtSlot(str, str)
    def messageSlot(self, nickname, text):
        # User's message
        self.m_messages.append("[%s] - %s" % (nickname.upper(), text))

        if len(self.m_messages) > 100:
            self.m_messages.pop(0)

        self.rebuildHistory()
        QtGui.QApplication.processEvents()

        # Bot's message
        """
            Computes the reply
        """
        theReply = self.seq2seq_bot.reply(text, self.sess)
        self.m_messages.append("[%s] - %s" % ("BOT", theReply))

        if len(self.m_messages) > 100:
            self.m_messages.pop(0)

        self.rebuildHistory()

    @QtCore.pyqtSlot(str, str)
    def actionSlot(self, nickname, text):
        self.m_messages.append("--------\n* %s %s\n--------" % (nickname, text))

        if len(self.m_messages) > 100:
            self.m_messages.pop(0)

        self.rebuildHistory()

    @QtCore.pyqtSlot(str)
    def textChangedSlot(self, newText):
        self.sendButton.setEnabled(newText != '')

    @QtCore.pyqtSlot()
    def sendClickedSlot(self):
        msg = QtDBus.QDBusMessage.createSignal('/', 'com.trolltech.chat',
                                               'message')
        msg << self.m_nickname << self.messageLineEdit.text()
        QtDBus.QDBusConnection.sessionBus().send(msg)
        self.messageLineEdit.setText('')

    @QtCore.pyqtSlot()
    def changeNickname(self):
        dialog = NicknameDialog(self)

        if dialog.exec_() == QtGui.QDialog.Accepted:
            old = self.m_nickname
            self.m_nickname = dialog.nickname.text().strip()
            self.action.emit(old, "is now known as %s" % self.m_nickname)

    @QtCore.pyqtSlot()
    def aboutQt(self):
        QtGui.QMessageBox.aboutQt(self)

    @QtCore.pyqtSlot()
    def exiting(self):
        self.action.emit(self.m_nickname, "leaves the chat")


class NicknameDialog(QtGui.QDialog, Ui_NicknameDialog):
    def __init__(self, parent=None):
        super(NicknameDialog, self).__init__(parent)

        self.setupUi(self)


if __name__ == '__main__':
    import sys

    app = QtGui.QApplication(sys.argv)

    if not QtDBus.QDBusConnection.sessionBus().isConnected():
        sys.stderr.write("Cannot connect to the D-Bus session bus.\n"
                         "Please check your system settings and try again.\n")
        sys.exit(1)

    chat = ChatMainWindow()
    chat.show()

    sys.exit(app.exec_())
