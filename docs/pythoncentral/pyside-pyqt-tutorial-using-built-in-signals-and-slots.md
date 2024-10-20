# PySide/PyQt 教程:使用内置信号和插槽

> 原文：<https://www.pythoncentral.io/pyside-pyqt-tutorial-using-built-in-signals-and-slots/>

在上一期中，我们学习了如何创建和设置交互式小部件，以及如何使用两种不同的方法将它们排列成简单和复杂的布局。今天，我们将讨论允许您的应用程序响应用户触发事件的 Python/Qt 方式:信号和插槽。

当用户采取一个动作时——点击一个按钮，在组合框中选择一个值，在文本框中输入——这个小部件发出一个*信号*。这个信号本身什么也不做；它必须连接到一个*插槽*，这个插槽是一个充当信号接收者的对象，如果给定一个信号，它就对其进行操作。

## 连接内置 PySide/PyQt 信号

Qt 部件内置了许多信号。例如，当一个`QPushButton`被点击时，它发出它的`clicked`信号。`clicked`信号可以连接到充当插槽的功能(仅摘录；运行它需要更多的代码):

*   [PySide](#custom-tab-0-pyside)
*   [PyQt](#custom-tab-0-pyqt)

*   [PySide](#)

[python]
@Slot()
def clicked_slot():
''' This is called when the button is clicked. '''
print('Ouch!')

#创建按钮
BTN = q button(' Sample ')

#将其点击的信号连接到我们的槽位
BTN . clicked . Connect(clicked _ slot)
[/python]

*   [PyQt](#)

[python]
@pyqtSlot()
def clicked_slot():
''' This is called when the button is clicked. '''
print('Ouch!')

#创建按钮
BTN = q button(' Sample ')

#将其点击的信号连接到我们的槽位
BTN . clicked . Connect(clicked _ slot)
[/python]

注意在`clicked_slot`的定义上面使用了`@Slot()`装饰符；虽然不是绝对必要的，但它提供了关于如何调用`clicked_slot`的 C++ Qt 库提示。(关于 decorator 的更多信息，请参见[Python decorator 概述文章](https://www.pythoncentral.io/python-decorators-overview/ "Python Decorators Overview")。)稍后我们会看到更多关于`@Slot`宏的信息。目前来说，知道按钮被点击时会发出`clicked`信号，这个信号会调用它所连接的函数；有着少年般的幽默感，它会发出‘哎哟！’。

对于一个不那么幼稚(并且实际上是可执行的)的例子，让我们看看一个`QPushButton`如何发出它的三个相关信号，`pressed`、`released`和`clicked`。

*   [PySide](#custom-tab-1-pyside)
*   [PyQt](#custom-tab-1-pyqt)

*   [PySide](#)

[python]
import sys
from PySide.QtCore import Slot
from PySide.QtGui import *

# ...在此插入其余的导入内容
#导入内容必须在所有其他内容之前...

#创建一个 Qt app 和一个窗口
app = QApplication(sys.argv)

win = q widget()
win . setwindowtitle('测试窗口')

#在窗口中创建一个按钮
btn = QPushButton('Test '，win)

@Slot()
def on_click():
' ' '告知按钮何时被点击'
打印('点击')

@Slot()
def on_press():
' ' '告知按钮何时被按下‘
印刷(‘按下’)

@Slot()
def on_release():
' ' '告知按钮何时释放‘
刊印(‘公布’)

#将信号连接到插槽
BTN . clicked . Connect(on _ click)
BTN . pressed . Connect(on _ press)
BTN . released . Connect(on _ release)

#显示窗口并运行 app
win . Show()
app . exec _()
[/python]

*   [PyQt](#)

[python]
import sys
from PyQt4.QtCore import pyqtSlot
from PyQt4.QtGui import *

# ...在此插入其余的导入内容
#导入内容必须在所有其他内容之前...

#创建一个 Qt app 和一个窗口
app = QApplication(sys.argv)

win = q widget()
win . setwindowtitle('测试窗口')

#在窗口中创建一个按钮
btn = QPushButton('Test '，win)

@ pyqtSlot()
def on _ click():
' ' '告知按钮何时被点击'
打印('点击')

@ pyqtSlot()
def on _ press():
' ' '告知按钮何时按下‘
印刷(‘按下’)

@ pyqtSlot()
def on _ release():
' ' '告知按钮何时释放‘
刊印(‘公布’)

#将信号连接到插槽
BTN . clicked . connect(on _ click)
BTN . pressed . connect(on _ press)
BTN . released . connect(on _ release)

#显示窗口并运行 app
win . Show()
app . exec _()
[/python]

当您运行应用程序并单击按钮时，它将打印:

```py

pressed

released

clicked

```

按钮按下时发出`pressed`信号，松开时发出`released`信号，最后，当这两个动作都完成时，发出`clicked`信号。

### **完成我们的示例应用程序**

现在，很容易完成上一期文章中的示例应用程序。我们将向`LayoutExample`类添加一个 slot 方法，该方法将显示构建的问候语:

*   [PySide](#custom-tab-2-pyside)
*   [PyQt](#custom-tab-2-pyqt)

*   [PySide](#)

[python]
@Slot()
def show_greeting(self):
self.greeting.setText('%s, %s!' %
(self.salutations[self.salutation.currentIndex()],
self.recipient.text()))
[/python]

*   [PyQt](#)

[python]
@pyqtSlot()
def show_greeting(self):
self.greeting.setText('%s, %s!' %
(self.salutations[self.salutation.currentIndex()],
self.recipient.text()))
[/python]

注意，我们使用`recipient` `QLineEdit`的`text()`方法来检索用户在那里输入的文本，使用`salutation` `QComboBox`的`currentIndex()`方法来获取用户选择的称呼的索引。我们还使用了`Slot()`装饰符来表示`show_greeting`将被用作一个槽。

然后，我们可以简单地将构建按钮的`clicked`信号连接到该方法:

```py

self.build_button.clicked.connect(self.show_greeting)

```

我们最后的例子，总的来说，看起来是这样的:

*   [PySide](#custom-tab-3-pyside)
*   [PyQt](#custom-tab-3-pyqt)

*   [PySide](#)

[python]
import sys
from PySide.QtCore import Slot
from PySide.QtGui import *

#每个 Qt 应用程序必须有且只有一个 Qt application 对象；
#它接收传递给脚本的命令行参数，因为它们
#可用于定制应用程序的外观和行为
Qt _ app = QA application(sys . argv)

class layout example(q widget):
' ' ' py side 绝对定位的例子；主窗口
继承了 QWidget，这是一个方便的空窗口小部件。''

def __init__(self):
#将对象初始化为 QWidget，
#设置其标题和最小宽度
QWidget。_ _ init _ _(self)
self . setwindowtitle('动态迎宾')
self . setminimumwwidth(400)

#创建布局整个表单的 QVBoxLayout
self . layout = QVBoxLayout()

#创建管理带标签控件的表单布局
self . form _ layout = QFormLayout()

自我.问候=[‘嗨’、
、【嗨】、
、【hey】、
、【Hi】、
、【wassup】、
、【yo】

#创建并填充组合框以选择称呼
self.salutation = QComboBox(self)
self . Salutation . additems(self . salutations)
#将其添加到带有标签
self . form _ layout . addrow(&Salutation:'，self . Salutation)的表单布局中

#创建条目控件以指定一个
#收件人并设置其占位符文本
self . recipient = qline edit(self)
self . recipient . setplaceholdertext("例如' world '或' Matey ' ")

#将其添加到带有标签
self . form _ layout . addrow(&Recipient:'，self.recipient)的表单布局中

#创建并添加标签以显示问候语文本
self.greeting = QLabel('，self)
self . form _ layout . addrow(' Greeting:'，self.greeting)

#将表单布局添加到主 VBox 布局
self . layout . Add layout(self . form _ layout)

# Add stretch 将表单布局与按钮
self.layout.addStretch(1)分开

#创建一个水平的框布局来放置按钮
self.button_box = QHBoxLayout()

# Add stretch 将按钮推到最右边
self . button _ box . Add stretch(1)

#创建标题为
self . Build _ button = q push button(&Build Greeting，self)的构建按钮

#将按钮的点击信号连接到 show _ greeting
self . build _ button . clicked . Connect(self . show _ greeting)

#将其添加到按钮框
self . button _ box . Add widget(self . build _ button)

#将按钮框添加到主 VBox 布局的底部

#将 VBox 布局设置为窗口的主布局
self.setLayout(self.layout)

@ Slot()
def Show _ greeting(self):
' ' '显示构造的问候语' '
self.greeting.setText('%s，%s！'%
(self . salutations[self . salutation . current index()]，
self.recipient.text()))

def run(self):
#显示表单
self.show()
#运行 qt 应用程序
qt_app.exec_()

#创建应用程序窗口的实例并运行它
app = layout example()
app . run()
[/python]

*   [PyQt](#)

[python]
import sys
from PyQt4.QtCore import pyqtSlot
from PyQt4.QtGui import *

#每个 Qt 应用程序必须有且只有一个 Qt application 对象；
#它接收传递给脚本的命令行参数，因为它们
#可用于定制应用程序的外观和行为
Qt _ app = QA application(sys . argv)

class layout example(q widget):
' ' ' py side 绝对定位的例子；主窗口
继承了 QWidget，这是一个方便的空窗口小部件。''

def __init__(self):
#将对象初始化为 QWidget，
#设置其标题和最小宽度。
QWidget。_ _ init _ _(self)
self . setwindowtitle('动态迎宾')
self . setminimumwwidth(400)

#创建布局整个表单的 QVBoxLayout
self . layout = QVBoxLayout()

#创建管理带标签控件的表单布局
self . form _ layout = QFormLayout()

自我.问候=[‘嗨’、
、【嗨】、
、【hey】、
、【Hi】、
、【wassup】、
、【yo】

#创建并填充组合框以选择称呼
self.salutation = QComboBox(self)
self . Salutation . additems(self . salutations)
#将其添加到带有标签
self . form _ layout . addrow(&Salutation:'，self . Salutation)的表单布局中

#创建条目控件以指定一个
#收件人并设置其占位符文本
self . recipient = qline edit(self)
self . recipient . setplaceholdertext("例如' world '或' Matey ' ")

#将其添加到带有标签
self . form _ layout . addrow(&Recipient:'，self.recipient)的表单布局中

#创建并添加标签以显示问候语文本
self.greeting = QLabel('，self)
self . form _ layout . addrow(' Greeting:'，self.greeting)

# Cdd 窗体布局到主 VBox 布局
self . layout . add layout(self . form _ layout)

# Add stretch 将表单布局与按钮
self.layout.addStretch(1)分开

#创建一个水平的框布局来放置按钮
self.button_box = QHBoxLayout()

# Add stretch 将按钮推到最右边
self . button _ box . Add stretch(1)

#创建标题为
self . Build _ button = q push button(&Build Greeting，self)的构建按钮

#将按钮的点击信号连接到 show _ greeting
self . build _ button . clicked . Connect(self . show _ greeting)

#将其添加到按钮框
self . button _ box . Add widget(self . build _ button)

#将按钮框添加到主 VBox 布局的底部
self . layout . Add layout(self . button _ box)

#将 VBox 布局设置为窗口的主布局
self.setLayout(self.layout)

@ pyqtSlot()
def Show _ greeting(self):
' ' '显示构造的问候语' '
self.greeting.setText('%s，%s！'%
(self . salutations[self . salutation . current index()]，
self.recipient.text()))

def run(self):
#显示表单
self.show()
#运行 qt 应用程序
qt_app.exec_()

#创建应用程序窗口的实例并运行它
app = layout example()
app . run()
[/python]

运行它，你将得到和以前一样的窗口，除了现在当 Build 按钮被按下时，它实际上生成我们的问候。(注意，可以将相同的方法添加到我们上次的绝对定位示例中，效果相同。)

既然我们已经知道如何将内置信号连接到我们创建的插槽，我们就为下一部分做好了准备，在下一部分中，我们将学习如何创建我们自己的信号并将它们连接到插槽。