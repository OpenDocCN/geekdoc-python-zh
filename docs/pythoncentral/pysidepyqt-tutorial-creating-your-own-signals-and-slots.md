# PySide/PyQt 教程:创建自己的信号和插槽

> 原文：<https://www.pythoncentral.io/pysidepyqt-tutorial-creating-your-own-signals-and-slots/>

但是，您不必仅仅依赖 Qt 小部件提供的信号；你可以创造你自己的。信号是使用信号类创建的。一个简单的信号定义是:

*   [PySide](#custom-tab-0-pyside)
*   [PyQt](#custom-tab-0-pyqt)

*   [PySide](#)

[python]
from PySide.QtCore import Signal
tapped = Signal()
[/python]

*   [PyQt](#)

[python]
from PyQt4.QtCore import pyqtSignal
tapped = pyqtSignal()
[/python]

然后，当对象被点击的条件满足时，调用信号的`emit`方法，信号被发出，调用它所连接的任何插槽:

```py

thing.tapped.emit()

```

这有两个好处:首先，它允许你的对象的用户以熟悉的方式与它们交互；第二，它允许更灵活地使用对象，将对象上动作的定义效果留给使用它们的代码。

## 简单的 PySide/PyQt 信号发射示例

让我们定义一个简单的`PunchingBag`类，它只做一件事:当它的`punch`被调用时，它发出一个`punched`信号:

*   [PySide](#custom-tab-1-pyside)
*   [PyQt](#custom-tab-1-pyqt)

*   [PySide](#)

[python]
from PySide.QtCore import QObject, Signal, Slot

class PunchingBag(QObject):
' ' '代表一个出气筒；当你按下它时，它会发出一个信号，表明它被按下了。‘
拳打脚踢=信号()

def __init__(self):
#将 PunchingBag 初始化为 QObject
QObject。__init__(self)

def punch(self):
' ' '出拳包' ' ' '
self . pucked . emit()
[/python]

*   [PyQt](#)

[python]
from PyQt4.QtCore import QObject, pyqtSignal, pyqtSlot

class PunchingBag(QObject):
' ' '代表一个出气筒；当你按下它时，它会发出一个信号，表明它被按下了。‘
punched = pyqtSignal()

def __init__(self):
#将 PunchingBag 初始化为 QObject
QObject。__init__(self)

def punch(self):
' ' '出拳包' ' ' '
self . pucked . emit()
[/python]

你可以很容易地看到我们做了什么。`PunchingBag`继承了`QObject`所以它可以发出信号；它有一个名为`punched`的信号，不携带任何数据；它有一个`punch`方法，除了发出`punched`信号之外什么也不做。

为了让我们的`PunchingBag`有用，我们需要将它的`punched`信号连接到一个做一些事情的插槽。我们将定义一个简单的函数，它在控制台上打印“Bag was punched ”,实例化我们的`PunchingBag`,并将其`punched`信号连接到插槽:

*   [PySide](#custom-tab-2-pyside)
*   [PyQt](#custom-tab-2-pyqt)

*   [PySide](#)

[python]
@Slot()
def say_punched():
''' Give evidence that a bag was punched. '''
print('Bag was punched.')

bag = PunchingBag()
#将袋子的打孔信号连接到 say_punched 插槽
bag . punched . Connect(say _ punched)
[/python]

*   [PyQt](#)

[python]
@pyqtSlot()
def say_punched():
''' Give evidence that a bag was punched. '''
print('Bag was punched.')

bag = PunchingBag()
#将袋子的打孔信号连接到 say_punched 插槽
bag . punched . Connect(say _ punched)
[/python]

然后，我们打一下袋子，看看会发生什么:

```py

# Punch the bag 10 times

for i in range(10):

    bag.punch()

```

当您将其全部放入脚本并运行它时，它将打印:

```py

Bag was punched.

Bag was punched.

Bag was punched.

Bag was punched.

Bag was punched.

Bag was punched.

Bag was punched.

Bag was punched.

Bag was punched.

Bag was punched.

```

有效，但不是特别令人印象深刻。然而，你可以看到它的用处:我们的出气筒非常适合任何你需要对出气筒做出反应的地方，因为`PunchingBag`把对出气筒做出反应的实现留给了使用它的代码。

## 载有数据的 PySide/PyQt 信号

创建信号时，您可以做的最有趣的事情之一是让它们携带数据。例如，您可以让一个信号携带一个整数，因此:

*   [PySide](#custom-tab-3-pyside)
*   [PyQt](#custom-tab-3-pyqt)

*   [PySide](#)

[python]
updated = Signal(int)
[/python]

*   [PyQt](#)

[python]
updated = pyqtSignal(int)
[/python]

或者字符串:

*   [PySide](#custom-tab-4-pyside)
*   [PyQt](#custom-tab-4-pyqt)

*   [PySide](#)

[python]
updated = Signal(str)
[/python]

*   [PyQt](#)

[python]
updated = pyqtSignal(str)
[/python]

数据类型可以是任何 Python 类型名称或标识 C++数据类型的字符串。由于本教程假定没有 C++知识，我们将坚持使用 Python 类型。

### PySide/PyQt 信号发送圈

让我们用属性`x`、`y`和`r`定义一个圆，分别表示圆心的`x`和`y`位置，以及它的半径。您可能希望在调整圆大小时发出一个信号，在移动圆时发出另一个信号；我们将分别称它们为`resized`和`moved`。

有可能让与`resized`和`moved`信号连接的插槽检查圆的新位置或大小，并相应地作出响应，但是如果发送的信号可以包括该信息，则更方便，并且通过插槽函数需要较少的圆知识。

*   [PySide](#custom-tab-5-pyside)
*   [PyQt](#custom-tab-5-pyqt)

*   [PySide](#)

[python]
from PySide.QtCore import QObject, Signal, Slot

类 Circle(QObject):
' ' '表示一个由圆心的 x、y
坐标和半径 r 定义的圆，' ' '
#调整圆大小时发出的信号，
#携带其整数半径
调整大小=信号(int)
#移动圆时发出的信号，携带
#其圆心的 x、y 坐标。
moved = Signal(int，int)

def __init__(self，x，y，r):
#将圆初始化为 QObject，这样它就可以发出信号
QObject。__init__(self)

#“隐藏”这些值，并通过 properties
self 公开它们。_x = x
自我。_y = y
自我。_r = r

@property
def x(self):
返回 self。_x

@x.setter
def x(self，new_x):
self。_x = new_x
#中心移动后，用新坐标
self.moved.emit(new_x，self.y)发出
#移动信号

@property
def y(self):
返回 self。_y
@y.setter
def y(self，new_y):
self。_y = new_y
#中心移动后，用新坐标
self.moved.emit(self.x，new_y)发出移动后的
#信号

@property
def r(self):
返回 self。_r

@r.setter
def r(self，new_r):
self。_r = new_r
#半径改变后，发射新半径的
#调整大小信号
self . resized . emit(new _ r)
[/python]

*   [PyQt](#)

[python]
from PyQt4.QtCore import QObject, pyqtSignal, pyqtSlot

类 Circle(QObject):
' ' '表示一个由圆心的 x、y
坐标和半径 r 定义的圆，' ' '
#调整圆大小时发出的信号，
#携带其整数半径
调整大小= pyqtSignal(int)
#移动圆时发出的信号，携带
#其圆心的 x、y 坐标。
moved = pyqtSignal(int，int)

def __init__(self，x，y，r):
#将圆初始化为 QObject，这样它就可以发出信号
QObject。__init__(self)

#“隐藏”这些值，并通过 properties
self 公开它们。_x = x
自我。_y = y
自我。_r = r

@property
def x(self):
返回 self。_x

@x.setter
def x(self，new_x):
self。_x = new_x
#中心移动后，用新坐标
self.moved.emit(new_x，self.y)发出
#移动信号

@property
def y(self):
返回 self。_y
@y.setter
def y(self，new_y):
self。_y = new_y
#中心移动后，用新坐标
self.moved.emit(self.x，new_y)发出移动后的
#信号

@property
def r(self):
返回 self。_r

@r.setter
def r(self，new_r):
self。_r = new_r
#半径改变后，发射新半径的
#调整大小信号
self . resized . emit(new _ r)
[/python]

请注意以下要点:

*   `Circle`继承了`QObject`，所以它可以发出信号。
*   信号是用它们将要连接的插槽的签名创建的。
*   同一个信号可以在多个地方发出。

现在，让我们定义一些可以连接到圆圈信号的插槽。还记得上次我们说我们会看到更多关于`@Slot`装饰工的内容吗？我们现在有了携带数据的信号，所以我们将看看如何制作可以接收数据的插槽。要使一个插槽接受来自一个信号的数据，我们只需用与它的信号相同的签名来定义它:

*   [PySide](#custom-tab-6-pyside)
*   [PyQt](#custom-tab-6-pyqt)

*   [PySide](#)

[python]
# A slot for the "moved" signal, accepting the x and y coordinates
@Slot(int, int)
def on_moved(x, y):
print('Circle was moved to (%s, %s).' % (x, y))

#一个用于“调整大小”信号的槽，接受半径
@ Slot(int)
def on _ resized(r):
print(' Circle 被调整大小为半径% s . % r)
[/python]

*   [PyQt](#)

[python]
# A slot for the "moved" signal, accepting the x and y coordinates
@pyqtSlot(int, int)
def on_moved(x, y):
print('Circle was moved to (%s, %s).' % (x, y))

#一个用于“调整大小”信号的槽，接受半径
@ pyqtSlot(int)
def on _ resized(r):
print(' Circle 被调整大小为半径%s.' % r)
[/python]

非常简单直观。要了解更多关于 [Python decorators](https://www.pythoncentral.io/python-decorators-overview/ "Python Decorators Overview") 的信息，你可能想看看文章- [Python Decorators 概述](https://www.pythoncentral.io/python-decorators-overview/ "Python Decorators Overview")来熟悉一下自己。

最后，让我们实例化一个圆，将信号连接到插槽，并移动和调整它的大小:

```py

c = Circle(5, 5, 4)
#将圆圈的信号连接到我们的简单插槽
c . moved . Connect(on _ moved)
c . resized . Connect(on _ resized)
#将圆向右移动一个单位
 c.x += 1
#将圆的半径增加一个单位
 c.r += 1 

```

当您运行结果脚本时，您的输出应该是:

```py

Circle was moved to (6, 5).

Circle was resized to radius 5.

```

现在我们已经对信号和插槽有了更好的理解，我们准备使用一些更高级的小部件。在我们的下一期文章中，我们将开始讨论`QListWidget`和`QListView`，两种创建列表框控件的方法。