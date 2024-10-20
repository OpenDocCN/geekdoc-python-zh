# PySide/PyQt 简介:基本部件和 Hello，World！

> 原文：<https://www.pythoncentral.io/intro-to-pysidepyqt-basic-widgets-and-hello-world/>

这一部分介绍了 PySide 和 PyQt 的最基本的要点。我们将谈一谈它们使用的对象种类，并通过几个非常简单的例子向您介绍 Python/Qt 应用程序是如何构造的。

首先，Qt 对象的基本概述。Qt 提供了许多类来处理各种各样的事情:XML、多媒体、数据库集成、网络等等，但是我们现在最关心的是可见的元素——窗口、对话框和控件。Qt 的所有可见元素都被称为小部件，是一个公共父类`QWidget`的后代。在本教程中，我们将使用“widget”作为 Qt 应用程序中任何可见元素的通称。

Qt widgets are themable. They look more-or-less native on Windows and most Linux setups, though Mac OS X theming is still a work in progress; right now, Python/Qt applications on Mac OS X look like they do on Linux. You can also specify custom themes to give your application a unique look and feel.

## 第一个 Python/Qt 应用程序:Hello，World

我们将从一个非常简单的应用程序开始，它显示一个带有标签的窗口，标签上写着“Hello，world！”这是以一种容易掌握的风格编写的，但不可模仿——我们稍后会解决这个问题。

*   [PySide](#custom-tab-0-pyside)
*   [PyQt](#custom-tab-0-pyqt)

*   [PySide](#)

```py

# Allow access to command-line arguments

import sys
#从 PySide 导入 Qt 
的核心和 GUI 元素。来自 PySide 的 QtCore import * 
。QtGui 导入*
#每个 Qt 应用程序必须有且只有一个 Qt application 对象；
 #它接收传递给脚本的命令行参数，因为它们
 #可用于定制应用程序的外观和行为
Qt _ app = QA application(sys . argv)
#用我们的文本
 label = QLabel('Hello，world！')
#将其显示为独立的小部件
 label.show()
#运行应用程序的事件循环
 qt_app.exec_() 

```

*   [PyQt](#)

```py

# Allow access to command-line arguments

import sys
# SIP 允许我们选择希望使用的 API
导入 SIP
#使用更现代的 PyQt API(Python 2 . x 中默认不启用)；
 #必须在导入任何提供指定 API 的模块之前
 sip.setapi('QDate '，2) 
 sip.setapi('QDateTime '，2) 
 sip.setapi('QString '，2) 
 sip.setapi('QTextStream '，2) 
 sip.setapi('QTime '，2) 
 sip.setapi('QUrl '，2) 
 sip.setapi('QVariant '，2)
#从 PyQt4 导入所有 Qt 
。Qt 导入*
#每个 Qt 应用程序必须有且只有一个 Qt application 对象；
 #它接收传递给脚本的命令行参数，因为它们
 #可用于定制应用程序的外观和行为
Qt _ app = QA application(sys . argv)
#用我们的文本
 label = QLabel('Hello，world！')
#将其显示为独立的小部件
 label.show()
#运行应用程序的事件循环
 qt_app.exec_() 

```

对我们所做工作的高度概括:

*   创建 Qt 应用程序
*   创建小部件
*   将其显示为一个窗口
*   运行应用程序的事件循环

这是任何 Qt 应用程序的基本轮廓。无论打开多少个窗口，每个应用程序都必须有且只有一个`QApplication`对象，它初始化应用程序，处理控制流、事件调度和应用程序级设置，并在应用程序关闭时进行清理。

创建的小部件没有父部件，这意味着它显示为一个窗口；这是应用程序的启动窗口。如图所示，然后调用`QApplication`对象的`exec_`方法，这将启动应用程序的主事件循环。

关于这个例子的一些细节:

1.  注意，`QApplication`的构造函数接收`sys.argv`作为参数；这允许用户使用命令行参数来控制 Python/Qt 应用程序的外观、感觉和行为。
2.  我们的主要小部件是一个`QLabel`，它仅仅显示文本；任何小部件——也就是继承自`QWidget`的任何东西——都可以显示为一个窗口。3.

A note on the PyQt version: there's a fair amount of boilerplate code preceding the creation of the `QApplication` object. That selects the API 2 version of each object's behavior instead of the obsolescent API 1, which is the default for Python 2.x. In the future, our examples for both PySide and PyQt will omit the `import` section for space and clarity. But don't forget that it needs to be there. (Actually, all of the `sip` lines could have been omitted from this example without any effect, as could the `PySide.QtCore` import, as it doesn't use any of those objects directly; I've included them as an example for the future.)

# 两个基本部件

让我们来看两个最基本的 Python/Qt 小部件。首先，我们将回顾它们所有的父代，`QWidget`；然后，我们将看一个继承自它的最简单的小部件。

## QWidget

QWidget 的构造函数有两个参数，`parent QWidget`和`flags QWindowFlags`，这两个参数由它的所有后代共享。小部件的父部件拥有该小部件，当父部件被销毁时，子部件在其父部件被销毁时被销毁，并且其几何形状通常受到其父部件的几何形状的限制。如果父控件是`None`或者没有提供父控件，则小部件归应用程序的`QApplication`对象所有，并且是一个顶级小部件，即一个窗口。如果窗口显示，参数`flags`控制部件的各种属性；通常，默认值 0 是正确的选择。

通常，您会像这样构造一个`QWidget`:

```py

widget = QWidget()

```

或者

```py

widget = QWidget(some_parent)

```

一个`QWidget`经常被用来创建一个顶层窗口，因此:

```py

qt_app = QApplication(sys.argv)
#创建一个小部件
 widget = QWidget()
#将其显示为独立的 widget 
 widget.show()
#运行应用程序的事件循环
 qt_app.exec_() 

```

`QWidget`类有许多方法，但大多数方法在另一个小部件的上下文中讨论更有用。然而，我们很快就会用到的一个方法是`setMinimumSize`方法，它接受一个`QtCore.QSize`作为它的参数；一个`QSize`代表一个小部件的二维(宽×高)像素度量。

```py

widget.setMinimumSize(QSize(800, 600))

```

另外一个可以被所有 widgets 使用的`QWidget`方法是`setWindowTitle`；如果小部件显示为顶层窗口，这将设置其标题:

```py

widget.setWindowTitle('I Am A Window!')

```

## QLabel

我们已经在“你好，世界！”中使用了一个`QLabel`应用程序，但我们将仔细研究它。它主要用于显示纯文本或富文本、静态图像或视频，通常是非交互式的。

它有两个类似的构造函数，一个与`QWidget`相同，另一个采用一个`text` [unicode](https://www.pythoncentral.io/python-unicode-encode-decode-strings-python-2x/ "Encoding and Decoding Strings (in Python 2.x)") 字符串来指定显示的文本:

```py

label = QLabel(parent_widget)

```

或者

```py

label = QLabel('Hello, world!', parent_widget)

```

默认情况下，标签的内容靠左对齐，但是可以使用`QLabel`的`setAlignment`方法将其更改为任意的`PySide.QtCore.Qt.Alignment`，如下所示:

```py

label.setAlignment(Qt.AlignCenter)

```

您也可以使用`QLabel`的`setIndent`方法设置缩进；缩进是从内容对齐的一侧以像素为单位指定的；例如，如果对准是`Qt.AlignRight`，缩进将从右边开始。

要在`QLabel`中换行，请使用`QLabel.setWordWrap(True)`；称之为“T2”式的文字转换。

A `QLabel`有更多的方法，但这是一些最基本的。

# 更高级的“你好，世界！”

既然我们已经研究了`QWidget`类及其后代`QLabel`，我们可以对我们的“Hello，world！”更能说明 Python/Qt 编程的应用程序。

上次我们只是为小部件创建了全局变量，我们将把窗口的定义封装在一个继承自`QLabel`的新类中。在这种情况下，这似乎有点枯燥，但是我们将在后面的例子中扩展这个概念。

```py

# Remember that we're omitting the import

# section from our examples for brevity
#创建 QA application 对象
Qt _ app = QA application(sys . argv)
class HelloWorldApp(QLabel): 
' ' '一个显示文本“Hello，world！”' '
 def __init__(self): 
 #将对象初始化为 QLabel 
 QLabel。__init__(self，“你好，世界！”)
#设置大小、对齐方式和标题
self . Set minimumsize(QSize(600，400)) 
 self.setAlignment(Qt。
 self.setWindowTitle('你好，世界！')
def run(self): 
' ' '显示应用程序窗口并启动主事件循环' ' ' '
 self.show() 
 qt_app.exec_()
#创建应用程序的实例并运行它
 HelloWorldApp()。run() 

```

到目前为止，我们已经准备好在下一期中进行一些真正的内容，在下一期中，我们将讨论更多的小部件、布局容器的基础以及信号和插槽，它们是 Qt 允许应用程序响应用户动作的方式。