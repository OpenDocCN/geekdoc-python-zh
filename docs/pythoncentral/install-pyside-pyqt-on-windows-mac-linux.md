# 在 Windows、Mac 和 Linux 上安装 PySide 和 PyQt

> 原文：<https://www.pythoncentral.io/install-pyside-pyqt-on-windows-mac-linux/>

[在上一篇文章](https://www.pythoncentral.io/intro-to-pysidepyqt-basic-widgets-and-hello-world/ "Introduction to PyQt and PySide")中，我向大家介绍了 Qt 及其 Python 接口，PyQt 和 PySide 现在你已经对它们有所了解，选择一个并安装它。我推荐 PySide 有两个原因:首先，本教程是根据 PySide 构思的，可能会涵盖一些在 PyQt 中没有完全实现的主题；第二，它的许可对你将来的使用更加灵活。然而，这两种方法都可以。

下面将向您展示如何在 Windows、Mac 和 Linux 上安装 PySide 和 PyQt。二进制安装程序可用于大多数常见平台；链接和设置说明概述如下:

*   [窗户](#windows)
*   麦克·OS X
*   [Linux(基于 Ubuntu 和 Debian)](#debian)
*   [Linux(基于 CentOS 和 RPM)](#rpm)

## **视窗**

PySide 或 PyQt 的安装是通过一个简单的点击式安装程序来完成的。对于 PySide，从[releases.qtproject.com](https://wiki.qt.io/PySide_Binaries_Windows "PySide Windows binaries")获得适合你的 Python 版本的二进制文件。运行安装程序，确认 Python 安装的位置(应该可以正确地自动检测到)并选择一个安装目录，您应该在几秒钟内完成 PySide 安装。

PyQt 非常相似，除了您只能选择部分安装而不是完整安装:不要。你会想要例子和演示。它们值这个空间。从[河岸](https://riverbankcomputing.com/software/pyqt/download "PyQt Windows binaries")获取 PyQt 安装程序。

## **麦克 OS X**

安装 PySide 的 Mac OS X 二进制文件可以从 [Qt 项目](https://wiki.qt.io/PySide_Binaries_MacOSX "Qt Project")中获得。

对于 PyQt，使用由 [PyQtX 项目](http://sourceforge.net/projects/pyqtx/files/Complete/ "PyQtX project")提供的二进制文件。为您的 Python 版本选择完整版本，它提供 Qt 和 PyQt，除非您确定您已经在正确的版本中安装了 Qt；然后使用最少的安装程序。

如果你正在使用[自制软件](https://brew.sh/ "Homebrew")，你可以:

```py
 brew install pyside 
```

或者

```py
 brew install pyqt 
```

从命令行。您也可以使用 MacPorts:

```py
 port-install pyNN-pyside 
```

更改`NN`以匹配您的 Python 版本。同样，您可以:

```py
 port-install pyNN-pyqt4 
```

安装 PyQt。

## **Linux(基于 Ubuntu 和 Debian)**

对于基于 Debian 和 Ubuntu 的 Linux 发行版，安装 PySide 或 PyQt 很简单；只是做:

```py
 sudo apt-get install python-pyside 
```

从命令行。对于 PyQt:

```py
 sudo apt-get install python-qt4 
```

或者，使用 Synaptic 安装您选择的`python-pyside`或`python-qt4`。

## **Linux(基于 CentOS 和 RPM)**

对于大多数基于 RPM 的发行版，使用`yum`安装 PySide 或 PyQt 也很简单；只是做:

```py
 yum install python-pyside pyside-tools 
```

以 root 身份从命令行安装 PySide。为了 PyQt，做

```py
 yum install PyQt4 
```

现在您已经安装了 PySide 或 PyQt，我们几乎可以开始学习使用它了——但是首先，我们必须讨论编辑器和 ide。我们将在下一期文章中这样做。