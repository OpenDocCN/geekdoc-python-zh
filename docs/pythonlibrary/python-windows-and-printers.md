# Python、Windows 和打印机

> 原文：<https://www.blog.pythonlibrary.org/2010/02/14/python-windows-and-printers/>

除了软件开发之外，我还做大量的技术支持工作。在我们的小店里，我们可以对任何与技术相关的问题进行故障诊断，从网络到软件到打印机。我认为最烦人的一点是试图让打印机按照用户想要的方式工作。另一个问题是为用户设置打印机，这些用户必须在 PC 之间漫游，这是他们工作的一部分。这些用户通常只需要在任何给定时间位于其特定位置的打印机。很难适应这种类型的用户，尤其是如果电脑被 24/7 使用的话，我的情况就是如此。这就是 Python 的用武之地。

在本文中，我将向您展示如何访问机器上当前安装的打印机，更改默认打印机并安装另一台打印机。我还将向您展示如何访问有关已安装打印机的各种信息，因为这些信息有助于编写其他管理脚本。

要继续学习，您需要 Python 2.4 - 3.x 和 [PyWin32 包](http://sourceforge.net/projects/pywin32/files/)。

今天的第一个技巧是，让我们看看我们的电脑上目前安装了哪些打印机:

```py

import win32print
printers = win32print.EnumPrinters(5)
print printers

```

可以在 EnumPrinters 调用中使用不同的整数来获取更多或更少的信息。更多信息见[文档](http://docs.activestate.com/activepython/2.5/pywin32/win32print__EnumPrinters_meth.html)(你可能也需要看看 MSDN)。无论如何，这是一个示例输出:

 `((8388608, 'SnagIt 9,SnagIt 9 Printer,', 'SnagIt 9', ''), (8388608, 'Samsung ML-2250 Series PCL 6,Samsung ML-2250 Series PCL 6,', 'Samsung ML-2250 Series PCL 6', ''), (8388608, 'PDFCreator,PDFCreator,', 'PDFCreator', 'eDoc Printer'), (8388608, 'Microsoft XPS Document Writer,Microsoft XPS Document Writer,', 'Microsoft XPS Document Writer', ''))` 

如您所见，EnumPrinters 调用返回一个具有嵌套元组的元组。如果我没记错的话，如果打印机是网络打印机，那么最后一个参数将是 UNC 路径。在我工作的地方，我们不得不淘汰一些装有打印机的服务器，并需要一种方法来更改用户的打印机设置，以便它们指向新的路径。使用上面收集的信息使这变得容易多了。

例如，如果我的脚本遍历该列表，发现一台打印机正在使用一个过时的 UNC 路径，我可以这样做来修复它:

```py

import win32print
win32print.DeletePrinterConnection('\\\\oldUNC\path\to\printer')
win32print.AddPrinterConnection('\\\\newUNC\path\to\printer')

```

安装打印机的另一种方法是使用低级命令行调用子进程模块:

```py

import subprocess
subprocess.call(r'rundll32 printui.dll PrintUIEntry /in /q /n \\UNC\path\to\printer')

```

对于我上面提到的漫游用户的情况，我通常还需要设置默认打印机，这样用户就不会意外地打印到不同的部门。我发现有两种方法非常有效。如果您知道打印机的名称，您可以使用以下内容:

```py

import win32print
win32print.SetDefaultPrinter('EPSON Stylus C86 Series')

```

在上面的代码中，我将默认值设置为 Epson。该名称应该与 Windows 中“打印机和传真”对话框中显示的名称完全相同(在 Windows XP 上，转到“开始”、“设置”、“打印机和传真”)。另一种方法是使用另一个子流程调用:

```py

import subprocess
subprocess.call(r'rundll32 printui.dll PrintUIEntry /y /n \\UNC\path\to\printer')

```

win32print 还支持许多其他功能。您可以启动和停止打印作业，设置打印作业的优先级，获取打印机的配置，安排作业等等。我希望这能对你有所帮助。

**延伸阅读**

*   [win32 打印文档](http://docs.activestate.com/activepython/2.5/pywin32/win32print.html)
*   [关注 WMI 的新印刷工作](http://timgolden.me.uk/python/wmi/cookbook.html#watch-for-new-print-jobs)
*   [用 WMI 显示打印作业](http://timgolden.me.uk/python/wmi/cookbook.html#show-print-jobs)