# PyInstaller:打包 Python 应用程序(Windows、Mac 和 Linux)

> 原文：<https://www.pythoncentral.io/pyinstaller-package-python-applications-windows-mac-linux/>

PyInstaller 是一个用于将 Python 脚本转换成独立应用程序的程序。PyInstaller 允许您在计算机上运行用 Python 编写的应用程序，而不需要用户安装 Python。当您需要将程序作为独立的应用程序分发给最终用户时，这是一个很好的选择。PyInstaller 目前只支持 Python 2.3 到 2.7。

PyInstaller 声称可以兼容很多现成的第三方库或包。完全支持 PyQt、Django 和 matplotlib。

首先，从官方[站点](http://www.pyinstaller.org/ "PyInstaller Official Site")下载并解压 PyInstaller。

PyInstaller 是一个应用程序，而不是一个包。所以没必要安装在你的电脑里。要开始，请打开命令提示符(Windows)或终端(Mac 和 Linux)并转到 PyInstaller 文件夹。

```py

cd pyinstaller

```

现在假设你要打包 myscript.py，我把它保存到 pyinstaller 文件夹:
【python】
# myscript . py
print(' Hello World！')

然后，为了创建可执行文件，只需运行`python pyinstaller.py myscript.py`，您将看到大量输出，一个名为`myscript`的文件夹将被创建，其中包含两个文件夹和一个文件。PyInstaller 使用`build`文件夹作为临时文件夹来创建可执行文件所需的文件。`dist`文件夹存储可执行文件和运行该可执行文件所需的所有文件。删除构建文件夹是安全的。名为`myscript.spec`的文件对于定制 PyInstaller 打包应用程序的方式很有用。

现在测试你的可执行文件是否有效:

*   [窗户](#custom-tab-0-windows)
*   [Mac 和 Linux](#custom-tab-0-mac-and-linux)

*   [窗户](#)

[python]
cd myscript/dist/myscript
myscript
[/python]

*   [Mac 和 Linux](#)

[python]
cd myscript/dist/myscript
./myscript
[/python]

你现在应该看到一个“你好，世界！”印在屏幕上。

记住，运行`python pyinstaller.py myscript.py`假设您的 path 环境变量中有 Python。如果不是这样，就在 Windows 中使用`C:\Python27\python.exe pyinstaller.py myscript.py`。在 Linux 和 Mac OS X 的大部分时间里，Python 会出现在你的 path 环境变量中。

## GUI 应用程序

现在是时候创建一个 GUI 应用程序了。在这个例子中，我们将使用 Tkinter:
【python】
# tkexample . py
' ' '一个非常基本的 Tkinter 例子'
从 Tkinter import *
root = Tk()
root . title('一个 Tk 应用')
Label(text= '我是标签')。pack(pady = 15)
root . main loop()

要打包它，必须使用`--windowed`标志，否则应用程序将无法启动:

```py

python pyinstaller.py --windowed tkexample.py

```

此时，您可以导航到 dist 文件夹，并通过双击它来运行应用程序。

**Mac OS X 用户注意:**如果你使用预装的 Python 版本，上面使用 Tkinter 的例子工作正常，如果你自己安装或更新 Python，你会发现运行打包的应用程序时会出现一些问题。

## 使用外部模块

前面的例子是从 Python 标准库中导入模块。默认情况下，PyInstaller 包含标准库模块。然而，如果我们安装了第三方库，PyInstaller 可能不会包含它。在大多数情况下，我们需要创建“钩子”来告诉 PyInstaller 包含这些模块。这方面的一个例子是一个使用 ReportLab 库制作 PDF 文件的应用程序:
【python】
# invoice . py
from report lab . PDF gen 导入画布
from reportlab.lib.pagesizes 导入信件
from reportlab.lib.units 导入 mm

if _ _ name _ _ = = ' _ _ main _ _ ':
name = u ' Mr。约翰·多伊'
城市= '佩雷拉'
地址= '榆树街'
电话= '555-7241'
c =画布。Canvas(filename='invoice.pdf '，pagesize= (letter[0]，letter[1]/2))
c . set font(' Helvetica '，10)
#打印客户数据
c.drawString(107*mm，120*mm，姓名)
c.drawString(107*mm，111*mm，城市)
c.drawString(107*mm，106*mm，地址)
c.drawString(107*mm

“钩子”模块是一个具有特殊名称的 Python 文件，用于告诉 PyInstaller 包含一个特定的模块。在 Google 上搜索时，我找到了打包 ReportLab 应用程序所需的钩子[这里](https://github.com/pyinstaller/pyinstaller/tree/develop/PyInstaller/hooks)，并把它们放在一个名为“hooks”的文件夹中:

+-hooks/
|-hook-report lab . pdf base . _ font data . py
|-hook-report lab . pdf base . py
|-hook-report lab . py

`hook-reportlab.py`和`hook-reportlab.pdfbase.py`为空文件，`hook-reportlab.pdfbase._fontdata.py`包含:
【python】
# hook-report lab . pdf base . _ font data . py
hidden imports =[
' _ font data _ enc _ macexpert '，
'_fontdata_enc_macroman '，
'_fontdata_enc_pdfdoc '，
'_fontdata_enc_standard '，
'_fontdata_enc_symbol ' '

现在为了打包可执行文件，我们必须运行`python pyinstaller.py --additional-hooks-dir=hooks/ invoice.py`。`additional-hooks-dir`标志告诉 PyInstaller 在指定的目录中搜索钩子。

## 结论

如果您的脚本只从 Python 标准库中导入模块，或者导入官方[支持的包](https://docs.microsoft.com/en-us/power-bi/connect-data/service-python-packages-support "Supported Packages - Pyinstaller")列表中包含的模块，Pyinstaller 会工作得很好。使用这些受支持的包使得打包应用程序变得非常简单，但是当我们需要使用第三方不支持的模块时，要让它工作起来可能会很棘手。