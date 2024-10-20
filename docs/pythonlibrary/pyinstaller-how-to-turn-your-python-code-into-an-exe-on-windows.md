# py installer——如何将你的 Python 代码转换成 Windows 上的 Exe 文件

> 原文：<https://www.blog.pythonlibrary.org/2021/05/27/pyinstaller-how-to-turn-your-python-code-into-an-exe-on-windows/>

您刚刚创建了一个非常棒的新应用程序。可能是游戏，也可能是图像查看器。无论您的应用程序是什么，您都希望与您的朋友或家人分享。然而，你知道他们不知道如何安装 Python 或任何依赖项。你是做什么的？你需要将你的代码转换成可执行文件的东西！

Python 有许多不同的工具可以用来将 Python 代码转换成 Windows 可执行文件。以下是一些您可以使用的不同工具:

*   PyInstaller(安装程序)
*   py2exe
*   cx _ 冻结
*   努特卡
*   公文包

这些不同的工具都可以用来为 Windows 创建可执行文件。它们的工作方式略有不同，但最终结果是您将拥有一个可执行文件，也许还有一些您需要分发的其他文件。

PyInstaller 和公文包可用于创建 Windows 和 MacOS 可执行文件。Nuitka 有一点不同，它在将 Python 代码转换成可执行文件之前，先将其转换成 C 代码。这意味着结果比 PyInstaller 的可执行文件小得多。

然而，对于本文，您将关注于 **PyInstaller** 。为此，它是最受欢迎的包之一，并得到了很多支持。PyInstaller 也有很好的文档，并且有许多教程可供使用。

在本文中，您将了解到:

*   安装 PyInstaller
*   为命令行应用程序创建可执行文件
*   为 GUI 创建可执行文件

让我们将一些代码转换成 Windows 可执行文件！

## 安装 PyInstaller

要开始，您需要安装 PyInstaller。幸运的是，PyInstaller 是一个 Python 包，可以使用`pip`轻松安装:

```py
python -m pip install pyinstaller
```

该命令将在您的计算机上安装 PyInstaller 及其所需的任何依赖项。现在，您应该准备好使用 PyInstaller 创建可执行文件了！

## 为命令行应用程序创建可执行文件

下一步是选择一些你想转换成可执行文件的代码。你可以使用我的书 [Python 101: 2nd Edition](https://leanpub.com/py101) 的**第 32 章** 中的 [**PySearch** 实用程序，把它变成二进制。代码如下:](https://github.com/driscollis/python101code)

```py
# pysearch.py

import argparse
import pathlib

def search_folder(path, extension, file_size=None):
    """
    Search folder for files
    """
    folder = pathlib.Path(path)
    files = list(folder.rglob(f'*.{extension}'))

    if not files:
        print(f'No files found with {extension=}')
        return

    if file_size is not None:
        files = [f for f in files
                 if f.stat().st_size > file_size]

    print(f'{len(files)} *.{extension} files found:')
    for file_path in files:
        print(file_path)

def main():
    parser = argparse.ArgumentParser(
        'PySearch',
        description='PySearch - The Python Powered File Searcher')
    parser.add_argument('-p', '--path',
                        help='The path to search for files',
                        required=True,
                        dest='path')
    parser.add_argument('-e', '--ext',
                        help='The extension to search for',
                        required=True,
                        dest='extension')
    parser.add_argument('-s', '--size',
                        help='The file size to filter on in bytes',
                        type=int,
                        dest='size',
                        default=None)

    args = parser.parse_args()
    search_folder(args.path, args.extension, args.size)

if __name__ == '__main__':
    main()
```

接下来，在 Windows 中打开命令提示符(cmd.exe ),导航到包含您的`pysearch.py`文件的文件夹。要将 Python 代码转换为二进制可执行文件，您需要运行以下命令:

```py
pyinstaller pysearch.py
```

如果 Python 不在您的 Windows 路径上，您可能需要键入到`pyinstaller`的完整路径来运行它。它将位于一个**脚本**文件夹中，无论你的 Python 安装在系统的哪个位置。

当您运行该命令时，您将看到类似于以下内容的一些输出:

```py
6531 INFO: PyInstaller: 3.6
6576 INFO: Python: 3.8.2
6707 INFO: Platform: Windows-10-10.0.10586-SP0
6828 INFO: wrote C:\Users\mike\AppData\Local\Programs\Python\Python38-32\pysearch.spec
6880 INFO: UPX is not available.
7110 INFO: Extending PYTHONPATH with paths
['C:\\Users\\mike\\AppData\\Local\\Programs\\Python\\Python38-32',
 'C:\\Users\\mike\\AppData\\Local\\Programs\\Python\\Python38-32']
7120 INFO: checking Analysis
7124 INFO: Building Analysis because Analysis-00.toc is non existent
7128 INFO: Initializing module dependency graph...
7153 INFO: Caching module graph hooks...
7172 INFO: Analyzing base_library.zip ...
```

PyInstaller 非常冗长，会打印出大量输出。完成后，你将有一个`dist`文件夹，里面有一个`pysearch`文件夹。在`pysearch`文件夹中有许多其他文件，包括一个名为`pysearch.exe`的文件。您可以尝试在命令提示符下导航到`pysearch`文件夹，然后运行`pysearch.exe`:

```py
C:\Users\mike\AppData\Local\Programs\Python\Python38-32\dist\pysearch>pysearch.exe
usage: PySearch [-h] -p PATH -e EXTENSION [-s SIZE]
PySearch: error: the following arguments are required: -p/--path, -e/--ext
```

这看起来是一个非常成功的构建！然而，如果你想把可执行文件给你的朋友，你必须给他们整个`pysearch`文件夹，因为那里的所有其他文件也是必需的。

您可以通过传递`--onefile`标志来解决这个问题，如下所示:

```py
pyinstaller pysearch.py --onefile
```

该命令的输出类似于第一个命令。这次当你进入`dist`文件夹时，你会发现一个名为`pysearch.exe`的文件，而不是一个装满文件的文件夹。

## 为 GUI 创建可执行文件

为 GUI 创建可执行文件与为命令行应用程序创建略有不同。原因是 GUI 是主界面，PyInstaller 的默认界面是用户将使用命令提示符或控制台窗口。如果您运行在上一节中学习的 PyInstaller 命令，它将成功创建您的可执行文件。但是，当您使用可执行文件时，除了 GUI 之外，您还会看到一个命令提示符。

你通常不想这样。要抑制命令提示符，您需要使用`--noconsole`标志。

为了测试这是如何工作的，从 [Python 101:第二版](https://amzn.to/2Zo1ARG)的**第 42 章**中获取用 wxPython 创建的图像查看器的代码。为了方便起见，下面是代码:

```py
# image_viewer.py

import wx

class ImagePanel(wx.Panel):

    def __init__(self, parent, image_size):
        super().__init__(parent)
        self.max_size = 240

        img = wx.Image(*image_size)
        self.image_ctrl = wx.StaticBitmap(self, 
                                          bitmap=wx.Bitmap(img))

        browse_btn = wx.Button(self, label='Browse')
        browse_btn.Bind(wx.EVT_BUTTON, self.on_browse)

        self.photo_txt = wx.TextCtrl(self, size=(200, -1))

        main_sizer = wx.BoxSizer(wx.VERTICAL)
        hsizer = wx.BoxSizer(wx.HORIZONTAL)

        main_sizer.Add(self.image_ctrl, 0, wx.ALL, 5)
        hsizer.Add(browse_btn, 0, wx.ALL, 5)
        hsizer.Add(self.photo_txt, 0, wx.ALL, 5)
        main_sizer.Add(hsizer, 0, wx.ALL, 5)

        self.SetSizer(main_sizer)
        main_sizer.Fit(parent)
        self.Layout()

    def on_browse(self, event):
        """
        Browse for an image file
        @param event: The event object
        """
        wildcard = "JPEG files (*.jpg)|*.jpg"
        with wx.FileDialog(None, "Choose a file",
                           wildcard=wildcard,
                           style=wx.ID_OPEN) as dialog:
            if dialog.ShowModal() == wx.ID_OK:
                self.photo_txt.SetValue(dialog.GetPath())
                self.load_image()

    def load_image(self):
        """
        Load the image and display it to the user
        """
        filepath = self.photo_txt.GetValue()
        img = wx.Image(filepath, wx.BITMAP_TYPE_ANY)

        # scale the image, preserving the aspect ratio
        W = img.GetWidth()
        H = img.GetHeight()
        if W > H:
            NewW = self.max_size
            NewH = self.max_size * H / W
        else:
            NewH = self.max_size
            NewW = self.max_size * W / H
        img = img.Scale(NewW,NewH)

        self.image_ctrl.SetBitmap(wx.Bitmap(img))
        self.Refresh()

class MainFrame(wx.Frame):

    def __init__(self):
        super().__init__(None, title='Image Viewer')
        panel = ImagePanel(self, image_size=(240,240))
        self.Show()

if __name__ == '__main__':
    app = wx.App(redirect=False)
    frame = MainFrame()
    app.MainLoop()
```

要将其转换为可执行文件，您可以运行以下 PyInstaller 命令:

```py
pyinstaller.exe image_viewer.py --noconsole
```

请注意，在这里使用`--onefile`标志的是**而不是**。Windows Defender 会将使用`--onefile`创建的 GUI 标记为恶意软件并将其删除。您可以通过不使用`--onefile`标志或对可执行文件进行数字签名来解决这个问题。从 Windows 10 开始，所有 GUI 应用程序都需要签名，否则将被视为恶意软件。

微软有一个**签名工具**你可以使用，但是你需要购买一个数字证书或者用 **Makecert** 、一个. NET 工具或者类似的东西创建一个自签名证书。

## 包扎

用 Python 创建可执行文件有很多不同的方法。在本文中，您使用了 PyInstaller。您了解了以下主题:

*   安装 PyInstaller
*   为命令行应用程序创建可执行文件
*   为 GUI 创建可执行文件

PyInstaller 有许多其他标志，您可以使用它们在生成可执行文件时修改其行为。如果你在 PyInstaller 上遇到问题，有一个邮件列表可以帮助你。或者你可以在谷歌和 StackOverflow 上搜索。出现的大多数常见问题都包含在 PyInstaller 文档中，或者很容易通过在线搜索发现。

| [![](img/9437a5e03f2225dbc315c4e7e5b908b3.png)](https://leanpub.com/py101/) | 您想了解更多关于 Python 的知识吗？

### Python 101 -第二版

#### **立即在 [Leanpub](https://leanpub.com/py101) 或[亚马逊](https://amzn.to/2Zo1ARG) 购买**

 |