# PyWin32:如何获得应用程序的版本号

> 原文：<https://www.blog.pythonlibrary.org/2014/10/23/pywin32-how-to-get-an-applications-version-number/>

有时你需要知道你正在使用的软件版本。找到这些信息的正常方法通常是打开程序，进入**帮助**菜单，点击**关于**菜单项。但这是一个 Python 博客，我们想通过编程来实现！要在 Windows 机器上做到这一点，我们需要 [PyWin32](http://sourceforge.net/projects/pywin32/) 。在本文中，我们将研究两种不同的获取应用程序版本号的方法。

* * *

### 使用 win32api 获取版本

首先，我们将使用 PyWin32 的 **win32api** 模块获取版本号。其实挺好用的。让我们来看看:

```py

from win32api import GetFileVersionInfo, LOWORD, HIWORD

def get_version_number(filename):
    try:
        info = GetFileVersionInfo (filename, "\\")
        ms = info['FileVersionMS']
        ls = info['FileVersionLS']
        return HIWORD (ms), LOWORD (ms), HIWORD (ls), LOWORD (ls)
    except:
        return "Unknown version"

if __name__ == "__main__":
    version = ".".join([str (i) for i in get_version_number (
        r'C:\Program Files\Internet Explorer\iexplore.exe')])
    print version

```

这里我们用一个路径调用 **GetFileVersionInfo** ，然后尝试解析结果。如果我们不能解析它，那就意味着这个方法没有返回任何有用的东西，这将导致一个异常被抛出。我们捕捉异常并返回一个字符串，告诉我们找不到版本号。对于本例，我们检查安装了哪个版本的 Internet Explorer。

* * *

### 使用 win32com 获取版本

为了让事情变得更有趣，在下面的例子中，我们使用 PyWin32 的 win32com 模块来检查 Google Chrome 的版本号。让我们来看看:

```py

# based on http://stackoverflow.com/questions/580924/python-windows-file-version-attribute
from win32com.client import Dispatch

def get_version_via_com(filename):
    parser = Dispatch("Scripting.FileSystemObject")
    version = parser.GetFileVersion(filename)
    return version

if __name__ == "__main__":
    path = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
    print get_version_via_com(path)

```

我们在这里所做的就是导入 win32com 的 **Dispatch** 类并创建该类的一个实例。接下来，我们调用它的 **GetFileVersion** 方法，并将路径传递给我们的可执行文件。最后，我们返回结果，这个结果或者是数字，或者是一条消息，表明没有可用的版本信息。我更喜欢第二种方法，因为当没有找到版本信息时，它会自动返回一条消息。

* * *

### 包扎

现在您知道了如何在 Windows 上检查应用程序版本号。如果您需要检查关键软件是否需要升级，或者您需要确保它没有升级，因为一些其他应用程序需要旧版本，这可能会很有帮助。

* * *

### 相关阅读

*   stack overflow:[Python windows 文件版本属性](http://stackoverflow.com/q/580924/393194)
*   [用 Python 获取系统信息](https://www.blog.pythonlibrary.org/2010/01/27/getting-windows-system-information-with-python/)