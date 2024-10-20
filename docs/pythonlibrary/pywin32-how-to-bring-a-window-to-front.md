# 如何将一个窗口放到最前面

> 原文：<https://www.blog.pythonlibrary.org/2014/10/20/pywin32-how-to-bring-a-window-to-front/>

我最近看到有人问如何在 Windows 中把一个窗口放在最前面，我意识到我有一些旧的未发布的代码可能会帮助别人完成这项任务。很久以前，Tim Golden(可能还有 PyWin32 邮件列表上的其他一些人)向我展示了如何在 windows XP 上让 Windows 出现在最前面，尽管应该注意它也可以在 Windows 7 上工作。如果您想继续学习，您需要下载并安装您自己的 [PyWin32](http://sourceforge.net/projects/pywin32/) 副本。

我们需要选择一些东西放在前面。我喜欢用记事本进行测试，因为我知道它会出现在现有的每一个 Windows 桌面上。打开记事本，然后把其他应用程序的窗口放在它前面。

现在我们准备看一些代码:

```py

import win32gui

def windowEnumerationHandler(hwnd, top_windows):
    top_windows.append((hwnd, win32gui.GetWindowText(hwnd)))

if __name__ == "__main__":
    results = []
    top_windows = []
    win32gui.EnumWindows(windowEnumerationHandler, top_windows)
    for i in top_windows:
        if "notepad" in i[1].lower():
            print i
            win32gui.ShowWindow(i[0],5)
            win32gui.SetForegroundWindow(i[0])
            break

```

对于这个小脚本，我们只需要 PyWin32 的 **win32gui** 模块。我们编写了一个小函数，它接受一个窗口句柄和一个 Python 列表。然后我们调用 win32gui 的 **EnumWindows** 方法，该方法接受一个回调和一个额外的参数，该参数是一个 Python 对象。根据[文档](http://docs.activestate.com/activepython/3.1/pywin32/win32gui__EnumWindows_meth.html)，EnumWindows 方法“通过将句柄传递给每个窗口，依次传递给应用程序定义的回调函数，枚举屏幕上的所有顶级窗口”。所以我们把我们的方法传递给它，它枚举窗口，把每个窗口的句柄加上我们的 Python 列表传递给我们的函数。这有点像一个乱七八糟的室内设计师。

一旦完成，你的 **top_windows** 列表将会充满大量的项目，其中大部分你甚至不知道正在运行。如果你愿意，你可以打印我们报告并检查你的结果。这真的很有趣。但是出于我们的目的，我们将跳过它，只循环遍历列表，查找单词“Notepad”。一旦我们找到它，我们使用 win32gui 的 **ShowWindow** 和 **SetForegroundWindow** 方法将应用程序带到前台。

请注意，确实需要寻找一个唯一的字符串，以便调出正确的窗口。如果您运行多个记事本实例并打开不同文件，会发生什么情况？使用当前代码，您将把找到的第一个 Notepad 实例向前移动，这可能不是您想要的。

你可能想知道为什么有人会愿意首先这么做。在我的例子中，我曾经有一个项目，我必须将一个特定的窗口放到前台，并使用 SendKeys 自动输入它。这是一段丑陋脆弱的代码，我不希望它发生在任何人身上。幸运的是，现在有更好的工具来处理这类事情，比如 [pywinauto](https://code.google.com/p/pywinauto/) ，但是您可能仍然会发现这段代码对您遇到的一些深奥的事情很有帮助。玩得开心！

*注意:这段代码是在 Windows 7 上使用 Python 2.7.8 和 PyWin32 219 测试的。*