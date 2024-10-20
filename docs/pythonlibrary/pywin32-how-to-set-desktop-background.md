# PyWin32:如何在 Windows 上设置桌面背景

> 原文：<https://www.blog.pythonlibrary.org/2014/10/22/pywin32-how-to-set-desktop-background/>

在我做系统管理员的时候，我们考虑在登录时将用户的窗口桌面背景设置为特定的图像。因为我负责用 Python 编写的登录脚本，所以我决定做一些研究，看看是否有办法做到这一点。在本文中，我们将研究完成这项任务的两种不同方法。本文中的代码是在 Windows 7 上使用 Python 2.7.8 和 PyWin32 219 测试的。

对于第一种方法，您需要下载一份 [PyWin32](http://sourceforge.net/projects/pywin32/) 并安装它。现在让我们看看代码:

```py

# based on http://dzone.com/snippets/set-windows-desktop-wallpaper
import win32api, win32con, win32gui

#----------------------------------------------------------------------
def setWallpaper(path):
    key = win32api.RegOpenKeyEx(win32con.HKEY_CURRENT_USER,"Control Panel\\Desktop",0,win32con.KEY_SET_VALUE)
    win32api.RegSetValueEx(key, "WallpaperStyle", 0, win32con.REG_SZ, "0")
    win32api.RegSetValueEx(key, "TileWallpaper", 0, win32con.REG_SZ, "0")
    win32gui.SystemParametersInfo(win32con.SPI_SETDESKWALLPAPER, path, 1+2)

if __name__ == "__main__":
    path = r'C:\Users\Public\Pictures\Sample Pictures\Jellyfish.jpg'
    setWallpaper(path)

```

在这个例子中，我们使用微软随 Windows 提供的一个示例图像。在上面的代码中，我们编辑了一个 Windows 注册表项。如果你愿意，可以使用 Python 自己的 [_winreg 模块](https://docs.python.org/2/library/_winreg.html)来完成前 3 行。最后一行告诉 Windows 将桌面设置为我们提供的图像。

现在让我们看看另一种方法，它利用了 [ctypes 模块](https://docs.python.org/2/library/ctypes.html)和 PyWin32。

```py

import ctypes
import win32con

def setWallpaperWithCtypes(path):
    # This code is based on the following two links
    # http://mail.python.org/pipermail/python-win32/2005-January/002893.html
    # http://code.activestate.com/recipes/435877-change-the-wallpaper-under-windows/
    cs = ctypes.c_buffer(path)
    ok = ctypes.windll.user32.SystemParametersInfoA(win32con.SPI_SETDESKWALLPAPER, 0, cs, 0)

if __name__ == "__main__":
    path = r'C:\Users\Public\Pictures\Sample Pictures\Jellyfish.jpg'
    setWallpaperWithCtypes(path)

```

在这段代码中，我们创建了一个 buffer 对象，然后我们将它传递给与上一个示例中基本相同的命令，即 **SystemParametersInfoA** 。您会注意到，在后一种情况下，我们不需要编辑注册表。如果您查看示例代码中列出的链接，您会注意到一些用户发现 Windows XP 只允许将位图设置为桌面背景。我在 Windows 7 上用 JPEG 格式进行了测试，效果不错。

现在你可以创建自己的脚本来随机改变桌面的背景！或者，您可能只是在自己的系统管理职责中使用这些知识。玩得开心！