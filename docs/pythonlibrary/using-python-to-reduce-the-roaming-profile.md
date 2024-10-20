# 使用 Python 简化漫游配置文件

> 原文：<https://www.blog.pythonlibrary.org/2010/02/09/using-python-to-reduce-the-roaming-profile/>

漫游配置文件是福也是祸。如果用户使用互联网，他们浏览器的缓存文件将疯狂增长。如果用户将程序下载到他们的桌面，或者在他们的配置文件中的任何地方创建大的 Powerpoint 文件，那么无论用户何时登录或注销，都必须对它们进行管理。这个问题有几种解决方案:磁盘配额，阻止下载或放入个人资料的能力，等等。在本文中，我将向您展示如何使用 Python 从用户配置文件中排除特定的目录。

这基本上只是一个 Windows 注册表黑客。和往常一样，在对注册表进行任何更改之前，请务必备份注册表，以防出现严重问题，导致您的计算机无法启动。

```py

from _winreg import *

try:
    key = OpenKey(HKEY_CURRENT_USER, r"SOFTWARE\Microsoft\Windows NT\CurrentVersion\Winlogon",
                  0, KEY_ALL_ACCESS)
except WindowsError:
    key = CreateKey(HKEY_CURRENT_USER, r"SOFTWARE\Microsoft\Windows NT\CurrentVersion\Winlogon")

# Exclude directories from roaming profile 
prof_dirs = "Local Settings;Temporary Internet Files;History;Temp;My Documents;Recent"
SetValueEx(key, "ExcludeProfileDirs", 0, REG_SZ, prof_dirs)     
CloseKey(key)

```

这段代码非常简单。首先，我们从 _winreg 导入各种模块和常量。然后，我们尝试打开适当的注册表项，如果该项不存在，就创建它。接下来，我们创建了一个分号分隔的目录字符串，以从漫游配置文件中排除。最后，我们设置适当的值并关闭键。

这就是这个简单脚本的全部内容！