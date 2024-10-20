# 使用 Python 启用屏幕锁定

> 原文：<https://www.blog.pythonlibrary.org/2010/02/09/enabling-screen-locking-with-python/>

几个月前，我的雇主需要锁定我们的一些工作站，以兼容我们从另一个政府机构安装的一些新软件。我们需要在这么多分钟过去后强制这些机器锁定，并且我们需要让用户无法更改这些设置。在本文中，您将了解如何做到这一点，另外，我还将向您展示如何使用 Python 按需锁定您的 Windows 机器。

## 黑进注册表锁定机器

首先，我们来看看我的原始脚本，然后我们将对它进行一点重构，以使代码更好:

```py

from _winreg import CreateKey, SetValueEx
from _winreg import HKEY_CURRENT_USER, HKEY_USERS
from _winreg import REG_DWORD, REG_SZ

try:
    i = 0
    while True:
        subkey = EnumKey(HKEY_USERS, i)
        if len(subkey) > 30:
            break
        i += 1
except WindowsError:
    # WindowsError: [Errno 259] No more data is available
    # looped through all the subkeys without finding the right one
    raise WindowsError("Could not apply workstation lock settings!")

keyOne = CreateKey(HKEY_USERS, r'%s\Control Panel\Desktop' % subkey)
keyTwo = CreateKey(HKEY_CURRENT_USER, r'Software\Microsoft\Windows\CurrentVersion\Policies\System')

# enable screen saver security
SetValueEx(keyOne, 'ScreenSaverIsSecure', 0, REG_DWORD, 1)
# set screen saver timeout
SetValueEx(keyOne, 'ScreenSaveTimeOut', 0, REG_SZ, '420')
# set screen saver
SetValueEx(keyOne, 'SCRNSAVE.EXE', 0, REG_SZ, 'logon.scr')
# disable screen saver tab
SetValueEx(keyTwo, 'NoDispScrSavPage', 0, REG_DWORD, 1)

CloseKey(keyOne)
CloseKey(keyTwo)

```

发现这一点花了一些时间，但是要设置正确的键，我们需要在 HKEY _ 用户配置单元下找到第一个长度大于 30 个字符的子键。我确信可能有更好的方法来做这件事，但是我还没有找到。无论如何，一旦我们找到了长键，我们就跳出循环，打开我们需要的键，或者创建它们，如果它们不存在的话。这就是我们使用 CreateKey 的原因，因为它将做到这一点。接下来，我们设置四个值，然后关闭按键以应用新的设置。您可以阅读注释来了解每个键的作用。现在，让我们稍微改进一下代码，使其成为一个函数:

```py

from _winreg import *

def modifyRegistry(key, sub_key, valueName, valueType, value):
    """
    A simple function used to change values in
    the Windows Registry.
    """
    try:
        key_handle = OpenKey(key, sub_key, 0, KEY_ALL_ACCESS)
    except WindowsError:
        key_handle = CreateKey(key, sub_key)

    SetValueEx(key_handle, valueName, 0, valueType, value)
    CloseKey(key_handle)

try:
    i = 0
    while True:
        subkey = EnumKey(HKEY_USERS, i)
        if len(subkey) > 30:
            break
        i += 1
except WindowsError:
    # WindowsError: [Errno 259] No more data is available
    # looped through all the subkeys without finding the right one
    raise WindowsError("Could not apply workstation lock settings!")

subkey = r'%s\Control Panel\Desktop' % subkey
data= [('ScreenSaverIsSecure', REG_DWORD, 1),
              ('ScreenSaveTimeOut', REG_SZ, '420'),
              ('SCRNSAVE.EXE', REG_SZ, 'logon.scr')]

for valueName, valueType, value in data:
    modifyRegistry(HKEY_USERS, subkey, valueName, 
                   valueType, value)

modifyRegistry(HKEY_CURRENT_USER,
               r'Software\Microsoft\Windows\CurrentVersion\Policies\System',
               'NoDispScrSavPage', REG_DWORD, 1)

```

如您所见，首先我们导入 *_winreg* 模块中的所有内容。不建议这样做，因为您可能会意外地覆盖您已经导入的函数，这就是为什么这有时被称为“毒害名称空间”。然而，我见过的几乎所有使用 _winreg 模块的例子都是这样做的。请参见第一个示例，了解从中导入的正确方法。

接下来，我们创建一个可以打开密钥的通用函数，或者创建密钥(如果它不存在的话)。该函数还将为我们设置值和关闭键。之后，我们基本上做了与上一个例子相同的事情:我们遍历 HKEY _ 用户配置单元并适当地中断。为了稍微混合一下，我们创建了一个保存元组列表的*数据*变量。我们对其进行循环，并使用适当的参数调用我们的函数，为了更好地测量，我们演示了如何在循环之外调用它。

## 以编程方式锁定机器

现在您可能会想，我们已经介绍了如何以编程方式锁定机器。从某种意义上来说，我们做到了。但我们真正做的是设置一个计时器，在将来机器空闲时锁定机器。如果我们现在想锁定机器呢？你们中的一些人可能会想，我们应该只按 Windows 键加“L ”,这是一个好主意。然而，我创建这个脚本的原因是因为我必须不时地用 VNC 远程连接到我的机器，当使用 VNC 时，我需要通过多个步骤来锁定机器，而如果你正确设置了 Python，你只需双击一个脚本文件，让它为你锁定。这就是这个小脚本的作用:

```py

import os

winpath = os.environ["windir"]
os.system(winpath + r'\system32\rundll32 user32.dll, LockWorkStation')

```

这三行脚本导入 *os* 模块，使用其 *environ* 方法获取 Windows 目录，然后调用 *os.system* 来锁定机器。如果您在计算机上打开一个 DOS 窗口，并在其中键入以下内容，您会得到完全相同的效果:

 `C:\windows\system32\rundll32 user32.dll, LockWorkStation` 

## 包扎

现在你知道如何用 Python 锁定你的机器了。如果您将第一个示例放在登录脚本中，那么您可以使用它锁定网络上的一些或所有机器。如果您的用户喜欢闲逛或参加许多会议，但让他们的机器保持登录状态，这将非常方便。这可以防止他们窥探，也可以保护你的公司免受间谍活动的侵扰。