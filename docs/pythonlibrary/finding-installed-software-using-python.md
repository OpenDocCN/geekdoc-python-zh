# 使用 Python 查找已安装的软件

> 原文：<https://www.blog.pythonlibrary.org/2010/03/03/finding-installed-software-using-python/>

你有没有想过你的电脑上安装了什么软件？大多数使用 Windows 的人可能会通过添加/删除程序来找到这些信息，但他们不是程序员。不，程序员必须编写脚本，因为这样做是我们的天性。实际上，我这么做还有另一个原因:我的老板希望我记录用户电脑上安装了什么，这样我们就能知道用户是否安装了未经授权的软件。因此，尝试这样做还有一个实际的原因。

经过一些研究，我发现大多数行为良好的软件会在 Windows 注册表中存储关于它们自己的各种信息。具体来说，它们通常将这些信息存储在以下位置:HKEY _ LOCAL _ MACHINE \ Software \ Microsoft \ Windows \ current version \ Uninstall

应该注意的是，并非所有您安装的程序都会将该信息放入注册表中。例如，有些供应商根本不使用注册中心。此外，恶意软件之类的恶意程序可能也不会把任何东西放在那里。然而，超过 90%的人会这样做，这是获取信息的好方法。要获得更多信息，您可能需要使用 Python 的操作系统模块的“walk”方法进行一些目录遍历，并通过与注册表中的内容进行比较来查找不匹配的内容，或者从您从原始基础机器保存的一些存储映像中查找不匹配的内容。

总之，说够了。让我们来看看代码！注:如果你想跟着做，那么你需要下载并安装蒂姆·戈登的 [WMI 模块](http://timgolden.me.uk/python/wmi/index.html)。

```py

import StringIO
import traceback
import wmi
from _winreg import (HKEY_LOCAL_MACHINE, KEY_ALL_ACCESS, 
                     OpenKey, EnumValue, QueryValueEx)

softFile = open('softLog.log', 'w')
errorLog = open('errors.log', 'w')

r = wmi.Registry ()
result, names = r.EnumKey (hDefKey=HKEY_LOCAL_MACHINE, sSubKeyName=r"Software\Microsoft\Windows\CurrentVersion\Uninstall")

softFile.write('These subkeys are found under "HKEY_LOCAL_MACHINE\Software\Microsoft\Windows\CurrentVersion\Uninstall"\n\n')
errorLog.write("Errors\n\n")
separator = "*" * 80
keyPath = r"Software\Microsoft\Windows\CurrentVersion\Uninstall"

for subkey in names:
    try:
        softFile.write(separator + '\n\n')
        path = keyPath + "\\" + subkey
        key = OpenKey(HKEY_LOCAL_MACHINE, path, 0, KEY_ALL_ACCESS) 
        try:
            temp = QueryValueEx(key, 'DisplayName')
            display = str(temp[0])
            softFile.write('Display Name: ' + display + '\nRegkey: ' + subkey + '\n')
        except:
            softFile.write('Regkey: ' + subkey + '\n')

    except:
        fp = StringIO.StringIO()
        traceback.print_exc(file=fp)
        errorMessage = fp.getvalue()
        error = 'Error for ' + key + '. Message follows:\n' + errorMessage
        errorLog.write(error)
        errorLog.write("\n\n")

softFile.close()
errorLog.close()

```

这是一个很短的片段，但这里有很多东西。让我们打开它。首先，我们导入我们需要的模块，然后打开几个文件: *softFile* 将把我们的脚本找到的所有软件存储在“softLog.log”中，而我们的 *errorLog* 变量将把我们遇到的任何错误存储在“errors.log”中。接下来，我们使用 WMI 来枚举“Uninstall”键中的子项。之后，我们给每个日志文件写一些标题信息，使它们更容易阅读。

最后一个重要部分出现在循环结构中。在这里，我们循环从我们的 WMI 调用返回的结果。在我们的循环中，我们试图提取两条信息:软件的显示名称和与之相关的键。还有很多其他信息，但似乎没有任何标准可以遵循，所以我没有抓住任何一个。您将需要更复杂的错误处理来很好地提取它(或者使用生成器)。无论如何，如果我们试图取出的任何一部分失败了，我们会发现错误并继续下去。嵌套异常处理捕捉与获取显示名称相关的错误，而外部异常处理程序捕捉与访问注册表相关的错误。我们或许应该将这些异常显式化(比如，后者应该是 WindowsError，我认为)，但这只是一个快速而肮脏的脚本。你认为有必要的话，可以随意延长。

如果我们在任一位置遇到错误，我们会记录一些内容。在嵌套的情况下，我们只是将 Regkey 的名称记录到“softLog.log”中，而在外部的情况下，我们只是将一个错误记录到“errors.log”中。最后，我们通过关闭文件进行清理。

下面是我在 Windows XP 机器上运行这个脚本时得到的部分示例:

 `These subkeys are found under "HKEY_LOCAL_MACHINE\Software\Microsoft\Windows\CurrentVersion\Uninstall"`

********************************************************************************

显示名称:Windows 驱动程序包-Garmin(grmnusb)Garmin Devices(03/08/2007 2 . 2 . 1 . 0)
Regkey:45a 7283175 c 62 fac 673 f 913 C1 f 532 c 5361 f 97841
* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

Regkey: AddressBook
********************************************************************************

显示名称:Adobe Flash Player 10 ActiveX
Regkey:Adobe Flash Player ActiveX
* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

显示名称:Adobe Flash Player 10 插件
Regkey: Adobe Flash Player 插件
* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

显示名称:Python 2.4 ado db-2.00
Regkey:ado db-py 2.4
* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

显示名称:代理搜掠
Regkey:代理搜掠
* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

显示名称:亚马逊游戏&软件下载器
Regkey:亚马逊游戏&软件下载器 _ is1
* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

显示名称:亚马逊 MP3 下载器 1.0.3
Regkey:亚马逊 MP3 下载器
* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

显示名称:ZoneAlarm Spy Blocker Toolbar
Regkey:Ask Toolbar _ is1
* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

显示名称:Aspell 英语词典-0.50-2
Regkey: Aspell 英语词典 _ is1
* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

现在你知道如何提取安装在你的 Windows 系统上的大部分软件了。我主要在使用 Python 2.5 的 Windows XP 上测试了这一点，但它应该也能在 Windows Vista 和 7 上运行。Windows 7 和 Vista 通常会强制默认用户以比 XP 更低的权限运行，因此您可能需要以管理员身份或模拟管理员身份运行此脚本。玩得开心！