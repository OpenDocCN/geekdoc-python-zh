# 关于 Python 的更多 Windows 系统信息

> 原文：<https://www.blog.pythonlibrary.org/2010/02/06/more-windows-system-information-with-python/>

上个月我写了一篇关于获取 Windows 系统信息的文章，我在我的一条评论中提到有另一个脚本可以做这些事情，但是我找不到它。嗯，今天我去挖掘了一下，找到了我想要的剧本。因此，我们要回到兔子洞去寻找更多的技巧和窍门，以便使用 Python 获得关于 Windows 奇妙世界的信息。

以下脚本摘自我在职时使用和维护的登录脚本。我们通常需要几种方法来识别特定用户的机器。幸运的是，大多数工作站都有几个唯一的标识符，比如 IP、MAC 地址和工作站名称(尽管这些都不一定是唯一的...例如，我们实际上有一个工作站，它的网卡与我们的一台服务器的 MAC 相同。不管怎样，让我们来看看代码吧！

## 如何获得您工作站的名称

在本节中，我们将使用平台模块来获取我们计算机的名称。在我之前的文章中，我们实际上提到了这个技巧，但是因为我们在下一个片段中需要这个信息，所以我将在这里重复这个技巧:

```py

from platform import node
computer_name = node()

```

一点也不痛苦，对吧？只有两行代码，我们得到了我们需要的。但是实际上至少还有一种方法可以得到它:

```py

import socket
computer_name = socket.gethostname()

```

这个片段也非常简单，尽管第一个片段稍微短一些。我们所要做的就是导入内置的 *socket* 模块，并调用它的 *gethostname* 方法。现在我们已经准备好获取我们电脑的 IP 地址了。

## 如何用 Python 获取你的 PC 的 IP 地址

我们可以使用上面收集的信息来获取我们电脑的 IP 地址:

```py

import socket
ip_address = socket.gethostbyname(computer_name)
# or we could do this:
ip_address2 = socket.gethostbyname(socket.gethostname())

```

在这个例子中，我们再次使用了 *socket* 模块，但是这次我们使用了它的 *gethostbyname* 方法并传入了 PC 的名称。然后*插座*模块将返回 IP 地址。

也可以用[蒂姆·戈登的](http://ramblings.timgolden.me.uk/)T2 的【WMI】模块。下面的例子来自他精彩的 WMI 烹饪书:

```py

import wmi
c = wmi.WMI ()

for interface in c.Win32_NetworkAdapterConfiguration (IPEnabled=1):
  print interface.Description
  for ip_address in interface.IPAddress:
    print ip_address
  print

```

它所做的只是遍历已安装的网络适配器，并打印出它们各自的描述和 IP 地址。

## 如何用 Python 获取 MAC 地址

现在我们可以将注意力转向获取 MAC 地址。我们将研究两种不同的方法来获得它，从 ActiveState 方法开始:

```py

def get_macaddress(host='localhost'):
    """ Returns the MAC address of a network host, requires >= WIN2K. """
    # http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/347812
    import ctypes
    import socket
    import struct

    # Check for api availability
    try:
        SendARP = ctypes.windll.Iphlpapi.SendARP
    except:
        raise NotImplementedError('Usage only on Windows 2000 and above')

    # Doesn't work with loopbacks, but let's try and help.
    if host == '127.0.0.1' or host.lower() == 'localhost':
        host = socket.gethostname()

    # gethostbyname blocks, so use it wisely.
    try:
        inetaddr = ctypes.windll.wsock32.inet_addr(host)
        if inetaddr in (0, -1):
            raise Exception
    except:
        hostip = socket.gethostbyname(host)
        inetaddr = ctypes.windll.wsock32.inet_addr(hostip)

    buffer = ctypes.c_buffer(6)
    addlen = ctypes.c_ulong(ctypes.sizeof(buffer))
    if SendARP(inetaddr, 0, ctypes.byref(buffer), ctypes.byref(addlen)) != 0:
        raise WindowsError('Retreival of mac address(%s) - failed' % host)

    # Convert binary data into a string.
    macaddr = ''
    for intval in struct.unpack('BBBBBB', buffer):
        if intval > 15:
            replacestr = '0x'
        else:
            replacestr = 'x'
        if macaddr != '':
            macaddr = ':'.join([macaddr, hex(intval).replace(replacestr, '')])
        else:
            macaddr = ''.join([macaddr, hex(intval).replace(replacestr, '')])

    return macaddr.upper()

```

由于上面的代码不是我写的，所以就不深入了。然而，我的理解是，这个脚本首先检查它是否可以执行 ARP 请求，这仅在 Windows 2000 和更高版本上可用。一旦得到确认，它就会尝试使用 ctypes 模块来获取 inet 地址。完成之后，它会通过一些我不太理解的东西来建立 MAC 地址。

当我第一次开始维护这段代码时，我认为一定有更好的方法来获取 MAC 地址。我想也许 Tim Golden 的 WMI 模块或者 PyWin32 包会是答案。我很确定他给了我下面的片段，或者我是在 Python 邮件列表档案中找到的:

```py

def getMAC_wmi():
    """uses wmi interface to find MAC address"""    
    interfaces = []
    import wmi
    c = wmi.WMI ()
    for interface in c.Win32_NetworkAdapterConfiguration (IPEnabled=1):
        if interface.DNSDomain == 'www.myDomain.com':
            return interface.MACAddress

```

不幸的是，虽然这种方法有效，但它明显降低了登录脚本的速度，所以我最终还是使用了原来的方法。我想 Golden 先生已经发布了一个较新版本的 wmi 模块，所以现在可能更快了。

## 如何获得用户名

用 Python 获取当前用户的登录名很简单。你所需要的就是 [PyWin32 包](http://sourceforge.net/projects/pywin32/files/)。

```py

from win32api import GetUserName
userid = GetUserName()

```

一次快速导入，我们就有了两行代码的用户名。

## 如何找到用户所在的组

我们可以使用上面获得的 userid 来找出它属于哪个组。

```py

import os
from win32api import GetUserName
from win32com.client import GetObject

def _GetGroups(user):
    """Returns a list of the groups that 'user' belongs to."""
    groups = []
    for group in user.Groups ():
        groups.append (group.Name)
    return groups

userid = GetUserName()
pdcName = os.getenv('dcName', 'primaryDomainController')

try:
    user = GetObject("WinNT://%s/%s,user" % (pdcName, userid))
    fullName = user.FullName
    myGroups = _GetGroups(user)
except Exception, e:
    try:
        from win32net import NetUserGetGroups,NetUserGetInfo
        myGroups = []
        groups = NetUserGetGroups(pdcName,userid)
        userInfo = NetUserGetInfo(pdcName,userid,2)
        fullName = userInfo['full_name']
        for g in groups:
            myGroups.append(g[0])
    except Exception, e:
        fullname = "Unknown"
        myGroups = []

```

这比我们之前看到的任何一个脚本都要吓人，但实际上非常容易理解。首先，我们导入我们需要的模块或方法。接下来我们有一个简单的函数，它将一个用户对象作为唯一的参数。该函数将遍历该用户的组，并将它们添加到一个列表中，然后返回给调用者。难题的下一部分是获得主域控制器，我们使用 *os* 模块来完成。

我们将 pdcName 和 userid 传递给 GetObject(它是 win32com 模块的一部分)来获取我们的用户对象。如果工作正常，那么我们可以获得用户的全名和组。如果它失败了，那么我们捕获错误，并尝试从 win32net 模块中获取一些函数信息。如果这也失败了，那么我们就设置一些默认值。

## 包扎

希望你已经学到了一些有价值的技巧，可以用在你自己的代码中。我已经在 Python 2.4+中使用这些脚本好几年了，它们工作得非常好！