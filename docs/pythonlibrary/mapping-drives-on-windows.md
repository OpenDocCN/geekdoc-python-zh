# 在 Windows 上映射驱动器

> 原文：<https://www.blog.pythonlibrary.org/2008/05/12/mapping-drives-on-windows/>

我必须帮助从 Kixtart 翻译到 Python 的第一批脚本之一是我们的地图驱动脚本。在其中，我们将根据用户所在的组和/或自定义注册表条目来映射驱动器。以下是 Kixtart 中每个类别的部分示例:

 `IF READVALUE("HKEY_LOCAL_MACHINE\SOFTWARE\MyOrg", "Office")= "officeName"
$Drive="g:" $Path="\\serverName\" + @userid Call "@lserver\\folderName"
ENDIF`

IF in group(" Dept XYZ ")
$ Drive = " g:" $ Path = " \ \ serverName \ "+@ userid Call " @ lserver \ \ folderName "
ELSE
ENDIF

现在，您会注意到这个脚本正在调用另一个名为“ConnectDrive”的脚本来进行实际的映射。基本上它包含了错误处理和下面几行:

 `:ConnectDrive`

使用$Drive /DELETE
使用$Drive $Path

现在让我们来看看我在 Python 中用什么来代替 Kixtart 代码。首先，我们需要获得用户所在的组。您会注意到下面有两种方法可以使用 PyWin32 包的不同部分来获得我们需要的组信息。

```py

from win32com.client import GetObject as _GetObject 

try:
    user = _GetObject("WinNT://%s/%s,user" % (pdcName, userid))
    fullName = user.FullName
    myGroups = _GetGroups(user)
except:
    try:
        from win32net import NetUserGetGroups,NetUserGetInfo
        myGroups = []
        groups = NetUserGetGroups(pdcName,userid)
        userInfo = NetUserGetInfo(pdcName,userid,2)
        fullName = userInfo['full_name']
        for g in groups:
            myGroups.append(g[0])
    except:
        fullname = "Unknown"
        myGroups = []

```

然后我们可以进行驱动器映射。请注意，我尝试取消任何已经映射到我想要映射到的驱动器号的映射。我这样做是因为我们有用户将插入 USB 驱动器，接管我的脚本映射的驱动器。有时这行得通，有时行不通。

```py

import subprocess
import win32wnet
from win32netcon import RESOURCETYPE_DISK as DISK

drive_mappings = []
if "Dept XYZ" in myGroups:
    drive_mappings.append(('V:', '\\\\ServerName\\folderName'))

for mapping in drive_mappings:
    try:
        # Try to disconnect anything that was previously mapped to that drive letter
        win32wnet.WNetCancelConnection2(mapping[0],1,0)
    except Exception, err:
        print 'Error mapping drive!'

    try:
        win32wnet.WNetAddConnection2(DISK, mapping[0], mapping[1])
    except Exception, err:
        if 'already in use' in err[2]:
            # change the drive letter since it's being mis-assigned
            subprocess.call(r'diskpart /s \\%s\path\to\log\change_g.txt' % pdcName)
            # try mapping again
            win32wnet.WNetAddConnection2(DISK, mapping[0], mapping[1])

```

这就是全部了。它最终比 Kixtart 更复杂一点，但是我认为我的代码可读性更好，我不必搜索多个文件来理解它。