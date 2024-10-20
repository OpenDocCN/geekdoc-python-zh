# 获取 Windows 上的远程驱动器空间

> 原文：<https://www.blog.pythonlibrary.org/2010/09/09/getting-remote-drive-space-on-windows/>

在我目前的工作岗位上工作了大约一年后，当我们还在将最后几台 Windows 98 机器升级到 Windows XP 时，我们需要检查我们网络上哪些机器的磁盘空间越来越少。这个问题之所以会突然出现，是因为我们在几台有 10 GB 硬盘、几台有 20 GB 硬盘、一两台只有 4 GB 硬盘的机器上安装了 Windows XP。总之，在网上做了一些调查后，我发现 PyWin32 包可以完成我需要的功能。

以下是我使用的代码:

```py

import win32com.client as com

def TotalSize(drive):
    """ Return the TotalSize of a shared drive [GB]"""
    try:
        fso = com.Dispatch("Scripting.FileSystemObject")
        drv = fso.GetDrive(drive)
        return drv.TotalSize/2**30
    except:
        return 0

def FreeSpace(drive):
    """ Return the FreeSpace of a shared drive [GB]"""
    try:
        fso = com.Dispatch("Scripting.FileSystemObject")
        drv = fso.GetDrive(drive)
        return drv.FreeSpace/2**30
    except:
        return 0

workstations = ['computeNameOne']
print 'Hard drive sizes:'
for compName in workstations:
    drive = '\\\\' + compName + '\\c$'
    print '*************************************************\n'
    print compName
    print 'TotalSize of %s = %f GB' % (drive, TotalSize(drive))
    print 'FreeSpace on %s = %f GB' % (drive, FreeSpace(drive))
    print '*************************************************\n'

```

注意在底部，我使用了一个列表。通常，我会列出我想检查的每台计算机的名称。然后，我将遍历这些名称，并从一台拥有完全域管理权限的机器上将我需要的正确路径放在一起。

为了让它工作，我们需要导入 win32com.client 并调用下面的代码: **com。Dispatch("脚本。FileSystemObject")** 。这将为我们提供一个 COM 对象，我们可以通过查询来获得一个 disk 对象。一旦我们有了这些，我们可以问磁盘它有多少总空间和多少空闲空间。今天来看看这段代码，我会将这两个函数合并成一个函数，并返回一个元组。如您所见，我对结果做了一点数学运算，让它返回以千兆字节为单位的大小。

这就是全部了。简单的东西。我怀疑我没有完全写好这段代码，因为变量名太烂了。它可能来自于一个活动状态的食谱或者一个论坛，但是我忘了注明它的属性。如果你认识这个代码，请在评论中告诉我！