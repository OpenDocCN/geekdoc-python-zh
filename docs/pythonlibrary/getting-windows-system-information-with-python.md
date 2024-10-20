# 使用 Python 获取 Windows 系统信息

> 原文：<https://www.blog.pythonlibrary.org/2010/01/27/getting-windows-system-information-with-python/>

我不得不为我的雇主想出的另一个脚本处理关于我们每个用户的物理机器的各种信息。我们希望跟踪他们的 CPU 速度、硬盘大小和 RAM 容量(以及其他信息),这样我们就能知道什么时候该升级他们的电脑了。从互联网上的各个地方收集所有的碎片是一件非常痛苦的事情，所以为了省去你的麻烦，我将把我发现的贴出来。请注意，这些代码中有很多是从 ActiveState 或邮件列表上的各种食谱中提取的。以下大部分内容几乎一字不差地出现在这份食谱中。

出于我不明白的原因，我们想知道用户在他们的机器上安装了哪个 Windows 操作系统。当我开始的时候，我们混合了 Windows 98 和 Windows XP Professional，后者超过了 90%。不管怎样，下面是我们用来判断用户正在运行什么的脚本:

```py

def get_registry_value(key, subkey, value):
    import _winreg
    key = getattr(_winreg, key)
    handle = _winreg.OpenKey(key, subkey)
    (value, type) = _winreg.QueryValueEx(handle, value)
    return value

def os_version():
    def get(key):
        return get_registry_value(
            "HKEY_LOCAL_MACHINE", 
            "SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion",
            key)
    os = get("ProductName")
    sp = get("CSDVersion")
    build = get("CurrentBuildNumber")
    return "%s %s (build %s)" % (os, sp, build)

```

正确运行这段代码的方法是调用 *os_version* 函数。这将使用两个不同的键调用嵌套的 *get* 函数两次，以获取操作系统和服务包。从嵌套函数中调用了 *get_registry_value* 两次，我们也直接调用它来获取构建信息。最后，我们使用字符串替换来组合一个代表 PC 操作系统版本的字符串。

下面的代码片段用于找出客户端电脑中的处理器:

```py

def cpu():
    try:
        cputype = get_registry_value(
            "HKEY_LOCAL_MACHINE", 
            "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0",
            "ProcessorNameString")
    except:
        import wmi, pythoncom
        pythoncom.CoInitialize() 
        c = wmi.WMI()
        for i in c.Win32_Processor ():
            cputype = i.Name
        pythoncom.CoUninitialize()

    if cputype == 'AMD Athlon(tm)':
        c = wmi.WMI()
        for i in c.Win32_Processor ():
            cpuspeed = i.MaxClockSpeed
        cputype = 'AMD Athlon(tm) %.2f Ghz' % (cpuspeed / 1000.0)
    elif cputype == 'AMD Athlon(tm) Processor':
        import wmi
        c = wmi.WMI()
        for i in c.Win32_Processor ():
            cpuspeed = i.MaxClockSpeed
        cputype = 'AMD Athlon(tm) %s' % cpuspeed
    else:
        pass
    return cputype

```

注意，如果你想继续下去，你需要下载蒂姆·戈尔登的 [WMI (Windows 管理规范)模块](http://timgolden.me.uk/python/wmi/index.html)，我认为它依赖于 [PyWin32 包](http://sourceforge.net/projects/pywin32/)(即使不是，在后面的例子中你也需要那个包)。请注意，我们首先尝试使用在前面的示例中定义的函数从 Windows 注册表中获取 cpu 类型。如果失败了，我们就用 WMI 模块来得到它。这样做的主要原因是在注册表中查找比拨打 WMI 电话要快。还要注意，我们将 WMI 代码包装在 pythoncom 方法中。我记得，之所以有这些，是因为我们在一个线程中运行这个脚本，你需要初始化的东西，让 WMI 高兴。否则，你会以一些愚蠢的错误或崩溃而告终。如果你不想用线，那么我想你可以去掉那些线。

这个片段中的下一部分是一个条件，它检查返回哪种 cpu 类型。如果它是一个 AMD 处理器，那么我们做另一个 WMI 调用来获得处理器的时钟速度，并将其添加到 CPU 字符串中。否则，我们只是按原样返回字符串(这通常意味着安装了 Intel 芯片)。

我们使用 VNC 连接到我们的大多数机器，所以我们想知道机器的名称。为此，我们运行以下命令:

```py

from platform import node

def compname():
    try:
        return get_registry_value(
            "HKEY_LOCAL_MACHINE",
            'SYSTEM\\ControlSet001\\Control\\ComputerName\\ComputerName',
            'ComputerName')
    except:
        compName = node
        return compName

```

同样，我们首先检查 Windows 注册表，看看我们想要的东西是否存放在那里。如果失败，那么我们使用 Python 的*平台*模块。如您所见，我们使用了 bare except 语句，这是我们的糟糕设计。我必须解决这个问题。在我的辩护中，我在成为 Python 程序员的第一年就做了大部分这样的东西，我真的不知道还有什么更好的。无论如何，还有另一种方法可以得到这些信息，那就是 WMI:

```py

c = wmi.WMI()
for i in c.Win32_ComputerSystem():
    compname = i.Name

```

这不是很简单吗？我真的很喜欢这个东西，虽然我不知道为什么我们必须使用一个循环来获取 WMI 的信息。如果你知道，请在下面的评论中给我留言。

下一个话题是网络浏览器。在我工作的地方，除了 Internet Explorer，我们还在所有的机器上安装了 Mozilla Firefox。奇怪的是，我们有许多供应商没有升级他们的软件或网站，使其在 Internet Explorer 8 上正常运行，所以当微软推出更新时，我们有时会遇到问题。这为我们介绍了下一段代码:

```py

def firefox_version():
    try:
        version = get_registry_value(
            "HKEY_LOCAL_MACHINE", 
            "SOFTWARE\\Mozilla\\Mozilla Firefox",
            "CurrentVersion")
        version = (u"Mozilla Firefox", version)
    except WindowsError:
        version = None
    return version

def iexplore_version():
    try:
        version = get_registry_value(
            "HKEY_LOCAL_MACHINE", 
            "SOFTWARE\\Microsoft\\Internet Explorer",
            "Version")
        version = (u"Internet Explorer", version)
    except WindowsError:
        version = None
    return version

def browsers():
    browsers = []
    firefox = firefox_version()
    if firefox:
        browsers.append(firefox)
    iexplore = iexplore_version()
    if iexplore:
        browsers.append(iexplore)

    return browsers

```

正确的运行方式是调用*浏览器*函数。它将调用另外两个函数，这两个函数继续重用我们从 Windows 注册表中获取所需信息的 *get_registry_value* 。这件作品有点无聊，但很容易使用。

我们的下一个脚本将获得大约安装的 RAM:

```py

import ctypes

def ram():
    kernel32 = ctypes.windll.kernel32
    c_ulong = ctypes.c_ulong
    class MEMORYSTATUS(ctypes.Structure):
        _fields_ = [
            ('dwLength', c_ulong),
            ('dwMemoryLoad', c_ulong),
            ('dwTotalPhys', c_ulong),
            ('dwAvailPhys', c_ulong),
            ('dwTotalPageFile', c_ulong),
            ('dwAvailPageFile', c_ulong),
            ('dwTotalVirtual', c_ulong),
            ('dwAvailVirtual', c_ulong)
        ]

    memoryStatus = MEMORYSTATUS()
    memoryStatus.dwLength = ctypes.sizeof(MEMORYSTATUS)
    kernel32.GlobalMemoryStatus(ctypes.byref(memoryStatus))
    mem = memoryStatus.dwTotalPhys / (1024*1024)
    availRam = memoryStatus.dwAvailPhys / (1024*1024)
    if mem >= 1000:
        mem = mem/1000
        totalRam = str(mem) + ' GB'
    else:
#        mem = mem/1000000
        totalRam = str(mem) + ' MB'
    return (totalRam, availRam)

```

当我最初写这篇文章时，我使用了下面的 WMI 方法:

```py

c = wmi.WMI()
for i in c.Win32_ComputerSystem():
    mem = int(i.TotalPhysicalMemory)

```

它后面的条件是我的，但它们的主要用途是使返回的数字更有意义。我还没有完全理解 ctypes 是如何工作的，但是可以说上面的代码比 WMI 方法在更低的层次上获得信息。可能也会快一点。

最后一部分是获取硬盘空间，这是通过以下方式完成的:

```py

def _disk_c(self):
    drive = unicode(os.getenv("SystemDrive"))
    freeuser = ctypes.c_int64()
    total = ctypes.c_int64()
    free = ctypes.c_int64()
    ctypes.windll.kernel32.GetDiskFreeSpaceExW(drive, 
                                    ctypes.byref(freeuser), 
                                    ctypes.byref(total), 
                                    ctypes.byref(free))
    return freeuser.value

```

这是原始配方作者使用 ctypes 的另一个例子。因为我不想听起来像个白痴，所以我不会假装理解得足够好来解释它。然而，Tim Golden 展示了一个类似的使用 WMI 查找硬盘剩余空闲空间的方法，您可能会感兴趣。去他的[网站](http://timgolden.me.uk/python/wmi/cookbook.html#show-the-percentage-free-space-for-each-fixed-disk)看看就知道了。

我希望这些都有意义。如果你有任何关于它的问题让我知道，我将尽力回答。