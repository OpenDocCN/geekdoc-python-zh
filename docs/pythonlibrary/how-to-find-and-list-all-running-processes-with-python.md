# 如何用 Python 找到并列出所有正在运行的进程

> 原文：<https://www.blog.pythonlibrary.org/2010/10/03/how-to-find-and-list-all-running-processes-with-python/>

前几天，我的任务是找到一种方法来获取 Windows XP 虚拟机上所有正在运行的进程的列表。我还应该包括每个进程使用了多少 CPU 和内存的信息。幸运的是，这不必是一个远程脚本，而是一个可以在客户机上运行的脚本。在到处搜索了一番后，我终于找到了解决办法。在这篇文章中，我们将看看一些拒绝以及最终的解决方案，这恰好是跨平台的工作。

我发现的第一批脚本之一是 2006 年 3 月的这个脚本:

```py

# http://mail.python.org/pipermail/python-win32/2006-March/004340.html
import win32com.client
wmi=win32com.client.GetObject('winmgmts:')
for p in wmi.InstancesOf('win32_process'):
    print p.Name, p.Properties_('ProcessId'), \
        int(p.Properties_('UserModeTime').Value)+int(p.Properties_('KernelModeTime').Value)
    children=wmi.ExecQuery('Select * from win32_process where ParentProcessId=%s' %p.Properties_('ProcessId'))
    for child in children:
        print '\t',child.Name,child.Properties_('ProcessId'), \
            int(child.Properties_('UserModeTime').Value)+int(child.Properties_('KernelModeTime').Value)

```

这个脚本需要 [PyWin32 包](http://sourceforge.net/projects/pywin32/)才能工作。然而，虽然这是一个方便的小脚本，但除了 ProcessId 之外，它没有显示我想要的任何内容。我并不真正关心用户或内核模式时间(即用户或内核的总 CPU 时间)。此外，我真的不喜欢使用 COM 的黑魔法，所以我最终拒绝了这个。

接下来是一个[活动状态配方](http://code.activestate.com/recipes/303339-getting-process-information-on-windows/)。看起来很有希望:

```py

# http://code.activestate.com/recipes/303339-getting-process-information-on-windows/
import win32pdh, string, win32api

def procids():
    #each instance is a process, you can have multiple processes w/same name
    junk, instances = win32pdh.EnumObjectItems(None,None,'process', win32pdh.PERF_DETAIL_WIZARD)
    proc_ids=[]
    proc_dict={}
    for instance in instances:
        if instance in proc_dict:
            proc_dict[instance] = proc_dict[instance] + 1
        else:
            proc_dict[instance]=0
    for instance, max_instances in proc_dict.items():
        for inum in xrange(max_instances+1):
            hq = win32pdh.OpenQuery() # initializes the query handle 
            path = win32pdh.MakeCounterPath( (None,'process',instance, None, inum,'ID Process') )
            counter_handle=win32pdh.AddCounter(hq, path) 
            win32pdh.CollectQueryData(hq) #collects data for the counter 
            type, val = win32pdh.GetFormattedCounterValue(counter_handle, win32pdh.PDH_FMT_LONG)
            proc_ids.append((instance,str(val)))
            win32pdh.CloseQuery(hq) 

    proc_ids.sort()
    return proc_ids

print procids()

```

唉，虽然这也从我的 Windows 系统中获得了一个进程列表(以及 PID)，但它没有给我任何关于 CPU 和内存利用率的信息。我认为如果我使用不同的计数器名称，这个方法可能会有效。我猜如果你愿意，你可以通过 MSDN 找到这些信息。我不想搞砸，所以我继续挖。

这个配方让我想到了下面这个基于 ctypes 的配方:

```py

# http://code.activestate.com/recipes/305279/

"""
Enumerates active processes as seen under windows Task Manager on Win NT/2k/XP using PSAPI.dll
(new api for processes) and using ctypes.Use it as you please.

Based on information from http://support.microsoft.com/default.aspx?scid=KB;EN-US;Q175030&ID=KB;EN-US;Q175030

By Eric Koome
email ekoome@yahoo.com
license GPL
"""
from ctypes import *

#PSAPI.DLL
psapi = windll.psapi
#Kernel32.DLL
kernel = windll.kernel32

def EnumProcesses():
    arr = c_ulong * 256
    lpidProcess= arr()
    cb = sizeof(lpidProcess)
    cbNeeded = c_ulong()
    hModule = c_ulong()
    count = c_ulong()
    modname = c_buffer(30)
    PROCESS_QUERY_INFORMATION = 0x0400
    PROCESS_VM_READ = 0x0010

    #Call Enumprocesses to get hold of process id's
    psapi.EnumProcesses(byref(lpidProcess),
                        cb,
                        byref(cbNeeded))

    #Number of processes returned
    nReturned = cbNeeded.value/sizeof(c_ulong())

    pidProcess = [i for i in lpidProcess][:nReturned]

    for pid in pidProcess:

        #Get handle to the process based on PID
        hProcess = kernel.OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ,
                                      False, pid)
        if hProcess:
            psapi.EnumProcessModules(hProcess, byref(hModule), sizeof(hModule), byref(count))
            psapi.GetModuleBaseNameA(hProcess, hModule.value, modname, sizeof(modname))
            print "".join([ i for i in modname if i != '\x00'])

            #-- Clean up
            for i in range(modname._length_):
                modname[i]='\x00'

            kernel.CloseHandle(hProcess)

if __name__ == '__main__':
    EnumProcesses()

```

这看起来很聪明，但是我很不擅长解析 ctypes。这是我想学的东西，但是我有一个截止日期，该死的！另外，这个程序只显示了正在运行的进程列表，但没有关于它们的信息。幸运的是，作者附上了参考文献，但我决定继续寻找。

接下来，我发现了一个关于使用蒂姆·戈登的便捷的 [WMI 模块](http://tgolden.sc.sabren.com/python/wmi/index.html)来做这类事情的帖子(下面是从该帖子直接复制的):

```py

# http://mail.python.org/pipermail/python-win32/2003-December/001482.html
>>> processes = WMI.InstancesOf('Win32_Process')
>>> len(processes)
41
>>> [process.Properties_('Name').Value for process in processes] # get
the process names
[u'System Idle Process', u'System', u'SMSS.EXE', u'CSRSS.EXE',
u'WINLOGON.EXE', u'SERVICES.EXE', u'LSASS.EXE', u'SVCHOST.EXE',
u'SVCHOST.EXE', u'SVCHOST.EXE', u'SVCHOST.EXE', u'SPOOLSV.EXE',
u'ati2evxx.exe', u'BAsfIpM.exe', u'defwatch.exe', u'inetinfo.exe',
u'mdm.exe', u'rtvscan.exe', u'SCARDSVR.EXE', u'WLTRYSVC.EXE',
u'BCMWLTRY.EXE', u'EXPLORER.EXE', u'Apoint.exe', u'carpserv.exe',
u'atiptaxx.exe', u'quickset.exe', u'DSentry.exe', u'Directcd.exe',
u'vptray.exe', u'ApntEx.exe', u'FaxCtrl.exe', u'digstream.exe',
u'CTFMON.EXE', u'wuauclt.exe', u'IEXPLORE.EXE', u'Pythonwin.exe',
u'MMC.EXE', u'OUTLOOK.EXE', u'LineMgr.exe', u'SAPISVR.EXE',
u'WMIPRVSE.EXE']

Here is how to get a single process and get its PID.

>>> p = WMI.ExecQuery('select * from Win32_Process where
Name="Pythonwin.exe"')
>>> [prop.Name for prop in p[0].Properties_] # let's look at all the
process property names
[u'Caption', u'CommandLine', u'CreationClassName', u'CreationDate',
u'CSCreationClassName', u'CSName', u'Description', u'ExecutablePath',
u'ExecutionState', u'Handle', u'HandleCount', u'InstallDate',
u'KernelModeTime', u'MaximumWorkingSetSize', u'MinimumWorkingSetSize',
u'Name', u'OSCreationClassName', u'OSName', u'OtherOperationCount',
u'OtherTransferCount', u'PageFaults', u'PageFileUsage',
u'ParentProcessId', u'PeakPageFileUsage', u'PeakVirtualSize',
u'PeakWorkingSetSize', u'Priority', u'PrivatePageCount', u'ProcessId',
u'QuotaNonPagedPoolUsage', u'QuotaPagedPoolUsage',
u'QuotaPeakNonPagedPoolUsage', u'QuotaPeakPagedPoolUsage',
u'ReadOperationCount', u'ReadTransferCount', u'SessionId', u'Status',
u'TerminationDate', u'ThreadCount', u'UserModeTime', u'VirtualSize',
u'WindowsVersion', u'WorkingSetSize', u'WriteOperationCount',
u'WriteTransferCount']
>>> p[0].Properties_('ProcessId').Value # get our ProcessId
928

```

这是一些很酷的东西，我在其他代码中使用了 Golden 的模块。然而，我仍然不确定使用哪个柜台来获取我的信息。我以为这些东西大部分都是为我编码的。好吧，结果是有一个软件包完全满足了我的需求，它可以在所有三个主要平台上运行！太神奇了！

## 跨平台解决方案！

这个包的名字是 [psutil](http://code.google.com/p/psutil/) ，这是我决定使用的。这是我最后得到的结果:

```py

import os
import psutil
import time

logPath = r'some\path\proclogs'
if not os.path.exists(logPath):
    os.mkdir(logPath)

separator = "-" * 80
format = "%7s %7s %12s %12s %30s, %s"
format2 = "%7.4f %7.2f %12s %12s %30s, %s"
while 1:
    procs = psutil.get_process_list()
    procs = sorted(procs, key=lambda proc: proc.name)

    logPath = r'some\path\proclogs\procLog%i.log' % int(time.time())
    f = open(logPath, 'w')
    f.write(separator + "\n")
    f.write(time.ctime() + "\n")
    f.write(format % ("%CPU", "%MEM", "VMS", "RSS", "NAME", "PATH"))
    f.write("\n")

    for proc in procs:
        cpu_percent = proc.get_cpu_percent()
        mem_percent = proc.get_memory_percent()
        rss, vms = proc.get_memory_info()
        rss = str(rss)
        vms = str(vms)
        name = proc.name
        path = proc.path
        f.write(format2 % (cpu_percent, mem_percent, vms, rss, name, path))
        f.write("\n\n")
    f.close()
    print "Finished log update!"
    time.sleep(300)
    print "writing new log data!"

```

是的，这是一个无限循环，是的，这通常是一件非常糟糕的事情(除了在 GUI 编程中)。然而，出于我的目的，我需要一种方法来每隔 5 分钟左右检查用户的进程，看看是什么导致机器行为如此怪异。因此，脚本需要永远运行，并将结果记录到唯一命名的文件中。这就是这个脚本所做的一切，还有一点格式化的魔力。你觉得合适就随意用或不用。

我希望这些资料对你有所帮助。希望它能让你省去我所有的挖掘工作！

*注意:虽然最后一个脚本似乎在 Windows XP 上工作得很好，但在 Windows 7 32 和 64 位上，您将得到一个“拒绝访问”的回溯，我怀疑这是由 Windows 7 增强的安全性引起的，但我会尝试找到一个解决方法。*

更新(10/09/2010)-psutil 的人不知道为什么它不能工作，但他们的一个开发者已经确认了这个问题。你可以关注他们的[谷歌群列表](http://groups.google.com/group/psutil/browse_frm/thread/ec8bf72fa18f79a2)。