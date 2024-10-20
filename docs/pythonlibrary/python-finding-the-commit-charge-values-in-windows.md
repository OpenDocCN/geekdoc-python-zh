# Python:在 Windows 中查找提交费用值

> 原文：<https://www.blog.pythonlibrary.org/2010/03/05/python-finding-the-commit-charge-values-in-windows/>

本周，我的任务是设法找出我们虚拟工作站上的峰值提交值。原因是我们试图节省资金，并想知道我们是否分配了太多的内存。我们不需要总提交费用或限制提交费用值，但是由于我在研究中已经知道如何获得这些值，所以我也将展示如何获得这些值。

当我第一次开始搜索这个主题时，我尝试了诸如“python 峰值提交值”及其变体这样的搜索术语。这让我一无所获，所以我用“wmi”替换了“oython ”,并在 MSDN 上找到了[Win32 _ performatteddata _ perfs _ Memory](http://msdn.microsoft.com/en-us/library/aa394268%28VS.85%29.aspx)类。我以为这就是了，但它只给了我提交费用限额和总提交费用。下面是我如何使用蒂姆·戈登的 WMI 模块得到这些值的:

```py

import wmi

c = wmi.WMI()

for item in c.Win32_PerfFormattedData_PerfOS_Memory():
    commitChargeLimit = int(item.CommitLimit) / 1024

for item in c.Win32_PerfFormattedData_PerfOS_Memory():
    commitChargeTotal = int(item.CommittedBytes) / 1024

print "Commit Charge Limit: ", commitChargeLimit
print "Commit Charge Total: ", commitChargeTotal

```

这是很好的东西，显示了在 MSDN 获取文档并将其翻译成可用的 Python 语言是多么容易。不幸的是，它并没有给我所需要的信息。我的下一站是 PyWin32 邮件列表，Mark Hammond 在那里告诉我 win32pdh 和 win32pdhutil 模块。这些公开了性能计数器，但是我也找不到使用它来获取这些信息的方法。幸运的是，我在 [sysinternals 论坛](http://forum.sysinternals.com/forum_posts.asp?TID=15540&PID=75852)上找到了一个旧帖子，给了我一个线索。它是这样说的:

据我所知，获得这一细节的唯一方法是从 SYSTEM_PERFORMANCE_INFORMATION 结构的 uMmPeakCommitLimit 成员中获取，当使用 SystemPerformanceInformation 类型调用它时，会将该成员传递给 NtQuerySystemInformation。


我问哈蒙德先生这是否意味着我需要使用 [ctypes](http://python.net/crew/theller/ctypes/) ，因为 [NtQuerySystemInformation](http://msdn.microsoft.com/en-us/library/ms724509%28VS.85%29.aspx) 类没有被 PyWin32 公开，他说“可能”。ctypes 模块非常低级，除了从 ActiveState 复制脚本时，我没有用过它。这是一个非常方便的模块，在 2.5 版本中被添加到了[标准库](http://docs.python.org/library/ctypes.html)中。据我所知，它是由托马斯·海勒创作的。

反正 ctypes 有自己的[邮件列表](https://lists.sourceforge.net/lists/listinfo/ctypes-users)，所以我决定去那里试试。我收到了两个回复，其中一个是那个人本人(海勒)。他给了我一个剧本，开始看起来不太管用，但是和他反复讨论之后，他把我弄明白了。结果如下:

```py

from ctypes import *

SystemBasicInformation = 0
SystemPerformanceInformation = 2

class SYSTEM_BASIC_INFORMATION(Structure):
    _fields_ = [("Reserved1", c_long * 10),
                ("NumberOfProcessors", c_byte),
                ("bUnknown2", c_byte),
                ("bUnknown3", c_short)
                ]

class SYSTEM_PERFORMANCE_INFORMATION(Structure):
    _fields_ = [("IdleTime", c_int64),
                ("ReadTransferCount", c_int64),
                ("WriteTransferCount", c_int64),
                ("OtherTransferCount", c_int64),
                ("ReadOperationCount", c_ulong),
                ("WriteOperationCount", c_ulong),
                ("OtherOperationCount", c_ulong),
                ("AvailablePages", c_ulong),
                ("TotalCommittedPages", c_ulong),
                ("TotalCommitLimit", c_ulong),
                ("PeakCommitment", c_ulong),
                ("PageFaults", c_ulong),
                ("WriteCopyFaults", c_ulong),
                ("TransitionFaults", c_ulong),
                ("Reserved1", c_ulong),
                ("DemandZeroFaults", c_ulong),
                ("PagesRead", c_ulong),
                ("PageReadIos", c_ulong),
                ("Reserved2", c_ulong * 2),
                ("PagefilePagesWritten", c_ulong),
                ("PagefilePageWriteIos", c_ulong),
                ("MappedFilePagesWritten", c_ulong),
                ("MappedFilePageWriteIos", c_ulong),
                ("PagedPoolUsage", c_ulong),
                ("NonPagedPoolUsage", c_ulong),
                ("PagedPoolAllocs", c_ulong),
                ("PagedPoolFrees", c_ulong),
                ("NonPagedPoolAllocs", c_ulong),
                ("NonPagedPoolFrees", c_ulong),
                ("TotalFreeSystemPtes", c_ulong),
                ("SystemCodePage", c_ulong),
                ("TotalSystemDriverPages", c_ulong),
                ("TotalSystemCodePages", c_ulong),
                ("SmallNonPagedLookasideListAllocateHits", c_ulong),
                ("SmallPagedLookasideListAllocateHits", c_ulong),
                ("Reserved3", c_ulong),
                ("MmSystemCachePage", c_ulong),
                ("PagedPoolPage", c_ulong),
                ("SystemDriverPage", c_ulong),
                ("FastReadNoWait", c_ulong),
                ("FastReadWait", c_ulong),
                ("FastReadResourceMiss", c_ulong),
                ("FastReadNotPossible", c_ulong),
                ("FastMdlReadNoWait", c_ulong),
                ("FastMdlReadWait", c_ulong),
                ("FastMdlReadResourceMiss", c_ulong),
                ("FastMdlReadNotPossible", c_ulong),
                ("MapDataNoWait", c_ulong),
                ("MapDataWait", c_ulong),
                ("MapDataNoWaitMiss", c_ulong),
                ("MapDataWaitMiss", c_ulong),
                ("PinMappedDataCount", c_ulong),
                ("PinReadNoWait", c_ulong),
                ("PinReadWait", c_ulong),
                ("PinReadNoWaitMiss", c_ulong),
                ("PinReadWaitMiss", c_ulong),
                ("CopyReadNoWait", c_ulong),
                ("CopyReadWait", c_ulong),
                ("CopyReadNoWaitMiss", c_ulong),
                ("CopyReadWaitMiss", c_ulong),
                ("MdlReadNoWait", c_ulong),
                ("MdlReadWait", c_ulong),
                ("MdlReadNoWaitMiss", c_ulong),
                ("MdlReadWaitMiss", c_ulong),
                ("ReadAheadIos", c_ulong),
                ("LazyWriteIos", c_ulong),
                ("LazyWritePages", c_ulong),
                ("DataFlushes", c_ulong),
                ("DataPages", c_ulong),
                ("ContextSwitches", c_ulong),
                ("FirstLevelTbFills", c_ulong),
                ("SecondLevelTbFills", c_ulong),
                ("SystemCalls", c_ulong)]

sbi = SYSTEM_BASIC_INFORMATION()
retlen = c_ulong()

res = windll.ntdll.NtQuerySystemInformation(SystemBasicInformation,
                                            byref(sbi),
                                            sizeof(sbi),
                                            byref(retlen))
print res, retlen
print sbi.NumberOfProcessors

spi = SYSTEM_PERFORMANCE_INFORMATION()
retlen = c_ulong()

res = windll.ntdll.NtQuerySystemInformation(SystemPerformanceInformation,
                                            byref(spi),
                                            sizeof(spi),
                                            byref(retlen))
print res, retlen
print "Peak commit: ",
print spi.PeakCommitment * 4096 / 1024

```

我真的不明白这里发生的一切，但我很高兴它起作用了。嗯，我应该说它在 Windows XP Professional 上工作，32 位，Python 2.5。我在 64 位的 Windows 7 上也尝试过，当脚本运行时，它返回“0L”。我猜 64 位操作系统需要稍微不同的脚本，但是因为我们所有的工作站目前都使用 32 位，所以这一点并不重要。Python 社区再一次帮助了我，向我展示了他们有多棒！