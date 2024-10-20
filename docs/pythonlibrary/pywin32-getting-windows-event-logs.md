# PyWin32:获取 Windows 事件日志

> 原文：<https://www.blog.pythonlibrary.org/2010/07/27/pywin32-getting-windows-event-logs/>

前几天，我关注的一个邮件列表上有一个关于访问 Windows 事件日志的帖子。我认为这是一个有趣的话题，所以我去寻找例子，并在 [ActiveState](http://docs.activestate.com/activepython/2.5/pywin32/Windows_NT_Eventlog.html) 上找到了一个非常好的例子。在这篇文章中，你会发现我的发现。

直接进入代码可能是最简单的。请注意，除了 Python 之外，您唯一需要的是 PyWin32 包。一旦你明白了这一点，你就可以继续下去了:

```py

import codecs
import os
import sys
import time
import traceback
import win32con
import win32evtlog
import win32evtlogutil
import winerror

#----------------------------------------------------------------------
def getAllEvents(server, logtypes, basePath):
    """
    """
    if not server:
        serverName = "localhost"
    else: 
        serverName = server
    for logtype in logtypes:
        path = os.path.join(basePath, "%s_%s_log.log" % (serverName, logtype))
        getEventLogs(server, logtype, path)

#----------------------------------------------------------------------
def getEventLogs(server, logtype, logPath):
    """
    Get the event logs from the specified machine according to the
    logtype (Example: Application) and save it to the appropriately
    named log file
    """
    print "Logging %s events" % logtype
    log = codecs.open(logPath, encoding='utf-8', mode='w')
    line_break = '-' * 80

    log.write("\n%s Log of %s Events\n" % (server, logtype))
    log.write("Created: %s\n\n" % time.ctime())
    log.write("\n" + line_break + "\n")
    hand = win32evtlog.OpenEventLog(server,logtype)
    total = win32evtlog.GetNumberOfEventLogRecords(hand)
    print "Total events in %s = %s" % (logtype, total)
    flags = win32evtlog.EVENTLOG_BACKWARDS_READ|win32evtlog.EVENTLOG_SEQUENTIAL_READ
    events = win32evtlog.ReadEventLog(hand,flags,0)
    evt_dict={win32con.EVENTLOG_AUDIT_FAILURE:'EVENTLOG_AUDIT_FAILURE',
              win32con.EVENTLOG_AUDIT_SUCCESS:'EVENTLOG_AUDIT_SUCCESS',
              win32con.EVENTLOG_INFORMATION_TYPE:'EVENTLOG_INFORMATION_TYPE',
              win32con.EVENTLOG_WARNING_TYPE:'EVENTLOG_WARNING_TYPE',
              win32con.EVENTLOG_ERROR_TYPE:'EVENTLOG_ERROR_TYPE'}

    try:
        events=1
        while events:
            events=win32evtlog.ReadEventLog(hand,flags,0)

            for ev_obj in events:
                the_time = ev_obj.TimeGenerated.Format() #'12/23/99 15:54:09'
                evt_id = str(winerror.HRESULT_CODE(ev_obj.EventID))
                computer = str(ev_obj.ComputerName)
                cat = ev_obj.EventCategory
        ##        seconds=date2sec(the_time)
                record = ev_obj.RecordNumber
                msg = win32evtlogutil.SafeFormatMessage(ev_obj, logtype)

                source = str(ev_obj.SourceName)
                if not ev_obj.EventType in evt_dict.keys():
                    evt_type = "unknown"
                else:
                    evt_type = str(evt_dict[ev_obj.EventType])
                log.write("Event Date/Time: %s\n" % the_time)
                log.write("Event ID / Type: %s / %s\n" % (evt_id, evt_type))
                log.write("Record #%s\n" % record)
                log.write("Source: %s\n\n" % source)
                log.write(msg)
                log.write("\n\n")
                log.write(line_break)
                log.write("\n\n")
    except:
        print traceback.print_exc(sys.exc_info())

    print "Log creation finished. Location of log is %s" % logPath

if __name__ == "__main__":
    server = None  # None = local machine
    logTypes = ["System", "Application", "Security"]
    getAllEvents(server, logTypes, "C:\downloads")

```

这种类型的脚本有几个潜在的警告。我以管理员的身份在我的电脑上测试了这段代码，并以域管理员的身份在工作中测试了这段代码。我没有作为任何其他类型的用户测试它。因此，如果您在运行这段代码时遇到问题，请检查您的权限。我在 Windows XP 和 Windows 7 上测试了这一点。在 Windows 7 上，UAC 似乎不会阻止这种活动，所以它和 XP 一样容易使用。然而，Windows 7 的事件在代码的消息部分有一些 unicode，而 XP 没有。注意这一点，并相应地处理它。

无论如何，让我们打开这个脚本，看看它是如何工作的。首先，我们有一些进口产品。我们使用*编解码器*模块以 utf-8 编码日志文件，以防消息中有一些狡猾的 unicode。我们使用 PyWin32 的 *win32evtlog* 模块打开事件日志并从中提取信息。根据我在开头提到的文章，要从日志中获取所有事件，需要调用 *win32evtlog。重复 ReadEventLog* 直到它停止返回事件。因此，我们使用 *while* 循环。在 while 循环中，我们使用一个 *for* 循环来遍历事件，并提取事件 ID、记录号、事件消息、事件源和一些其他的花絮。我们记录它，然后退出循环的*和*，同时*循环调用 *win32evtlog。再次读取事件日志*。*

我们使用 *traceback* 模块打印出脚本运行过程中出现的任何错误。这就是全部了！

## 包扎

如您所见，使用 PyWin32 包很容易。如果你卡住了，它有一些很棒的文档。如果文档不够好，你可以求助于 MSDN。PyWin32 是 Windows API 的轻量级包装，所以使用 MSDN 的指令相当简单。无论如何，我希望你学到了很多，并会发现它很有帮助。

## 进一步阅读

*   [Python 和 Unicode](http://docs.python.org/howto/unicode.html)
*   [PyWin32](http://sourceforge.net/projects/pywin32/)
*   来自 ActiveState 的 PyWin32 [文档](http://docs.activestate.com/activepython/2.5/pywin32/PyWin32.HTML)