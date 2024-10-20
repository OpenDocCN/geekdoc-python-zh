# PyWin32 -如何监控打印队列

> 原文：<https://www.blog.pythonlibrary.org/2013/12/19/pywin32-monitor-print-queue/>

前几天，我试图找出一种方法来监控 Windows 上的打印队列。手头的任务是记录哪些文档被成功地送到了打印机。这个想法是，当打印完成时，文档将被存档。要做这类事情，你需要 [PyWin32](http://sourceforge.net/projects/pywin32/) (又名:Python for Windows extensions)。在本文中，我们将查看一个检查打印队列的简单脚本。

代码如下:

```py

import time
import win32print

#----------------------------------------------------------------------
def print_job_checker():
    """
    Prints out all jobs in the print queue every 5 seconds
    """
    jobs = [1]
    while jobs:
        jobs = []
        for p in win32print.EnumPrinters(win32print.PRINTER_ENUM_LOCAL,
                                         None, 1):
            flags, desc, name, comment = p

            phandle = win32print.OpenPrinter(name)
            print_jobs = win32print.EnumJobs(phandle, 0, -1, 1)
            if print_jobs:
                jobs.extend(list(print_jobs))
            for job in print_jobs:
                print "printer name => " + name
                document = job["pDocument"]
                print "Document name => " + document
            win32print.ClosePrinter(phandle)

        time.sleep(5)
    print "No more jobs!"

#----------------------------------------------------------------------
if __name__ == "__main__":
    print_job_checker()

```

首先我们导入 **win32print** 和时间模块。我们需要 win32print 来访问打印机。我们创建了一个潜在的无限循环来检查打印队列中的作业。如果作业列表为空，这意味着打印队列中没有任何内容，该功能将退出。在上面的代码中，我们使用了 **win32print。enum prits()**循环检查安装在机器上的打印机。第一个参数是一个标志(win32print。PRINTER_ENUM_LOCAL)，第二个是名称(或者本例中没有名称)，第三个是信息级别。我们可以使用几个标志，比如 PRINTER_ENUM_SHARED、PRINTER_ENUM_LOCAL 或 PRINTER_ENUM_CONNECTIONS。我使用 PRINTER_ENUM_LOCAL，因为它返回了一个打印机名，其格式可以与 win32print 的 **OpenPrinter** 方法一起使用。我们这样做是为了通过句柄向打印机查询信息。为了获得打印作业的列表，我们调用 win32print。带有打印句柄的 EnumJobs()。然后我们遍历它们，打印出打印机名和文档名。你不需要做所有这些打印，但我发现它在我写代码的时候很有帮助。出于测试目的，我建议打开打印队列并将其设置为“暂停”,这样您就可以在准备好之前阻止打印纸。这使您仍然可以将项目添加到可以查询的打印队列中。

我在检查之间放置了 5 秒钟的延迟，以确保队列中没有新的更新。如果有打印作业，它会再次检查队列。否则它会跳出循环。

希望你会发现这段代码对你正在做的事情有用。尽情享受吧！

### 附加说明

*   蒂姆·戈尔登的文档 [win32print](http://timgolden.me.uk/pywin32-docs/win32print.html)
*   一个关于 Win32 的老线程打印在 [PyWin32 邮件列表上](https://mail.python.org/pipermail/python-win32/2003-November/001416.html)