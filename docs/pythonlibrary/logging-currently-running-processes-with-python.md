# 使用 Python 记录当前运行的进程

> 原文：<https://www.blog.pythonlibrary.org/2014/10/21/logging-currently-running-processes-with-python/>

我查看了我的一些旧代码，注意到这个旧脚本，其中我每 5 分钟创建一个所有运行进程的日志。我相信我最初编写代码是为了帮助我诊断正在消耗内存或占用 CPU 的流氓进程。我正在使用 [psutil 项目](https://code.google.com/p/psutil/)来获取我需要的信息，所以如果你想继续，你也需要下载并安装它。

代码如下:

```py

import os
import psutil
import time

#----------------------------------------------------------------------
def create_process_logs(log_dir):
    """
    Create a log of all the currently running processes
    """
    if not os.path.exists(log_dir):
        try:
            os.mkdir(log_dir)
        except:
            pass

    separator = "-" * 80
    col_format = "%7s %7s %12s %12s %30s"
    data_format = "%7.4f %7.2f %12s %12s %30s"
    while 1:
        procs = psutil.get_process_list()
        procs = sorted(procs, key=lambda proc: proc.name)

        log_path = os.path.join(log_dir, "procLog%i.log" % int(time.time()))
        f = open(log_path, 'w')
        f.write(separator + "\n")
        f.write(time.ctime() + "\n")
        f.write(col_format % ("%CPU", "%MEM", "VMS", "RSS", "NAME"))
        f.write("\n")

        for proc in procs:
            cpu_percent = proc.get_cpu_percent()
            mem_percent = proc.get_memory_percent()
            rss, vms = proc.get_memory_info()
            rss = str(rss)
            vms = str(vms)
            name = proc.name
            f.write(data_format % (cpu_percent, mem_percent, vms, rss, name))
            f.write("\n\n")
        f.close()
        print "Finished log update!"
        time.sleep(300)
        print "writing new log data!"

if __name__ == "__main__":
    log_dir = r"c:\users\USERNAME\documents"
    create_process_logs(log_dir)

```

让我们把它分解一下。这里我们传入一个日志目录，检查它是否存在，如果不存在就创建它。接下来，我们设置几个包含日志文件格式的变量。然后我们开始一个无限循环，使用 **psutil** 获取所有当前正在运行的进程。我们还按名称对流程进行分类。接下来，我们打开一个唯一命名的日志文件，写出每个进程的 CPU 和内存使用情况，以及它的虚拟机、RSS 和可执行文件的名称。然后，我们关闭文件，等待 5 分钟，然后再从头开始。

回想起来，将这些信息写入 SQLite 这样的数据库可能会更好，这样数据就可以被搜索和图形化。与此同时，希望你能在这里找到一些有用的信息，可以用于你自己的项目。