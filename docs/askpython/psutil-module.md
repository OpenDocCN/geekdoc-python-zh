# 在 Python 中使用 psutil 模块

> 原文：<https://www.askpython.com/python-modules/psutil-module>

先说 Python 中的 psutil 模块。在处理系统进程和分析的大部分时间里，我们倾向于希望有一种替代方法来检查系统的任务管理器，以了解我们的测试会产生什么影响。

沿着手动检查任务管理器信息的路线，我们开始意识到我们在一遍又一遍地做同样的过程。

当然，这在我们自己的爱好者眼中并不好看。

也就是说，您需要创建一个脚本，该脚本可以遍历系统进程，并在您运行脚本时提供一个报告。

这就是`psutil`模块发挥作用的地方，它提供了在处理系统进程时非常重要的特性。

`psutil`是最适合用于**系统监控**、**剖析和限制进程资源**和**运行进程管理**的模块。

让我们开始使用它吧！

## 安装 psutil 模块

使用 [pip 系统](https://www.askpython.com/python-modules/python-pip)的`psutil`安装程序非常简单，

```py
pip install psutil

```

如果您正在使用任何其他软件包管理器，您可能需要查看它们的文档来安装 psutil。

既然我们已经设置好模块并准备好运行，我们可以开始替换检查任务管理器的需求了。

## 在 Python 脚本中使用 psutil 模块

`psutil`模块是一个包含很多方法的模块，这些方法可以分为几个部分，**系统**，**进程**， **windows 服务**，以及**常量**。还有一些独立的方法可以归入杂项类别。

这些方法有很多种，我们将在这些小节中只介绍其中的一部分，但是，在每个小节的小节中都会提供文档的链接。

### 1.系统信息

`psutil`为我们提供了各种各样的函数，我们可以使用这些函数来接收关于 [CPU](https://psutil.readthedocs.io/en/latest/#cpu) 、[内存](https://psutil.readthedocs.io/en/latest/#memory)、[磁盘](https://psutil.readthedocs.io/en/latest/#disks)、[网络](https://psutil.readthedocs.io/en/latest/#network)、[传感器](https://psutil.readthedocs.io/en/latest/#sensors)和[其他系统信息](https://psutil.readthedocs.io/en/latest/#other-system-info)的信息。

从每一部分中测试出来的几个函数将为我们提供下面的代码和输出。

```py
# Importing the module before utilization
import psutil

# Retrieving information regarding the CPU
## Returns the system CPU times as a named tuple
print(psutil.cpu_times())

## Returns the system-wide CPU utilization as a percentage
print(psutil.cpu_percent())

## Returns the number of logical CPUs in the system
print(psutil.cpu_count())

## Returns the various CPU statistics as a tuple
print(psutil.cpu_stats())

## Returns the CPU frequency as a nameduple
print(psutil.cpu_freq())

```

当这些函数被打印到控制台时，我们会收到以下类型的日志，

```py
scputimes(user=242962.0, system=84860.32812499994, idle=432883.46875, interrupt=5186.125, dpc=4331.65625)
0.0
4
scpustats(ctx_switches=2378452881, interrupts=1779121851, soft_interrupts=0, syscalls=3840585413)
scpufreq(current=2000.0, min=0.0, max=2601.0)

```

如果你想了解的话，我们已经编写了一个[要点](https://gist.github.com/dat-adi/1358ac672b21176eb044d1f1f5f4782c)来简单概述`psutil`模块的使用。

模块中有更多的功能，可以在官方文档中找到。

### 2.处理

通过 psutil 模块提供的函数允许 Python 检索整个系统中当前正在运行的进程的相关信息。

这些进程有特定的 PID 或进程 ID，可以从系统中检索，我们可以使用它们来了解特定进程及其统计信息。

使用模块的功能来处理流程，我们可以以简单的方式检索关于某些流程的信息，

```py
# Importing the module before utilization
import psutil

# Returning a sorted list of currently running processes
print(psutil.pids())

# Returns an iterator which prevents the race condition for process stats
print(psutil.process_iter())

# Used to check whether a certain process exists in the current process list
print(psutil.pid_exists(0))

# An example to terminate and wait for the children
def on_terminate(proc):
    print("process {} terminated with exit code {}".format(proc, proc.returncode))

procs = psutil.Process().children()
for p in procs:
    p.terminate()
gone, alive = psutil.wait_procs(procs, timeout=3, callback=on_terminate)
for p in alive:
    p.kill()

```

更多对异常处理有用的函数，以及 process 类的利用都在[文档](https://psutil.readthedocs.io/en/latest/#processes)中，如果你想了解可以使用的参数，这些函数值得一试。

### 3.Windows 服务

模块还为我们提供了检查所有已安装的 windows 服务的功能。

*WindowsService* 类是通过名称来表示每个 Windows 服务的类，关于服务的细节主要使用`win_service_iter()`和`win_service_get()`函数来检索。

```py
# Importing the module before utilization
import psutil

# Returns an iterator yielding a WindowsService class instance
print(psutil.win_service_iter())
# To provide the list of all processes contained we can use the list() function
print(list(psutil.win_service_iter()))

# Gets a Windows service by name, returning a WindowsService instance
print(psutil.win_service_get('WpnUserService_6b5d2'))

```

psutil 的[官方文档中深入介绍了使用 Windows 服务的方法，如果您希望深入了解这些功能的话。](https://psutil.readthedocs.io/en/latest/#windows-services)

### 4.系统常数

`psutil`模块允许检查系统常量，这些常量提供一个布尔响应，表明特定常量是否适用于您使用的操作系统。

为了解释这一点，我们可以使用 Python 进行测试，

```py
# Importing the module
import psutil

# Checking whether the operating system is Linux based
print(psutil.LINUX)

# Windows based OS?
print(psutil.WINDOWS)

```

在我的例子中，操作系统是基于 Windows 的，因此，对于 LINUX，响应是假的，对于 Windows 是真的。

更多的系统常数可以被访问和验证，以便进一步进行操作，使用和识别要遵循的操作系统指令，并且可以在[文档](https://psutil.readthedocs.io/en/latest/#constants)中找到。

## 结论

使用`psutil`模块非常简单，使用它的应用程序非常有用，可以显示**日志信息**，并处理**资源消耗**在您的系统中是如何发生的。

查看我们的其他文章，关于 Python 中可以让您的生活更轻松的不同模块——[数据帧](https://www.askpython.com/python-modules/pandas/python-pandas-module-tutorial)、 [XML 解析](https://www.askpython.com/python/examples/python-xml-parser)、[文件压缩](https://www.askpython.com/python-modules/gzip-module-in-python)。

## 参考

*   [官方 psutil 文档](https://psutil.readthedocs.io/en/latest/)
*   [psutil 的源代码](https://github.com/giampaolo/psutil)