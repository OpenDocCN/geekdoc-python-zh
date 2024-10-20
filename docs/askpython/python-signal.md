# Python 信号模块——什么是信号以及如何创建信号？

> 原文：<https://www.askpython.com/python-modules/python-signal>

在本文中，我们将看看如何使用 Python 信号模块。

当我们想用 Python 处理某些信号时，这个模块非常有用。

首先，我们来看看信号是什么意思。

* * *

## 什么是信号？

信号是程序从操作系统接收信息的一种方式。当操作系统接收到某些事件时，它可以以信号的形式传递给程序。

例如，当我们按下键盘上的按键`Ctrl + C`时，操作系统会产生一个信号，并将这个信号传递给程序。对于这个特定的组合，信号`SIGINT`被生成并传递给程序。

对于所有常见的操作系统，分配这些信号都有一个标准模式，通常是整数的简称。

在 Python 中，这些信号是在`signal`模块中定义的。

```py
import signal

```

要查看系统中的所有有效信号(取决于操作系统)，您可以使用`signal.valid_signals()`

```py
import signal

valid_signals = signal.valid_signals()

print(valid_signals)

```

**输出**

```py
{<Signals.SIGHUP: 1>, <Signals.SIGINT: 2>, <Signals.SIGQUIT: 3>, <Signals.SIGILL: 4>, <Signals.SIGTRAP: 5>, <Signals.SIGABRT: 6>, <Signals.SIGBUS: 7>, <Signals.SIGFPE: 8>, <Signals.SIGKILL: 9>, <Signals.SIGUSR1: 10>, <Signals.SIGSEGV: 11>, <Signals.SIGUSR2: 12>, <Signals.SIGPIPE: 13>, <Signals.SIGALRM: 14>, <Signals.SIGTERM: 15>, 16, <Signals.SIGCHLD: 17>, <Signals.SIGCONT: 18>, <Signals.SIGSTOP: 19>, <Signals.SIGTSTP: 20>, <Signals.SIGTTIN: 21>, <Signals.SIGTTOU: 22>, <Signals.SIGURG: 23>, <Signals.SIGXCPU: 24>, <Signals.SIGXFSZ: 25>, <Signals.SIGVTALRM: 26>, <Signals.SIGPROF: 27>, <Signals.SIGWINCH: 28>, <Signals.SIGIO: 29>, <Signals.SIGPWR: 30>, <Signals.SIGSYS: 31>, <Signals.SIGRTMIN: 34>, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, <Signals.SIGRTMAX: 64>}

```

现在，对于所有的信号，都有一些默认的操作，操作系统会分配给每个程序。

如果我们想有一些其他的行为，我们可以使用信号处理器！

* * *

## 什么是 Python 信号处理器？

信号处理器是一个用户定义的函数，可以处理 Python 信号。

如果我们取信号`SIGINT`(中断信号)，默认行为将是停止当前运行的程序。

但是，我们可以指定一个信号处理器来检测这个信号，并进行我们自己的定制处理！

让我们来看看如何做到这一点。

```py
import signal  
import time  

# Our signal handler
def signal_handler(signum, frame):  
    print("Signal Number:", signum, " Frame: ", frame)  

def exit_handler(signum, frame):
    print('Exiting....')
    exit(0)

# Register our signal handler with `SIGINT`(CTRL + C)
signal.signal(signal.SIGINT, signal_handler)

# Register the exit handler with `SIGTSTP` (Ctrl + Z)
signal.signal(signal.SIGTSTP, exit_handler)

# While Loop
while 1:  
    print("Press Ctrl + C") 
    time.sleep(3) 

```

这里，在我们运行程序之后，当我们按 Ctrl + C 时，程序将转到`signal_handler()`函数，因为我们已经用`SIGINT` (Ctrl + C)注册了处理程序。

我们还有另一个处理程序`exit_handler()`，如果我们按 Ctrl + Z，它将退出程序，并发送一个`SIGTSTP`信号。

让我们看看输出

**输出**

```py
Press Ctrl + C
^CSignal Number: 2  Frame:  <frame at 0x7fe62f033640, file 'python_signals.py', line 22, code <module>>
^ZExiting....

```

这里我按 Ctrl + C 转到`signal_handler()`功能，然后按 Ctrl + Z 退出程序。注意，还有一个 stack frame 对象(`frame`)，用于跟踪主程序的运行时堆栈。

* * *

## 使用报警信号

我们可以使用`SIGALARM`信号向我们的程序发送报警信号。让我们编写一个简单的信号处理程序来处理这个 Python 信号。

```py
import signal  
import time  

def alarm_handler(signum, frame):  
    print('Alarm at:', time.ctime())  

# Register the alarm signal with our handler
signal.signal(signal.SIGALRM, alarm_handler)

signal.alarm(3)  # Set the alarm after 3 seconds  

print('Current time:', time.ctime())  

time.sleep(6)  # Make a sufficient delay for the alarm to happen 

```

在最后一行，我们休眠了足够长的时间(6 秒钟)让警报信号传递给我们的程序。否则，既然程序早就终止了，信号就收不到了！

**输出**

```py
Current time: Thu Jul 23 00:41:40 2020
Alarm at: Thu Jul 23 00:41:43 2020

```

* * *

## 结论

在本文中，我们学习了如何使用`signal`模块设置信号处理器来处理各种信号。访问这里的链接，了解更多关于 [Python 模块](https://www.askpython.com/python-modules)的信息

## 参考

*   Python 信号模块[文档](https://docs.python.org/3/library/signal.html)
*   关于 Python 信号的 JournalDev 文章

* * *