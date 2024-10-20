# Python 101:如何使子流程超时

> 原文：<https://www.blog.pythonlibrary.org/2016/05/17/python-101-how-to-timeout-a-subprocess/>

有一天，我遇到了一个用例，我需要与一个已经启动的子流程进行通信，但是我需要它超时。不幸的是，Python 2 没有办法让 communicate 方法调用超时，所以它只是阻塞，直到它返回或者进程本身关闭。我在 [StackOverflow](http://stackoverflow.com/questions/1191374/using-module-subprocess-with-timeout) 上发现了许多不同的方法，但我认为我最喜欢的是使用 Python 的线程模块的 Timer 类:

```py

import subprocess

from threading import Timer

kill = lambda process: process.kill()
cmd = ['ping', 'www.google.com']
ping = subprocess.Popen(
    cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

my_timer = Timer(5, kill, [ping])

try:
    my_timer.start()
    stdout, stderr = ping.communicate()
finally:
    my_timer.cancel()

```

这个特殊的例子并不完全符合我遇到的用例，但是很接近。基本上，我们这里有一个长期运行的流程，我们希望与之进行交互。在 Linux 上，如果您调用 ping，它将无限期运行。所以这是一个很好的例子。在这种情况下，我们编写 killing lambda 来调用进程的 kill 方法。然后，我们启动 ping 命令，将它放入一个计时器中，该计时器设置为 5 秒后到期，然后启动计时器。当进程运行时，我们收集它的 stdout 和 stderr，然后进程终止。最后，我们通过停止计时器来清理。

Python 3.5 增加了接受超时参数的 **run** 函数。根据[文档](https://docs.python.org/3/library/subprocess.html)，它将被传递给子流程的 communicate 方法，如果流程超时，将引发 TimeoutExpired 异常。让我们来试试:

```py

>>> import subprocess
>>> cmd = ['ping', 'www.google.com']
>>> subprocess.run(cmd, timeout=5)
PING www.google.com (216.58.216.196) 56(84) bytes of data.
64 bytes from ord31s21-in-f4.1e100.net (216.58.216.196): icmp_seq=1 ttl=55 time=16.3 ms
64 bytes from ord31s21-in-f4.1e100.net (216.58.216.196): icmp_seq=2 ttl=55 time=19.4 ms
64 bytes from ord31s21-in-f4.1e100.net (216.58.216.196): icmp_seq=3 ttl=55 time=20.0 ms
64 bytes from ord31s21-in-f4.1e100.net (216.58.216.196): icmp_seq=4 ttl=55 time=19.4 ms
64 bytes from ord31s21-in-f4.1e100.net (216.58.216.196): icmp_seq=5 ttl=55 time=17.0 ms
Traceback (most recent call last):
  Python Shell, prompt 3, line 1
  File "/usr/local/lib/python3.5/subprocess.py", line 711, in run
    stderr=stderr)
subprocess.TimeoutExpired: Command '['ping', 'www.google.com']' timed out after 5 seconds

```

很明显它是按照记载的方式工作的。为了真正有用，我们可能希望将子流程调用包装在一个异常处理程序中:

```py

>>> try:
...     subprocess.run(cmd, timeout=5)
... except subprocess.TimeoutExpired:
...     print('process ran too long')
... 
PING www.google.com (216.58.216.196) 56(84) bytes of data.
64 bytes from ord31s21-in-f196.1e100.net (216.58.216.196): icmp_seq=1 ttl=55 time=18.3 ms
64 bytes from ord31s21-in-f196.1e100.net (216.58.216.196): icmp_seq=2 ttl=55 time=21.1 ms
64 bytes from ord31s21-in-f196.1e100.net (216.58.216.196): icmp_seq=3 ttl=55 time=22.7 ms
64 bytes from ord31s21-in-f196.1e100.net (216.58.216.196): icmp_seq=4 ttl=55 time=20.3 ms
64 bytes from ord31s21-in-f196.1e100.net (216.58.216.196): icmp_seq=5 ttl=55 time=16.8 ms
process ran too long

```

既然我们可以捕捉异常，我们可以继续做其他事情或保存错误异常。有趣的是，在 Python 3.3 中，超时参数被添加到了子流程模块中。您可以在 subprocess.call、check_output 和 check_call 中使用它。在 Popen.wait()中也有。