# Python 101 -使用 Python 启动子流程

> 原文：<https://www.blog.pythonlibrary.org/2020/06/30/python-101-launching-subprocesses-with-python/>

有时候，当你正在编写一个应用程序，你需要运行另一个应用程序。例如，出于某种原因，您可能需要在 Windows 上打开 Microsoft 记事本。或者如果你在 Linux 上，你可能想运行 **grep** 。Python 支持通过`subprocess`模块启动外部应用程序。

从 Python 2.4 开始，`subprocess`模块就是 Python 的一部分。在此之前，您需要使用`os`模块。你会发现`subprocess`模块功能强大，使用简单。

在本文中，您将学习如何使用:

*   `subprocess.run()`功能
*   `subprocess.Popen()`类
*   `subprocess.Popen.communicate()`功能
*   用`stdin`和`stdout`读写

我们开始吧！

### `subprocess.run()`功能

在 **Python 3.5** 中增加了`run()`函数。`run()`功能是使用`subprocess`的推荐方法。

查看函数的定义通常有助于更好地理解它的工作原理:

```py
subprocess.run(args, *, stdin=None, input=None, stdout=None, stderr=None,
    capture_output=False, shell=False, cwd=None, timeout=None, check=False, 
    encoding=None, errors=None, text=None, env=None, universal_newlines=None)
```

你不需要知道所有这些参数是如何有效地使用`run()`的。事实上，大多数时候您可能只需要知道第一个参数是什么以及是否启用`shell`就可以了。其余的论点对于非常具体的用例是有帮助的。

让我们试着运行一个常见的 Linux / Mac 命令，`ls`。`ls`命令用于列出目录中的文件。默认情况下，它会列出您当前所在目录中的文件。

要使用`subprocess`运行它，您需要执行以下操作:

```py
>>> import subprocess
>>> subprocess.run(['ls'])
filename
CompletedProcess(args=['ls'], returncode=0)
```

您还可以设置`shell=True`，它将通过 shell 本身运行命令。大多数情况下，您不需要这样做，但是如果您需要对进程进行更多的控制，并且想要访问 shell 管道和通配符，这可能会很有用。

但是，如果您希望保留命令的输出，以便以后使用，该怎么办呢？让我们看看你接下来会怎么做！

### 获取输出

通常，您会希望从外部流程获得输出，然后对该数据进行处理。要从`run()`获得输出，您可以将`capture_output`参数设置为 True:

```py
>>> subprocess.run(['ls', '-l'], capture_output=True)
CompletedProcess(args=['ls', '-l'], returncode=0, 
    stdout=b'total 40\n-rw-r--r--@ 1 michael  staff  17083 Apr 15 13:17 some_file\n', 
    stderr=b'')
```

现在这并没有太大的帮助，因为您没有将返回的输出保存到变量中。继续更新代码，这样你就可以访问`stdout`。

```py
 >>> output = subprocess.run(['ls', '-l'], capture_output=True)
>>> output.stdout
b'total 40\n-rw-r--r--@ 1 michael  staff  17083 Apr 15 13:17 some_file\n'
```

`output`是一个`CompletedProcess`类实例，它允许您访问传入的`args`、`returncode`以及`stdout`和`stderr`。

一会儿你就会了解到`returncode`。`stderr`是大多数程序打印错误信息的地方，而`stdout`是用来显示信息性消息的。

如果您感兴趣，可以研究一下这段代码，看看当前这些属性中有什么:

```py
output = subprocess.run(['ls', '-l'], capture_output=True)
print(output.returncode)
print(output.stdout)
print(out.stderr)
```

让我们继续，了解下一个`Popen`。

### `subprocess.Popen()`类

自从添加了`subprocess`模块之后，`subprocess.Popen()`类就已经存在了。在 Python 3 中已经更新了几次。如果你有兴趣了解这些变化，你可以在这里阅读:

*   [https://docs . python . org/3/library/subprocess . html # popen-constructor](https://docs.python.org/3/library/subprocess.html#popen-constructor)

你可以把`Popen`看成是`run()`的低级版本。如果你有一个`run()`无法处理的异常用例，那么你应该使用`Popen`来代替。

现在，让我们看看如何使用`Popen`运行上一节中的命令:

```py
>>> import subprocess
>>> subprocess.Popen(['ls', '-l'])
<subprocess.Popen object at 0x10f88bdf0>
>>> total 40
-rw-r--r--@ 1 michael  staff  17083 Apr 15 13:17 some_file

>>>
```

语法几乎相同，除了您使用的是`Popen`而不是`run()`。

以下是从外部进程获取返回代码的方式:

```py
>>> process = subprocess.Popen(['ls', '-l'])
>>> total 40
-rw-r--r--@ 1 michael  staff  17083 Apr 15 13:17 some_file

>>> return_code = process.wait()
>>> return_code
0
>>>
```

`0`的一个`return_code`表示程序成功完成。如果你打开一个有用户界面的程序，比如微软记事本，你需要切换回你的 REPL 或者空闲会话来添加`process.wait()`行。这样做的原因是记事本会出现在程序的顶部。

如果您没有将`process.wait()`调用添加到您的脚本中，那么您将无法在手动关闭您可能已经通过`subprocess`启动的任何用户界面程序后捕获返回代码。

您可以使用您的`process`句柄通过`pid`属性访问进程 id。您也可以通过调用`process.kill()`来终止(SIGKILL)进程，或者通过`process.terminate()`来终止(SIGTERM)进程。

### `subprocess.Popen.communicate()`功能

有时候，您需要与自己创建的进程进行交流。您可以使用`Popen.communicate()`方法向流程发送数据以及提取数据。

对于本节，您将只使用`communicate()`来提取数据。让我们使用`communicate()`来获得使用`ifconfig`命令的信息，您可以使用它来获得关于 Linux 或 Mac 上的计算机网卡的信息。在 Windows 上，你可以使用`ipconfig`。请注意，根据您的操作系统，此命令中有一个字母的差异。

代码如下:

```py
>>> import subprocess
>>> cmd = ['ifconfig']
>>> process = subprocess.Popen(cmd, 
                               stdout=subprocess.PIPE,
                               encoding='utf-8')
>>> data = process.communicate()
>>> print(data[0])
lo0: flags=8049<UP,LOOPBACK,RUNNING,MULTICAST> mtu 16384
    options=1203<RXCSUM,TXCSUM,TXSTATUS,SW_TIMESTAMP>
    inet 127.0.0.1 netmask 0xff000000 
    inet6 ::1 prefixlen 128 
    inet6 fe80::1%lo0 prefixlen 64 scopeid 0x1 
    nd6 options=201<PERFORMNUD,DAD>
gif0: flags=8010<POINTOPOINT,MULTICAST> mtu 1280
stf0: flags=0<> mtu 1280
XHC20: flags=0<> mtu 0
# -------- truncated --------
```

这段代码的设置与上一段略有不同。让我们更详细地检查一下每一部分。

首先要注意的是，您将`stdout`参数设置为一个`subprocess.PIPE`。这允许您捕获该流程发送给`stdout`的任何内容。你还设置了`encoding`为`utf-8`。这样做的原因是为了使输出更容易阅读，因为默认情况下`subprocess.Popen`调用返回字节而不是字符串。

下一步是调用`communicate()`，它将从流程中捕获数据并返回。`communicate()`方法同时返回`stdout`和`stderr`，所以你将得到一个`tuple`。你在这里没有捕捉到`stderr`，所以那将是`None`。

最后，你打印出数据。该字符串相当长，所以输出在这里被截断。

让我们继续学习如何使用`subprocess`进行读写！

### 用`stdin`和`stdout`读写

让我们假设您今天的任务是编写一个 Python 程序，该程序检查您的 Linux 服务器上当前运行的进程，并打印出使用 Python 运行的进程。

您可以使用`ps -ef`获得当前正在运行的进程列表。通常情况下，您会使用该命令，并将其“传输”到另一个 Linux 命令行实用程序`grep`，用于搜索字符串文件。

以下是您可以使用的完整 Linux 命令:

```py
ps -ef | grep python
```

但是，您希望使用`subprocess`模块将该命令翻译成 Python。

有一种方法可以做到这一点:

```py
import subprocess

cmd = ['ps', '-ef']
ps = subprocess.Popen(cmd, stdout=subprocess.PIPE)

cmd = ['grep', 'python']
grep = subprocess.Popen(cmd, stdin=ps.stdout, stdout=subprocess.PIPE,
                        encoding='utf-8')

ps.stdout.close()
output, _ = grep.communicate()
python_processes = output.split('\n')
print(python_processes)
```

这段代码重新创建了`ps -ef`命令，并使用`subprocess.Popen`来调用它。使用`subprocess.PIPE`捕获命令的输出。然后您还创建了`grep`命令。

对于`grep`命令，您将其`stdin`设置为`ps`命令的输出。您还捕获了`grep`命令的`stdout`，并像以前一样将编码设置为`utf-8`。

这有效地从`ps`命令和“管道”中获得输出，或者将其输入到`grep`命令中。接下来，您`close()`了`ps`命令的`stdout`，并使用`grep`命令的`communicate()`方法从`grep`获得输出。

最后，在新行(`\n`)上分割输出，得到一个字符串的`list`，它应该是所有活动 Python 进程的列表。如果您现在没有运行任何活动的 Python 进程，那么输出将是一个空列表。

你总是可以自己运行`ps -ef`，找到除了`python`之外的其他东西来搜索，并尝试这样做。

### 包扎

`subprocess`模块是非常通用的，它为您提供了一个丰富的接口来处理外部进程。

在本文中，您了解了:

*   `subprocess.run()`功能
*   `subprocess.Popen()`类
*   `subprocess.Popen.communicate()`功能
*   用`stdin`和`stdout`读写

`subprocess`模块的内容比这里介绍的要多。然而，你现在应该能够正确使用`subprocess`了。来吧，试一试！