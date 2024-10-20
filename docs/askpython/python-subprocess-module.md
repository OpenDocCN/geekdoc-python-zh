# Python 子流程模块

> 原文：<https://www.askpython.com/python-modules/python-subprocess-module>

读者朋友们，你们好！在本文中，我们将详细关注 **Python 子流程模块**。

所以，让我们开始吧！

* * *

## Python 中的子流程模块是什么？

在了解子流程模块的功能之前，让我们考虑以下情况

*通常情况下，我们用 Python 编写代码来自动化一个过程，或者通过一些 UI 表单或者任何与数据相关的表单来获得不需要人工干预的结果。但是，如果我们需要获得某些操作或功能的系统级信息，该怎么办呢？我们如何在 Python 中处理系统级脚本？*

这就是`Python subprocess module`出现的时候。

使用子流程模块，我们可以处理 Python 代码中的系统级脚本。它允许我们与输入建立联系。系统进程的输出以及错误管道和代码。

子流程模块包含在 Python 环境中运行系统级脚本的各种方法:

1.  **subprocess.call()**
2.  **subprocess.run()**
3.  **subprocess . check _ output()**
4.  **子流程。Popen()** 和**通信()**函数

现在让我们一个一个来看看吧！

* * *

## 1.subprocess.call()函数

为了使用与子流程模块相关的功能，我们需要将模块导入到 Python 环境中。

**subprocess.call()** 函数运行作为参数的命令，并返回代码执行成功与否的值。

看看下面的语法！

```py
subprocess.call(arguments, shell=True)

```

**例 01:**

在下面的例子中，我们试图通过 Python 脚本执行“echo Hello world”。

同样，对于 call()函数，第一个参数(echo)被视为可执行命令，第一个参数之后的参数被视为命令行参数。

此外，我们需要指定 **shell = True** ，以便参数被视为字符串。如果设置为 False，参数将被视为路径或文件路径。

```py
import subprocess
print(subprocess.call(["echo" , "Hello world"],shell=True))

```

**输出:**

如下所示，它返回 0 作为返回代码。代码返回的任何其他值表示命令运行不成功。

```py
"Hello world"
0

```

**例 02:**

在本例中，我们通过 python 脚本执行了命令“ls -l”。

```py
import subprocess
subprocess.call(["ls","-l"],shell=True)

```

**输出:**

```py
-rw-r--r-- 1 smulani 1049033 Feb 27 10:40 check.py

```

* * *

## 2.subprocess.run()命令

如上所示，call()函数只返回所执行命令的返回代码。它没有帮助我们检查输入和检查参数。

同样，我们有 **subprocess.run()** 函数帮助我们在 python 代码中执行 bash 或系统脚本，并返回命令的返回代码。

此外，它还返回传递给函数的参数。这样，它可以帮助我们验证系统脚本的输入。

**举例:**

```py
import subprocess
print(subprocess.run(["echo","Hello World"],shell=True))

```

**输出:**

```py
"Hello World"
CompletedProcess(args=['echo', 'Hello World'], returncode=0)

```

* * *

## 3.子流程。Popen()函数

`subprocess.Popen()`功能使我们能够在内部以全新的过程执行子程序。此外，这可以用于在 python 中执行 shell 命令。

**语法:**

```py
subprocess.Popen(arguments, stdout=subprocess.PIPE,stderr=subprocess.PIPE,shell=True)

```

*   stdout:命令的输出值
*   stderr:命令返回的错误

***推荐阅读——[Python stdin，stdout，stderr](https://www.askpython.com/python/python-stdin-stdout-stderr)***

**举例:**

```py
import subprocess
process = subprocess.Popen(
    ['echo', 'Hello World'],shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
stdout, stderr = process.communicate()
print(stdout)

```

这里，我们还使用了 **communicate()** 函数。这个函数帮助我们直接从流程中读取和获取脚本的输入、输出和错误值，如上所示。

**输出:**

```py
b'"Hello World"\r\n'

```

* * *

## 结论

到此，我们就结束了这个话题。如果你遇到任何问题，欢迎在下面评论。更多与 Python 相关的文章，敬请关注我们。

在那之前，学习愉快！！🙂