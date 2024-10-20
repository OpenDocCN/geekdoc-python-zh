# 如何通过文件或外壳运行 Python 脚本

> 原文：<https://www.pythoncentral.io/execute-python-script-file-shell/>

如果你*不能执行或运行 Python 脚本*，那么编程就毫无意义。当您运行 Python 脚本时，解释器会将 Python 程序转换成计算机可以理解的东西。执行 Python 程序有两种方式:用 *shebang 行*调用 Python 解释器，以及使用交互式 Python shell。

## 将 Python 脚本作为文件运行

通常程序员编写独立的脚本，独立于真实环境。然后他们用一个。py”扩展名，它向操作系统和程序员表明该文件实际上是一个 Python 程序。在解释器被调用后，它读取并解释文件。Python 脚本在基于 Windows 和 Unix 的操作系统上运行的方式非常不同。我们将向您展示不同之处，以及如何在 Windows 和 Unix 平台上运行 Python 脚本。

### 使用命令提示符在 Windows 下运行 Python 脚本

Windows 用户必须将程序的路径作为参数传递给 Python 解释器。比如如下:

```py

C:\Python27\python.exe C:\Users\Username\Desktop\my_python_script.py

```

注意，您必须使用 Python 解释器的完整路径。如果你想简单地输入`python.exe C:\Users\Username\Desktop\my_python_script.py`，你必须将`python.exe`添加到你的`PATH`环境变量中。要做到这一点，请查看[将 Python 添加到路径环境](https://www.pythoncentral.io/add-python-to-path-python-is-not-recognized-as-an-internal-or-external-command/ "How to install Python 2.7 on Windows 7\. ‘python' is not recognized as an internal or external command.")的文章..

#### 窗口的 python.exe 对 pythonw.exe

注意，Windows 附带了两个 Python 可执行文件- `python.exe`和`pythonw.exe`。如果你想在运行脚本时弹出一个终端，使用`python.exe`，但是如果你不想弹出任何终端，使用`pythonw.exe`。`pythonw.exe`通常用于 GUI 程序，在这里你只想显示你的程序，而不是终端。

## 在 Mac、Linux、BSD、Unix 等下运行 Python 脚本

在像 Mac、BSD 或 Linux (Unix)这样的平台上，你可以在程序的第一行加上一个“shebang ”,表示 Python 解释器在硬盘上的位置。它的格式如下:

```py

#!/path/to/interpreter

```

用于 Python 解释器的常见 shebang 行如下:

```py

#!/usr/bin/env python

```

然后，您必须使用以下命令使脚本可执行:

```py

chmod +x my_python_script.py

```

与 Windows 不同，Python 解释器通常已经存在于`$PATH`环境变量中，因此没有必要添加它。

然后，您可以通过手动调用 Python 解释器来运行程序，如下所示:

```py

python firstprogram.py

```

## 用外壳执行 Python(实时解释器)

假设你已经安装了 Python 并且运行良好(如果你得到一个错误，[见这篇文章](https://www.pythoncentral.io/add-python-to-path-python-is-not-recognized-as-an-internal-or-external-command/ "How to install Python 2.7 on Windows 7\. ‘python' is not recognized as an internal or external command."))，打开终端或控制台，输入‘Python’并按下‘Enter’键。然后，您将立即被定向到 Python live 解释器。您的屏幕将显示如下信息:

```py

user@hostname:~ python

Python 3.3.0 (default, Nov 23 2012, 10:26:01)

[GCC 4.2.1 Compatible Apple Clang 4.1 ((tags/Apple/clang-421.11.66))] on darwin

Type "help", "copyright", "credits" or "license" for more information.

>>>

```

Python 程序员应该记住一件事:在使用实时解释器时，一切都是实时读取和解释的。例如，循环会立即迭代，除非它们是函数的一部分。所以需要一些心理规划。使用 Python shell 通常用于交互式执行代码。如果你想从解释器运行一个 Python 脚本，你必须`import`它或者调用 Python 可执行文件。