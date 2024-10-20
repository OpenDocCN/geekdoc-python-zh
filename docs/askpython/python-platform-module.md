# Python 平台模块–快速介绍

> 原文：<https://www.askpython.com/python-modules/python-platform-module>

Python 有一个平台模块，包含处理代码运行平台的函数。在本教程中，我们将讨论该模块，并看看它的大部分有用的功能。

## 关于平台模块

平台模块用于检索关于系统或平台的信息。我们可以使用这个模块来执行兼容性检查。当我们有一个需要满足某些条件的 Python 程序时，例如，处理器的架构、使用的操作系统或系统拥有的 Python 版本，那么就可以使用这个模块。

这些规范用于确定 Python 代码在系统上运行的好坏。

不仅是为了兼容性检查，模块也可以为了兼容性检查而使用。我们有许多程序告诉我们我们的平台规范，任何用 Python 编写的这样的程序都可以使用这个模块。

该模块名为“platform ”,因此要在没有别名的情况下导入它，我们可以这样做:

```py
import platform

```

## 平台模块提供的功能

现在让我们从可用的函数开始。对于每个函数，示例都在 Linux 虚拟机上运行。

***读也——[Python OS 模块](https://www.askpython.com/python-modules/python-os-module-10-must-know-functions)***

### 1。平台架构

返回一个元组，其中包含位架构(处理器总线中的位数)和平台使用的处理器的链接格式。这两个值都以字符串形式返回。

```py
platform.architecture()

```

```py
('64bit', 'ELF')
```

### 2。机器类型

返回包含平台的机器类型(处理器中使用的寄存器的大小)的字符串。

```py
platform.machine()

```

```py
'x86_64'
```

### 3。网状名字

返回包含平台网络名称的字符串(如果平台在网络中，则为平台显示该名称)。

```py
platform.node()

```

```py
'sumeet-VirtualBox'
```

### 4。平台信息

返回包含有关基础平台的有用信息的单个字符串。该函数检索尽可能多的信息，然后返回人类可读的字符串，因此对于不同的平台，它可能看起来不同。

```py
platform.platform()

```

```py
'Linux-5.4.0-58-generic-x86_64-with-glibc2.29'
```

### 5。处理器名称

返回包含平台使用的处理器的实际名称的单个字符串。

```py
platform.processor()

```

```py
'Intel64 Family 6 Model 158 Stepping 10, GenuineIntel'
```

### 6。Python 构建

返回一个元组，其中包含平台上 Python 安装的编译号和编译日期。元组中的两个值都是字符串。

```py
platform.python_build()

```

```py
('default', 'Jan 27 2021 15:41:15')
```

### 7。Python 编译器

返回一个字符串，其中包含用于在平台上编译 Python 的编译器的名称。

```py
platform.python_compiler()

```

```py
'GCC 9.3.0'
```

### 8。Python 实现

返回一个字符串，其中包含有关平台上安装的 Python 的实现的信息。

```py
platform.python_implementation()

```

```py
'CPython'
```

### 9。Python 版本

返回标识平台上安装的 Python 版本的字符串。

该字符串的格式为“`major.minor.patchlevel`”。

```py
platform.python_version()

```

```py
'3.8.5'
```

### 10。Python 版本元组

以[元组](https://www.askpython.com/python/tuple/python-tuple)的形式返回平台上安装的 Python 版本。

元组的格式为“`(major, minor, patchlevel)`”。

```py
platform.python_version_tuple()

```

```py
('3', '8', '5')
```

### 11。操作系统版本

以[字符串](https://www.askpython.com/python/string/strings-in-python)的形式返回操作系统的发布信息。

```py
platform.release()

```

```py
'5.4.0-58-generic'
```

### 12\. OS Name

以字符串形式返回平台上操作系统的名称。

```py
platform.system()

```

```py
'Linux'
```

### 13。操作系统发布版本

以字符串形式返回平台上操作系统的发布版本。

```py
platform.version()

```

```py
'#64-Ubuntu SMP Wed Dec 9 08:16:25 UTC 2020'
```

### 14。平台信息元组

返回具有六个属性的命名元组:系统、节点、版本、版本、机器和处理器。所有这些属性都有各自的函数，所以这个函数可以用来获取我们从其他函数获得的所有信息。

```py
platform.uname()

```

```py
uname_result(system='Linux', node='sumeet-VirtualBox', release='5.4.0-58-generic', version='#64-Ubuntu SMP Wed Dec 9 08:16:25 UTC 2020', machine='x86_64', processor='Intel64 Family 6 Model 158 Stepping 10, GenuineIntel')
```

## 结论

在本教程中，我们学习了 python 中的平台模块。我们讨论了它的许多重要功能，并看到了它们的输出。

我希望你有一个很好的学习时间，并在下一个教程中看到你。