# Python 中如何使用 sys.argv？

> 原文：<https://www.pythonforbeginners.com/system/python-sys-argv>

Python 为我们提供了不同的编程方式。在本文中，我们将讨论 python 中的 sys.argv 列表及其使用示例。

## Python 中的 sys.argv 是什么？

sys argv 列表是包含 python 程序中命令行参数的列表。每当我们从命令行界面运行 python 程序时，我们可以向程序传递不同的参数。该程序将 python 文件的所有参数和文件名存储在 sys.argv 列表中。

sys argv 列表的第一个元素包含 python 文件的名称。第二个元素包含命令行参数。

您还可以找到某个程序的命令行参数的总数。为此，可以使用 len()函数找到 sys.argv 列表的长度。len()函数将列表作为其输入，并返回 sys argv 列表的长度。python 程序中的参数总数比 sys.argv 列表的长度少一个。

如果要使用命令行参数，可能需要使用 sys.argv。

要了解 sys.argv list 的更多信息，可以阅读这篇关于 python 中的[命令行参数的文章。](https://avidpython.com/python-basics/command-line-argument-using-sys-argv-in-python/)

## Sys.argv 示例

要使用 sys argv，首先必须导入 sys 模块。然后，您可以使用 sys argv 列表获得 python 文件的名称和命令行参数的值。

*   sys.argv 列表包含索引为 0 的 python 文件的名称。
*   它包含索引 1 处的第一个命令行参数。
*   第二个命令行参数出现在索引 2 处，依此类推。

您可以在下面的示例中观察到这一点。

```py
import sys
print "This is the name of the script: ", sys.argv[0]
print "Number of arguments: ", len(sys.argv)
print "The arguments are: " , str(sys.argv)
```

输出:

```py
This is the name of the script: sysargv.py
Number of arguments in: 1
The arguments are: ['sysargv.py']
```

如果我用额外的参数再次运行它，我将得到以下输出:

```py
This is the name of the script: sysargv.py
Number of arguments in: 3
The arguments are: ['sysargv.py', 'arg1', 'arg2']
```

## 结论

在本文中，我们讨论了 sys argv 列表及其在 python 中的使用。要了解更多关于 python 编程的知识，您可以阅读这篇关于在 python 中使用 with open 打开文件的文章。您可能也会喜欢这篇关于 Python 中的[字符串操作的文章。](https://www.pythonforbeginners.com/basics/string-manipulation-in-python)