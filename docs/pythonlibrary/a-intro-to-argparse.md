# argparse 简介

> 原文：<https://www.blog.pythonlibrary.org/2015/10/08/a-intro-to-argparse/>

你有没有想过如何在 Python 中处理命令行参数？是的，有一个模块。它叫做 **argparse** ，是对 **optparse** 的替代。在本文中，我们将快速浏览一下这个有用的模块。先说简单的吧！

### 入门指南

我一直发现解释一个编码概念最简单的方法就是展示一些代码。这就是我们要做的。这里有一个超级简单的例子，它没有做任何事情:

```py
>>> import argparse
>>> parser = argparse.ArgumentParser(
...         description="A simple argument parser",
...         epilog="This is where you might put example usage"
...     )
... 
>>> parser.print_help()
usage: _sandbox.py [-h]

A simple argument parser

optional arguments:
  -h, --help  show this help message and exit

This is where you might put example usage

```

这里我们只是导入 argparse，给它一个描述，并设置一个用法部分。这里的想法是，当你向你正在创建的程序寻求帮助时，它会告诉你如何使用它。在这种情况下，它打印出一个简单的描述、默认的可选参数(在这种情况下为“-h”)和示例用法。

现在让我们把这个例子变得更具体一些。毕竟，您通常不会从命令行解析参数。因此，我们将代码移动到 Python 文件内的 Python 函数中:

```py
# arg_demo.py

import argparse

#----------------------------------------------------------------------
def get_args():
    """"""
    parser = argparse.ArgumentParser(
        description="A simple argument parser",
        epilog="This is where you might put example usage"
    )
    return parser.parse_args()

if __name__ == '__main__':
    get_args()

```

现在让我们从命令行调用这个脚本:

```py
python arg_demo.py -h

```

这将打印出我们前面看到的帮助文本。现在让我们了解一下如何添加一些我们自己的自定义参数。

* * *

### 添加参数

让我们编写一些代码，添加我们的解析器可以理解的三个新参数。我们将添加一个必需的参数和两个非必需的参数。我们还将看看如何添加一个默认类型和一个必需类型。代码如下:

```py
# arg_demo2.py

import argparse

#----------------------------------------------------------------------
def get_args():
    """"""
    parser = argparse.ArgumentParser(
        description="A simple argument parser",
        epilog="This is where you might put example usage"
    )

    # required argument
    parser.add_argument('-x', action="store", required=True,
                        help='Help text for option X')
    # optional arguments
    parser.add_argument('-y', help='Help text for option Y', default=False)
    parser.add_argument('-z', help='Help text for option Z', type=int)
    print(parser.parse_args())

if __name__ == '__main__':
    get_args()

```

现在让我们运行几次，这样您就可以看到发生了什么:

```py
mike@pc:~/py/argsparsing$ python arg_demo2.py 
usage: arg_demo2.py [-h] -x X [-y Y] [-z Z]
arg_demo2.py: error: argument -x is required

mike@pc:~/py/argsparsing$ python arg_demo2.py -x something
Namespace(x='something', y=False, z=None)

mike@pc:~/py/argsparsing$ python arg_demo2.py -x something -y text
Namespace(x='something', y='text', z=None)

mike@pc:~/py/argsparsing$ python arg_demo2.py -x something -z text
usage: arg_demo2.py [-h] -x X [-y Y] [-z Z]
arg_demo2.py: error: argument -z: invalid int value: 'text'

mike@pc:~/py/argsparsing$ python arg_demo2.py -x something -z 10
Namespace(x='something', y=False, z=10)

```

正如您所看到的，如果您运行代码而不传递任何参数，您将得到一个错误。接下来，我们只传递所需的参数，这样您就可以看到其他两个参数的默认值。然后，我们尝试将“文本”传递给“-y”参数，它会被存储起来，所以我们知道它不需要布尔值。最后两个例子展示了当您向'-z '参数传递一个无效值和一个有效值时会发生什么。

顺便说一下，参数名称的长度不必是一个字符。你可以改变那些更具描述性的东西，如“arg1”或“simulator”或任何你想要的东西。

* * *

### 包扎

您现在知道了如何创建参数解析器的基础。您可能对该模块的许多其他方面感兴趣，例如为要保存的参数定义一个备用目标名称，使用不同的前缀(即使用“+”而不是“-”)，创建参数组等等。我建议查看文档(链接如下)了解更多细节。

* * *

### 附加阅读

*   本周 Python 模块: [argparse](https://pymotw.com/2/argparse/)
*   官方文件
*   stack overflow-Python arg parse:[用 nargs=argparse 组合可选参数。余数](http://stackoverflow.com/questions/18622798/python-argparse-combine-optional-parameters-with-nargs-argparse-remainder)