# Python 201 -什么是链图？

> 原文：<https://www.blog.pythonlibrary.org/2016/03/29/python-201-what-is-a-chainmap/>

一个 **ChainMap** 是来自集合模块的一个类，它提供了将多个映射链接在一起的能力，这样它们最终成为一个单一的单元。如果您查看文档，您会注意到它接受***映射**，这意味着一个链图将接受任意数量的映射或字典，并将它们转换成一个您可以更新的单一视图。让我们来看一个例子，这样您就可以了解这是如何工作的:

```py

>>> from collections import ChainMap
>>> car_parts = {'hood': 500, 'engine': 5000, 'front_door': 750}
>>> car_options = {'A/C': 1000, 'Turbo': 2500, 'rollbar': 300}
>>> car_accessories = {'cover': 100, 'hood_ornament': 150, 'seat_cover': 99}
>>> car_pricing = ChainMap(car_accessories, car_options, car_parts)
>>> car_pricing['hood']
500

```

这里，我们从集合模块中导入 **ChainMap** 。接下来，我们创建三个字典。然后，我们通过传入刚刚创建的三个字典来创建一个链图实例。最后，我们尝试访问链表中的一个键。当我们这样做的时候，链表将遍历每一个链表，以查看这个键是否存在并且有一个值。如果是这样，那么链表将返回它找到的第一个匹配那个键的值。

如果您想要设置默认值，这尤其有用。让我们假设我们想要创建一个有一些默认值的应用程序。应用程序还会知道操作系统的环境变量。如果有一个环境变量与我们在应用程序中默认的一个键相匹配，那么这个环境将覆盖我们的默认值。让我们进一步假设我们可以将参数传递给我们的应用程序。这些参数优先于环境和默认值。这是一个链图可以真正发光的地方。让我们看一个基于 Python 文档的简单例子:

```py

import argparse
import os

from collections import ChainMap

def main():
    app_defaults = {'username':'admin', 'password':'admin'}

    parser = argparse.ArgumentParser()
    parser.add_argument('-u', '--username')
    parser.add_argument('-p', '--password')
    args = parser.parse_args()
    command_line_arguments = {key:value for key, value 
                              in vars(args).items() if value}

    chain = ChainMap(command_line_arguments, os.environ, 
                     app_defaults)
    print(chain['username'])

if __name__ == '__main__':
    main()
    os.environ['username'] = 'test'
    main()

```

让我们稍微分解一下。这里我们导入 Python 的 **argparse** 模块和 **os** 模块。我们还导入了链图。接下来我们有一个简单的函数，它有一些愚蠢的默认值。我见过一些流行的路由器使用这些默认设置。然后，我们设置我们的参数解析器，并告诉它如何处理某些命令行选项。您会注意到，argparse 没有提供获取其参数的字典对象的方法，所以我们使用字典理解来提取我们需要的内容。这里另一个很酷的部分是 Python 内置的**变量**的使用。如果你在没有参数的情况下调用它，var 的行为会像 Python 的内置**局部变量**一样。但是如果你确实传入了一个对象，那么 vars 就等同于 object 的 **__dict__** 属性。

换句话说，**变量(参数)**等于**参数。__dict__** 。最后，通过传入我们的命令行参数(如果有的话)，然后传入环境变量，最后传入默认值来创建我们的链图。在代码的最后，我们尝试调用我们的函数，然后设置一个环境变量并再次调用它。试一试，你会看到它像预期的那样打印出**管理**和**测试**。现在让我们尝试用命令行参数调用该脚本:

```py

python chain_map.py -u mike

```

当我运行这个程序时，我得到了两次麦克的回复。这是因为我们的命令行参数覆盖了所有其他内容。我们设置环境并不重要，因为我们的链图会首先查看命令行参数。

* * *

### 包扎

现在你知道什么是链图，以及如何使用它。我发现它们非常有趣，我想我已经有了一两个用例，我希望很快就能实现。

* * *

### 相关阅读

*   关于[链图](https://docs.python.org/3/library/collections.html#chainmap-objects)的 Python 文档
*   PyMOTW:chain map "[搜索多个字典](https://pymotw.com/3/collections/chainmap.html)