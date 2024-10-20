# Python 201 -什么是 deque？

> 原文：<https://www.blog.pythonlibrary.org/2016/04/14/python-201-whats-a-deque/>

根据 Python 文档， **deques** “是栈和队列的概括”。它们读作“deck”，是“双端队列”的缩写。它们是 Python 列表的替代容器。Deques 是线程安全的，支持从 deques 的任何一端进行内存有效的追加和弹出。列表针对快速固定长度操作进行了优化。您可以在 Python 文档中获得所有血淋淋的细节。一个双队列接受一个 **maxlen** 参数，该参数设置双队列的边界。否则，deque 将增长到任意大小。当一个有界的 deque 满了，任何添加的新项将导致相同数量的项从另一端弹出。

一般来说，如果你需要快速追加或者快速弹出，使用一个队列。如果你需要快速随机访问，使用列表。让我们花一些时间来看看如何创建和使用 deque。

```py

>>> from collections import deque
>>> import string
>>> d = deque(string.ascii_lowercase)
>>> for letter in d:
...     print(letter)

```

这里，我们从集合模块中导入了 deque，还导入了**字符串**模块。为了实际创建一个 deque 的实例，我们需要向它传递一个 iterable。在本例中，我们传递给它 **string.ascii_lowercase** ，它返回字母表中所有小写字母的列表。最后，我们循环我们的 deque 并打印出每一项。现在让我们看看 deque 拥有的一些方法。

```py

>>> d.append('bork')
>>> d
deque(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 
       'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 
       'y', 'z', 'bork'])
>>> d.appendleft('test')
>>> d
deque(['test', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 
       'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 
       'v', 'w', 'x', 'y', 'z', 'bork'])
>>> d.rotate(1)
>>> d
deque(['bork', 'test', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 
       'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 
       't', 'u', 'v', 'w', 'x', 'y', 'z'])

```

让我们把它分解一下。首先，我们将一个字符串追加到队列的右端。然后，我们将另一个字符串追加到队列的左侧。最后，我们调用 deque 上的 **rotate** 并向其传递一个 1，这会使其向右旋转一次。换句话说，它导致一个项目从右端旋转到前端。你可以给它传递一个负数来使队列向左旋转。

让我们通过查看一个基于 Python 文档的示例来结束这一部分:

```py

from collections import deque

def get_last(filename, n=5):
    """
    Returns the last n lines from the file
    """
    try:
        with open(filename) as f:
            return deque(f, n)
    except OSError:
        print("Error opening file: {}".format(filename))
        raise

```

这段代码的工作方式与 Linux 的 **tail** 程序非常相似。在这里，我们向脚本传递一个文件名以及我们希望返回的 n 行。dequee 绑定到我们作为 n 传入的任何数字，这意味着一旦 dequee 满了，当新的行被读入并添加到 dequee 时，旧的行从另一端弹出并被丢弃。我还将文件开头的**with**语句包装在一个简单的异常处理程序中，因为它很容易传入格式错误的路径。例如，这将捕获不存在的文件。

### 包扎

现在你知道 Python 的 deque 的基础了。这是集合模块中又一个方便的小工具。虽然我个人从来不需要这个特殊的集合，但它仍然是一个有用的结构供其他人使用。我希望您会在自己的代码中发现它的一些重要用途。