# Python 201:列表理解

> 原文：<https://www.blog.pythonlibrary.org/2012/07/28/python-201-list-comprehensions/>

Python 中的列表理解非常方便。他们也可能有点难以理解你什么时候以及为什么要使用他们。列表理解往往比仅仅使用简单的**作为**循环更难阅读。我们将花一些时间来看看如何构建理解列表，并学习如何使用它们。到本文结束时，你应该有足够的能力自信地使用它们。

列表理解基本上是一行循环的**,它产生一个 Python 列表数据结构。这里有一个简单的例子:**

```py

x = [i for i in range(5)]

```

这将返回一个包含整数 0-4 的列表。如果您需要快速创建列表，这将非常有用。例如，假设您正在解析一个文件并寻找某个特定的内容。你可以使用列表理解作为一种过滤器:

```py

if [i for i in line if "SOME TERM" in i]:
    # do something

```

我曾使用类似的代码快速浏览文件，解析出文件的特定行或部分。当你将功能融入其中时，你可以开始做一些真正酷的事情。假设您想要对列表中的每个元素应用一个函数，例如当您需要将一串字符串转换为整数时:

```py

>>> x = ['1', '2', '3', '4', '5']
>>> y = [int(i) for i in x]
>>> y
[1, 2, 3, 4, 5]

```

这种事情出现的频率比你想象的要高。我还不得不循环遍历一系列字符串并调用一个字符串方法，比如 strip on 它们，因为它们有各种各样的以空格结尾的前导:

```py

myStrings = [s.strip() for s in myStringList]

```

也有需要创建嵌套列表理解的情况。这样做的一个原因是将多个列表合并成一个。这个例子来自 Python [文档](http://docs.python.org/tutorial/datastructures.html):

```py

>>> vec = [[1,2,3], [4,5,6], [7,8,9]]
>>> [num for elem in vec for num in elem]
[1, 2, 3, 4, 5, 6, 7, 8, 9]

```

该文档还展示了其他几个关于嵌套列表理解的有趣例子。强烈推荐看一看！至此，您应该能够在自己的代码中使用列表理解，并很好地使用它们。只要发挥你的想象力，你就会发现很多你也可以利用它们的好地方。

### 进一步阅读

*   人教版 202 - [列表理解](http://www.python.org/dev/peps/pep-0202/)
*   [列表理解简介](http://carlgroner.me/Python/2011/11/09/An-Introduction-to-List-Comprehensions-in-Python.html)
*   另[列表理解教程](http://www.bogotobogo.com/python/python_list_comprehension.php)
*   [重载 Python 列表理解](http://blog.sigfpe.com/2012/03/overloading-python-list-comprehension.html)
*   [大家好，我叫杰西，我是滥用列表理解者](http://jessenoller.com/2008/03/28/hi-my-name-is-jesse-and-i-abuse-list-comprehensions/)