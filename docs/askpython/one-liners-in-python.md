# Python 中有用的一行程序——快速指南

> 原文：<https://www.askpython.com/python/examples/one-liners-in-python>

Python 是有史以来最强大和用户友好的编程语言之一。Python 在程序员中很受欢迎，因为它简化了复杂的任务。

本教程将介绍一些简单有趣的一行程序。我们开始吧！

* * *

## 把两本词典的内容合并成一本。

如果你使用的是 **Python3.9 或更高版本的**，你可以使用|来实现。

```py
x = {'a': 11, 'b': 22}
y = {'c': 13, 'd': 44}
z = x | y
print(z)

```

输出如下所示:

```py
{'a': 11, 'b': 22, 'c': 13, 'd': 44}

```

## 获取最常用的元素

让我们利用集合模块中的 **`most_common()`** 函数来实现这一点。

```py
from collections import Counter
l = ['1', 'b', '2', 'a', '3', 'z', '3', 'a', '2', '3']
Counter(l).most_common()[0][0]

```

代码返回**‘3’**作为正确的输出！

## 同时得到商和余数

`divmod()`返回一个元组，其功能源于它结合了模百分比和除法/运算符。

```py
Q, R = divmod(35632, 5)
print("Quo. - ",Q)
print("Rem. - ",R)

```

```py
Quo. -  7126
Rem. -  2

```

## 找出前 n 个斐波那契数列

这将是在 Python 中记忆 lambda 函数和递归的极好练习。

```py
fib = lambda x: x if x <= 1 else fib(x - 1) + fib(x - 2)
print(fib(20))
print(fib(5))

```

```py
6765
5

```

## 从列表中删除重复元素

```py
list(set(['1', '1', '2', '1', '3']))

```

在 Python 中，集合中的每个元素都是唯一的，因此不会有重复。

```py
['1', '3', '2']

```

* * *

## 结论

恭喜你！您刚刚学习了 Python 编程语言中 5 个有用的一行程序。希望你喜欢它！😇

喜欢这个教程吗？无论如何，我建议你看一下下面提到的教程:

1.  [Python 中更容易调试的技巧](https://www.askpython.com/python/tricks-for-easier-debugging-in-python)
2.  [在大学编程作业中取得好成绩的最佳技巧](https://www.askpython.com/python/best-tips-to-score-well-in-college-programming-assignment)
3.  [3 个 Matplotlib 绘图技巧使绘图有效](https://www.askpython.com/python-modules/matplotlib/matplotlib-plotting-tips)
4.  Python 中的竞争性编程:你需要知道什么？

感谢您抽出时间！希望你学到了新的东西！！😄

* * *