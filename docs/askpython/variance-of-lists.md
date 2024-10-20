# 使用 Python 计算列表方差的 3 种简单方法

> 原文：<https://www.askpython.com/python/examples/variance-of-lists>

在本教程中，我们将看看 Python 中计算列表方差的各种方法。

* * *

## Python 中计算列表方差的方法

在我们深入研究计算列表方差的方法之前，让我们了解一下方差的含义。

方差是统计学中至关重要的数学工具。它是用来处理海量数据的。对于给定的数据集合，它是标准差的平方。

方差也称为分布的第二中心矩。平方的平均值减去给定数据集的平均值的平方用于计算它。

***变量(X) = E[(X- )2]***

* * *

## 方法 1: Python 方差函数

Python 包含一个计算列表方差的内置函数。下面提供了语法及其参数的解释。

```py
variance( [data], mean )

```

[数据]:提供要计算方差的数据列表。
mean:非强制参数。它使用实际平均值。

* * *

## 方法 2:基本方法

这是计算列表方差的最基本的方法。在下面的例子中，我们使用上述公式计算平均值，然后计算方差。

我们在这里没有使用任何内置的方法，而是通过构建公式来手动计算列表的方差。

```py
l = [1, 2, 3, 4, 5, 6] 
print("The original list is : " + str(l)) 
length= int(len(l))
mean = sum(l) / length 
ans = sum((i - mean) ** 2 for i in l) / length
print("The variance of list is : " + str(ans)) 

```

代码的输出如下所示:

```py
The original list is : [1, 2, 3, 4, 5, 6]
The variance of list is : 2.9166666666666665

```

* * *

## 方法 3:使用统计模块

在这个例子中，我们使用了内置函数 variance()，这使得计算列表的方差变得非常容易。

```py
import statistics  
list1 = [1, 2, 3, 4, 5, 6] 
print("The original list is : " + str(list1)) 
ans = statistics.variance(list1) 
print("The variance of list is : " + str(ans)) 

```

代码的输出如下所示:

```py
The original list is : [1, 2, 3, 4, 5, 6]
The variance of list is : 3.5

```

* * *

## 结论

恭喜你！您刚刚学习了使用 Python 编程语言计算列表方差的三种不同方法。希望你喜欢它！😇

喜欢这个教程吗？我建议你看看下面提到的教程:

1.  [用 Python 计算 a^n:用 Python 计算功率的不同方法](https://www.askpython.com/python/examples/compute-raised-to-power)
2.  [Python 中如何计算距离？【简易分步指南】](https://www.askpython.com/python/examples/compute-distances-in-python)

感谢您抽出时间！希望你学到了新的东西！！😄

* * *