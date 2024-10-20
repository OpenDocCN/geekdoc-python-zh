# 从 Python 列表中提取元素的 5 种简单方法

> 原文：<https://www.askpython.com/python/list/extract-elements-python-list>

让我们学习从 Python 列表中提取元素的不同方法当需要在 Python 中的单个变量中存储多个项目时，我们需要使用列表。它是 python 内置的数据函数之一。它是在初始化变量时使用[ ]括号创建的。

在本文中，我们将看到创建列表的不同方法，还将学习从 python 列表中提取元素的不同方法。

## 1.使用索引从 Python 列表中提取元素

在第一个例子中，我们创建了一个名为“firstgrid”的列表，其中包含 6 个元素。print 语句打印索引中的“1”元素。

```py
firstgrid=["A","B","C","D","E","F"]

print(firstgrid[1])

```

```py
Output: 'B'

```

## 2.使用枚举打印列表中的项目

这里，我们创建了一个名为“vara”的变量，并将元素填充到列表中。然后，我们使用“varx”变量指定枚举函数来搜索“1，2，5”索引位置。

```py
vara=["10","11","12","13","14","15"]

print([varx[1] for varx in enumerate(vara) if varx[0] in [1,2,5]])

```

```py
Output: ['11', '12', '15']

```

## 3.使用循环提取列表元素

还可以使用循环从 Python 列表中提取元素。让我们看看使用循环从列表中提取单个元素的 3 种方法。

#### 方法 1:

直接使用循环来搜索指定的索引。

```py
vara=["10","11","12","13","14","15"]

print([vara[i] for i in (1,2,5)])

```

```py
Output: ['11', '12', '15']

```

#### 方法二:

将列表和索引位置存储到两个不同的变量中，然后运行循环来搜索这些索引位置。

```py
elements = [10, 11, 12, 13, 14, 15]
indices = (1,1,2,1,5)

result_list = [elements[i] for i in indices]
print(result_list)

```

```py
Output: [11, 11, 12, 11, 15]

```

#### 方法三:

在这个例子中，我们使用了不同的方法来创建我们的列表。range 函数创建一个列表，其中包含从 10 到 15 的 6 个元素。

```py
numbers = range(10, 16)
indices = (1, 1, 2, 1, 5)

result = [numbers[i] for i in indices]
print(result)

```

```py
Output: [12, 11, 11, 14, 15]

```

## 4.使用 Numpy 查看列表中的项目

我们还可以使用流行的 NumPy 库来帮助我们从 Python 列表中提取元素。让我们看看如何使用两种不同的方法来实现这一点。

#### 方法 1:

这里，我们使用了 numpy import 函数，使用 np.array 库函数从列表“ax”中的元素打印变量“sx”中指定的索引。

```py
ax = [10, 11, 12, 13, 14, 15];
sx = [1, 2, 5] ;

import numpy as np
print(list(np.array(ax)[sx]))

```

```py
Output: [11, 12, 15]

```

#### 方法二:

这个例子使用一个变量存储索引位置，另一个变量存储数组中的数字。print 语句打印存储在变量“sx”中的索引位置，该位置与包含列表“ay”的变量相关。

```py
sx = [1, 2, 5];
ay = np.array([10, 11, 12, 13, 14, 15])
print(ay[sx])

```

```py
Output: [11 12 15]

```

## 5.使用索引函数提取元素

index 函数指定程序来搜索括号中提到的给定索引，然后运行一个循环来检查存在的索引。语句“0 <= index < len(vara)”告诉编译器只搜索 0 和由“len(variable)”指定的最后一个索引之间的索引位置。因此，尽管程序告诉编译器搜索 4 个索引位置，我们在输出中只能看到 3 个元素。循环丢弃任何超出给定范围的索引位置。

```py
vara=["10","11","12","13","14","15"]
print([vara[index] for index in (1,2,5,20) if 0 <= index < len(vara)])

```

```py
Output: ['13', '12', '14']

```

## 结论

本文详细解释了从 python 列表中搜索和提取元素的不同方法。在本文中，我们学习了列表是如何创建的，以及从列表中提取元素的不同类型的 python 函数。希望这篇文章能对你有所帮助。