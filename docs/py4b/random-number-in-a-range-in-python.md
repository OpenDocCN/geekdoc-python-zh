# Python 中某个范围内的随机数

> 原文：<https://www.pythonforbeginners.com/basics/random-number-in-a-range-in-python>

Python 为我们提供了[随机模块](https://www.pythonforbeginners.com/random/how-to-use-the-random-module-in-python)来在我们的程序中生成随机数。在本文中，我们将讨论在 python 中创建给定范围内的随机数的不同方法。

## 使用 randint()函数在一个范围内产生随机数

要在一个范围内创建一个随机数，我们可以使用 `randint()` 函数。`randint()` 函数将范围的下限和上限分别作为第一个和第二个输入参数。执行后，它返回一个介于下限和上限之间的随机数。这里，下限和上限是包括在内的，并且新生成的随机数可以等于上限或下限。您可以使用`randint()`功能创建一个范围内的随机数，如下所示。

```py
import random

upper_limit = 1000
lower_limit = 0
print("The lower limit of the range is:", lower_limit)
print("The upper limit of the range is:", upper_limit)
number = random.randint(lower_limit, upper_limit)
print("The random number is:", number)
```

输出:

```py
The lower limit of the range is: 0
The upper limit of the range is: 1000
The random number is: 739
```

## 使用 choice()函数在一个范围内的随机数

不使用`randint()` 函数，我们可以使用 random 模块中定义的`choice()`函数来创建一个随机数。为此，我们将首先使用`range()`函数创建一个数字序列。`range()`函数将序列的下限作为第一个参数，将序列的上限作为第二个参数，将序列中两个连续数字的差作为第三个可选参数。执行后，它返回一个包含数字序列的列表。

创建列表后，我们将把它作为输入参数传递给`choice()`函数。choice 函数将列表作为其输入参数，并从列表中返回一个随机数。这样，我们可以在给定的范围内创建一个随机数，如下所示。

```py
import random

upper_limit = 1000
lower_limit = 0
print("The lower limit of the range is:", lower_limit)
print("The upper limit of the range is:", upper_limit)
range_of_nums = range(lower_limit, upper_limit)
number = random.choice(range_of_nums)
print("The random number is:", number)
```

输出:

```py
The lower limit of the range is: 0
The upper limit of the range is: 1000
The random number is: 848
```

要一次获得多个随机数，可以使用`choices()` 功能。`choices()`功能与`choice()`功能完全相同。此外，它将所需随机数的计数作为第二个输入参数，并返回指定随机数的列表，如下所示。

```py
import random

upper_limit = 1000
lower_limit = 0
print("The lower limit of the range is:", lower_limit)
print("The upper limit of the range is:", upper_limit)
range_of_nums = range(lower_limit, upper_limit)
numbers = random.choices(range_of_nums, k=3)
print("The three random numbers are:", numbers)
```

输出:

```py
The lower limit of the range is: 0
The upper limit of the range is: 1000
The three random numbers are: [105, 858, 971]
```

## 使用 Numpy 模块

我们还可以使用 numpy 模块中定义的函数生成一个范围内的随机数。例如，您可以使用`numpy.random.randint()`函数代替`random.randint()`函数来生成一个随机数，如下所示。

```py
import numpy

upper_limit = 1000
lower_limit = 0
print("The lower limit of the range is:", lower_limit)
print("The upper limit of the range is:", upper_limit)
number = numpy.random.randint(lower_limit, upper_limit)
print("The random number is:", number)
```

输出:

```py
The lower limit of the range is: 0
The upper limit of the range is: 1000
The random number is: 144
```

类似地，您可以使用 `numpy.random.choice()` 函数在如下范围内生成随机数。

```py
import numpy

upper_limit = 1000
lower_limit = 0
print("The lower limit of the range is:", lower_limit)
print("The upper limit of the range is:", upper_limit)
range_of_nums = range(lower_limit, upper_limit)
number = numpy.random.choice(range_of_nums)
print("The random number is:", number)
```

输出:

```py
The lower limit of the range is: 0
The upper limit of the range is: 1000
The random number is: 264
```

## 脑震荡

在本文中，我们讨论了用 python 在一个范围内生成随机数的不同方法。我们还讨论了如何创建数字列表。要了解更多关于 python 中的列表，你可以阅读这篇关于 python 中的[列表理解](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python)的文章。你可能也会喜欢这篇关于 python 中的[字典理解](https://www.pythonforbeginners.com/dictionary/dictionary-comprehension-in-python)的文章。