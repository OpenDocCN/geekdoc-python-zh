# 在 Python 中从列表中选择随机元素

> 原文：<https://www.pythonforbeginners.com/basics/select-random-element-from-a-list-in-python>

在用 python 编程时，我们可能需要在几种情况下从列表中选择一个随机元素。在本文中，我们将讨论在 python 中从列表中选择元素的不同方法。

## 使用随机模块从列表中选择随机元素

python 中的`random`模块为我们提供了不同的函数来生成随机数。我们也可以使用这个模块中定义的函数从列表中选择随机元素。

为了在 python 中从列表中选择一个随机元素，我们可以使用在`random`模块中定义的`choice()`函数。`choice()`函数将一个列表作为输入，并在每次执行时从列表中返回一个随机元素。

您可以在下面的示例中观察到这一点。

```py
import random

myList = [1, 2, 3, 45, 6, 8, 78, 23, 56, 7686, 123]
print("The list is:")
print(myList)
random_element = random.choice(myList)
print("The randomly selected element is:", random_element)
```

输出:

```py
The list is:
[1, 2, 3, 45, 6, 8, 78, 23, 56, 7686, 123]
The randomly selected element is: 8
```

## 使用秘密模块从列表中选择随机元素

`secrets`模块用于生成适用于管理数据(如密码、帐户认证、安全令牌和相关机密)的加密强随机数。然而，我们也可以使用这个模块从列表中选择一个随机元素。

secrets 模块中定义的`choice()`函数与 random 模块中定义的`choice()`函数工作方式相同。它接受一个列表作为输入，并从列表中返回一个元素，如下所示。

```py
import secrets

myList = [1, 2, 3, 45, 6, 8, 78, 23, 56, 7686, 123]
print("The list is:")
print(myList)
random_element = secrets.choice(myList)
print("The randomly selected element is:", random_element)
```

输出:

```py
The list is:
[1, 2, 3, 45, 6, 8, 78, 23, 56, 7686, 123]
The randomly selected element is: 45
```

## 使用 numpy 模块

我们也可以使用来自`numpy`模块的 choice()函数从列表中选择一个随机元素。`numpy`模块中的`choice()`功能与`random`模块或`secrets`模块的工作方式相同。您可以在下面的示例中观察到这一点。

```py
import numpy

myList = [1, 2, 3, 45, 6, 8, 78, 23, 56, 7686, 123]
print("The list is:")
print(myList)
random_element = numpy.random.choice(myList)
print("The randomly selected element is:", random_element)
```

输出:

```py
The list is:
[1, 2, 3, 45, 6, 8, 78, 23, 56, 7686, 123]
The randomly selected element is: 3
```

使用`numpy`模块时，我们有一个好处，我们甚至可以从列表中随机选择多个选项。为此，我们将使用函数的“`size`”参数。如果我们想从给定的列表中选择`n`随机元素，我们将把数字 n 作为第二个输入参数传递给在`numpy`模块中定义的`choice()`函数。执行后，该函数返回如下所示的 `n`元素列表。

```py
import numpy

myList = [1, 2, 3, 45, 6, 8, 78, 23, 56, 7686, 123]
print("The list is:")
print(myList)
random_elements = numpy.random.choice(myList, 4)
print("The randomly selected elements are:")
for x in random_elements:
    print(x)
```

输出:

```py
The list is:
[1, 2, 3, 45, 6, 8, 78, 23, 56, 7686, 123]
The randomly selected elements are:
78
3
6
23
```

在这里，您可以观察到在输出列表中可以多次选择一个元素。为了避免这种情况，我们将使用第三个参数，即`replace`，并将其设置为`False`。在此之后，一旦选择了一个元素，就不会考虑对其进行另一次选择。因此，一个元素在输出列表中只会出现一次。您可以在下面的示例中观察到这一点。

```py
import numpy

myList = [1, 2, 3, 45, 6, 8, 78, 23, 56, 7686, 123]
print("The list is:")
print(myList)
random_elements = numpy.random.choice(myList, 4, replace=False)
print("The randomly selected elements are:")
for x in random_elements:
    print(x)
```

输出:

```py
The list is:
[1, 2, 3, 45, 6, 8, 78, 23, 56, 7686, 123]
The randomly selected elements are:
1
56
3
2
```

## 结论

在本文中，我们讨论了几种在 python 中从列表中选择随机元素的方法。我们还看到了如何从列表中选择多个随机元素。要了解更多关于 python 中的列表，你可以阅读这篇关于[列表理解](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python)的文章。您可能也会喜欢这篇关于 python 中的[字符串连接的文章。](https://www.pythonforbeginners.com/concatenation/string-concatenation-and-formatting-in-python)