# Python 中二元元组到整数的转换

> 原文：<https://www.pythonforbeginners.com/basics/binary-tuple-to-integer-in-python>

二进制元组是只包含 0 和 1 作为元素的元组。在本文中，我们将讨论在 python 中将二进制元组转换为整数的不同方法。

## 如何在 Python 中将二进制元组转换成整数

考虑给我们以下二元元组。

```py
myTuple= (1,1,0,1,1,0,1)
```

现在，我们必须从这个二元元组创建一个整数。整数将包含二进制表示中元组的所有数字。所以，数字会是`(1101101)₂`。在十进制表示中，`(1101101)₂`的值等于`109`。所以，我们的程序应该给出输出`109`。

为了将二进制数转换成十进制数，我们将数字的位数乘以 2 的幂。最右边的位乘以 2⁰。最右边第二个数字乘以 2¹，右边第三个数字乘以 2²。同样，第 N 位乘以 2^(N-1)。之后，将每个数字的值相加，得到十进制表示。

例如，我们可以将`(1101101)₂` 转换为十进制表示如下。

`(1101101)₂ = 1x2⁶+1x2⁵+0x2⁴+1x2³+1x2²+0x2¹+1x2⁰`

`    =64+32+0+8+4+0+1`

`    =109`

为了在 python 中实现上述将二进制元组转换为整数的逻辑，我们将首先将变量`myInt`初始化为 0。我们还将使用 `len()`函数计算元组的长度。之后，我们将从右到左遍历元组，并使用 for 循环、 `range()`函数和元组长度将元素乘以 0 的幂。相乘后，我们将把这些值加到`myInt`中。在执行 for 循环后，我们将在`myInt`变量中得到整数输出。您可以在下面的示例中观察到这一点。

```py
myTuple = (1, 1, 0, 1, 1, 0, 1)
myInt = 0
length = len(myTuple)
for i in range(length):
    element = myTuple[length - i - 1]
    myInt = myInt + element*pow(2, i)

print("The tuple is:", myTuple)
print("The output integer is:", myInt)
```

输出:

```py
The tuple is: (1, 1, 0, 1, 1, 0, 1)
The output integer is: 109
```

## 使用字符串将二进制元组转换为整数

在 python 中，我们还可以使用字符串将二进制元组转换为整数。为此，我们将首先使用`str()`函数和`map()`函数将元组的所有元素转换为字符串。之后，我们将使用`join()`方法从元组的元素创建一个字符串。在字符串上调用 `join()`方法时，该方法将 iterable 对象作为输入，并返回由 iterable 元素组成的字符串。我们将首先创建一个空字符串，然后使用`join()` 方法从元组中获取字符串，如下所示。

```py
myTuple = (1, 1, 0, 1, 1, 0, 1)
newTuple = map(str, myTuple)
myStr = "".join(newTuple)

print("The tuple is:", myTuple)
print("The output string is:", myStr)
```

输出:

```py
The tuple is: (1, 1, 0, 1, 1, 0, 1)
The output string is: 1101101
```

获得字符串后，我们可以使用 int()函数直接将字符串转换为整数，如下所示。

```py
myTuple = (1, 1, 0, 1, 1, 0, 1)
newTuple = map(str, myTuple)
myStr = "".join(newTuple)
myInt = int(myStr, 2)
print("The tuple is:", myTuple)
print("The output integer is:", myInt)
```

输出:

```py
The tuple is: (1, 1, 0, 1, 1, 0, 1)
The output integer is: 109
```

在代码中，我们将值 2 作为第二个输入参数传递给了`int()`函数，以显示该字符串包含二进制数的位。

## 结论

在本文中，我们讨论了在 python 中将二进制元组转换为整数的两种方法。要了解更多关于字符串的知识，你可以阅读这篇关于 python 中的[字符串连接](https://www.pythonforbeginners.com/concatenation/string-concatenation-and-formatting-in-python)的文章。你可能也会喜欢这篇关于 python 中的[列表理解的文章。](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python)