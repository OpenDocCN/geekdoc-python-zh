# 使用 Python round()

> 原文：<https://www.askpython.com/python/built-in-methods/python-round>

当我们想要执行舍入运算时，Python **round()** 函数非常有用。

通常，您会希望通过舍入或减少长浮点小数来快速估计值。 **round()** 在这种情况下尤其有利。

让我们看看如何使用它，以及一些例子！

* * *

## Python round()函数的语法

功能很简单。它接受一个数字，并输出所需的整数。

```py
round(num, [ndigits])

```

这里，我们需要对`num`进行舍入，所以我们将其传递给`round()`。我们也可以使用`ndigits`指定舍入的精度。这将确保数字在小数点后四舍五入到 **ndigits** 精度。

如果要将其四舍五入为整数值，不需要提供这个值。在这种情况下，将返回最接近的整数值。

同样，如果数字的形式是`x.5`，那么，如果上舍入值是偶数，则这些值将被上舍入。否则，将向下取整。

例如，2.5 将被舍入到 2，因为 2 是最接近的偶数，而 3.5 将被舍入到 4。

至此，让我们来看一些例子:

* * *

## Python round()的示例

有些情况需要考虑，所以让我们一个一个来。

### 1.整数舍入

下面的代码片段显示了我们如何将数字四舍五入到最接近的整数。

```py
print(round(5, 2)) # Round 5 upto 2 decimal places. But 5 is already an integer!
print(round(6.4))
print(round(7.7))
print(round(3.5))

```

**输出**

```py
5
6
8
4

```

### 2.舍入到偶数边

如前所述，如果偶数边和奇数边都同样接近数字(`x.5`)，那么偶数边会发生舍入。

```py
print(round(10.5))
print(round(11.5))

```

**输出**

```py
10
12

```

### 3.用 ndigit=None 舍入

如果未指定`ndigit`参数(或`None`)，则将舍入到最接近的整数。

```py
print(round(2.5)) # Rounding will happen to the nearest even number
# OR
print(round(2.5, None))

```

**输出**

```py
2
2

```

### 4.用 ndigit < 0 舍入

我们也可以为参数`ndigit`传递一个负值。这将从小数点左边开始舍入！

因此，如果我们的原始数字在小数点后有 3 个数字，传递`ndigit = -3`将删除小数点前的 3 个数字并用 0 替换，得到 0！

```py
print(round(123.456, 0))
print(round(123.456, -1))
print(round(123.456, -2))
print(round(123.456, -3))

```

**输出**

```py
123
120.0
100.0
0.0

```

## 浮点数舍入的异常

因为浮点数是由其精度定义的，所以 Python 在计算过程中会近似这些数字。由于这些近似值，舍入它们有时会给我们带来意想不到的结果。

考虑下面的块，其中的结果可能会令人惊讶:

```py
>>> 0.1 + 0.1 == 0.2
True
>>> 0.1 + 0.1 + 0.1 == 0.3
False
>>> 0.1 + 0.1 + 0.1 + 0.1 == 0.4
True

```

让我们来看看更多的例子:

```py
>>> print(round(21.575, 2))
21.57
>>> print(round(1.23546, 2))
1.24
>>> print(round(-1.2345, 2))
-1.23

```

同样，由于浮点的近似值，我们没有得到精确的输出。在第一种情况下，我们会期望得到`21.58`，但是我们只得到`21.57`。

因此，在处理浮点运算时，一定要小心，因为输出中的这些小变化可能会给你带来大问题。

* * *

## 对自定义对象使用 Python round()

Python `round()`方法在内部调用`__round__()` dunder 方法。

如果我们正在构建一个定制的[类](https://www.askpython.com/python/oops/python-classes-objects)，我们可以覆盖这个方法。因此，对我们的对象的任何调用都将转到这个覆盖方法。

让我们快速看一下如何构建我们自己的定制圆场！

```py
class MyClass:
    def __init__(self, data):
        assert type(data) == type([])
        self.data = data # Assume List type data of numbers
    def __round__(self, num_digits=None):
        for idx in range(len(self.data)):
            self.data[idx] = round(self.data[idx], num_digits)
        return self

my_obj = MyClass([12.34, 12.345, 10.5])
my_obj = round(my_obj)
print(my_obj.data)

```

**输出**

```py
[12, 12, 10]

```

如您所见，我们能够为`MyClass`对象实现我们自己的定制舍入功能！

* * *

## 结论

在本文中，我们通过各种例子学习了如何在 Python 中使用`round()`函数。

## 参考

*   [Python 文档](https://docs.python.org/3.7/library/functions.html#round)为 round()
*   关于 Python round()的 JournalDev 文章

* * *