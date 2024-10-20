# Python 中的 reduce()函数

> 原文：<https://www.askpython.com/python/reduce-function>

在本文中，我们将会看到 python 中一个非常重要的函数，叫做 **reduce()。**

当我们编写某种程序时，我们会遇到各种各样的场景，当我们需要得到某种数学上的总数或累计值来使用那个结果并做一些事情。

假设我们有一个电子商务网站，我们的购物车中有商品和商品价格，我们想计算总金额，以便客户可以下订单。嗯， **reduce()函数**帮助我们计算总数，从而将一个值列表简化为一个值。

需要记住的另一件重要事情是, **reduce()函数不是 Python 内置函数的一部分。**当我们下载 python 解释器和 Python 包时， **functools** 是我们称之为工具带的东西，我们可以将它用于 Python 安装中附带的功能工具，从那里我们希望将 reduce 函数作为一个模块导入。

**functools** 模块用于处理或返回其他函数的高阶函数，在我们的例子中，它为我们提供了 reduce 函数。

要了解高阶函数，请参考[文档](https://docs.python.org/3/library/functools.html)

让我们通过一些代码示例来更好地理解 **reduce()函数。**

## 在 Python 中实现 Reduce 函数

首先，我们需要从 functools 模块中导入 reduce 函数

```py
from functools import reduce

```

声明一个数字列表，用它来执行计算

```py
my_prices = [40, 50, 60]

```

让我们为计算定义一个函数

```py
def calc_total(acc, eachItem):
    return acc + eachItem

```

**解释**:我们为我们的函数提供了两个参数， **acc** 或累加值，以及 **eachItem** ，当我们遍历列表时，它们被用作变量。不要担心，我们将回到累积值来更好地了解它的使用。

实现 **reduce()** 函数:

```py
total = reduce(calc_total, my_prices, 0)

```

reduce 函数总共有 3 个参数。

1.  第一个是我们之前在代码中定义的函数。
2.  第二个是值的列表或我们的序列，我们希望使用它来执行我们的操作，也是前面定义的。
3.  第三个参数 0 是初始值累计值，从技术上讲，它的默认值为 0，在我们的示例中这很好，但在更复杂的运算中，如果我们想要初始值为 1，那么必须将其定义为第三个参数，计算会将 1 作为运算的初始值。

**解释**:reduce()函数从初始的**累积值(0)** 开始，接受我们的函数 **(calc_total)** ，通过迭代所有的条目使用数字列表 **(my_prices)** ，并保留所执行函数每次迭代的累积值，并返回最终结果。

现在，我们已经有了代码，让我们打印出结果

```py
print(f"The summation for the list is {total}")

```

输出:

```py
The summation for the list is **150**
```

## 完整的代码

```py
# Import reduce
from functools import reduce

# Declare a list
my_prices = [40, 50, 60]

# Define our function
def calc_total(acc, eachItem):
    return acc + eachItem

# Use reduce(function_name, data, initial_value)
total = reduce(calc_total, my_prices, 0)

# Print the result
print(f"The summation for the list is {total}")

```

## 结论

我们看到我们的代码如我们预期的那样工作。我们得到的总数是 150 英镑。这就是 Python 中 **reduce()函数**的工作方式。它不仅在 Python 中，而且在大多数其他编程语言中都是一个非常重要的函数。reduce 函数的基本概念和用例在所有语言中几乎都是相同的，这使得很好地理解它们变得很有价值。

## 参考

请查看官方[文档](https://docs.python.org/3/)了解更多信息。