# NumPy gcd–返回两个数字的最大公约数

> 原文:# t0]https://www . aspython . com/python-modules/num py/numpy-gcd

正如我们在前面的练习中看到的， [NumPy 的 lcm 函数](https://www.askpython.com/python-modules/numpy/numpy-lcm)对于寻找两个或更多数字的最小公倍数非常有用，如果我们想要两个或更多数字的最大公约数呢？这就是 numpy gcd 函数发挥作用的地方。两个或多个数的最大公约数(GCD)是能整除这两个数的最大公因数。

***也读作:[NumPy LCM——返回两个数的最小公倍数](https://www.askpython.com/python-modules/numpy/numpy-lcm)***

## 什么是 NumPy gcd？

两个整数的最大公约数(GCD)是两个整数均分的最大整数。例如，16 和 24 的 GCD 是 8，因为 8 是 16 和 24 的最大整数。

NumPy gcd 是一个数学函数，**计算给予 **`nunpy.gcd()`** 函数的输入数字的 gcd 值**。

## NumPy gcd 的语法

让我们来看看函数的语法。

```py
numpy.gcd(x1 , x2)

```

输入的 **`x1`** 、 **`x2`** 不能是**负数**。

这里，`**x1**`和`**x2**`是输入的数字，可以是**单个数字**或者是一个 **NumPy 数组数字**。 **`numpy.gcd()`** 函数计算输入数字的 GCD(最大公约数)。

**注意:**数字不能是浮点数。

让我们写一些代码来更好地理解这个函数。

## 使用 NumPy gcd

两个整数的最大公约数可以用 NumPy 的 gcd 函数来计算。它是这样工作的:

### 单一数字的数量 gcd

```py
import numpy as np

print("The GCD of 12 and 46 is:",np.gcd(12 , 46))

print("The GCD of 12 and 24 is:",np.gcd(12 , 24))

print("The GCD of 25 and 50 is:",np.gcd(25 , 50))

print("The GCD of 5 and 100 is:",np.gcd(5 , 100))

print("The GCD of 17 and 87 is:",np.gcd(17 , 87))

```

### 输出

```py
The GCD of 12 and 46 is: 2
The GCD of 12 and 24 is: 12
The GCD of 25 and 50 is: 25
The GCD of 5 and 100 is: 5
The GCD of 17 and 87 is: 1

```

在上面的代码片段中，我们使用 **`import`** 语句导入了 NumPy 库。

在接下来的几行中，我们使用 **`np.gcd()`** 函数来计算数字的 GCD。注意我们得到两个数的 GCD 有多快，这就是 Python NumPy 库的强大之处🙂

我们先来了解一下 25 和 50 的 GCD 是怎么算出来的。我们可以观察到 25 和 50 相除的最大数是 25。所以，25 和 50 的 GCD 是 25，这也是由 **`np.gcd()`** 函数产生的输出。

现在，让我们看看如何计算一组数字的 GCD。

### 单个数数组的个数 gcd

在这里，我们将找到单个 NumPy 数组中**所有元素的 gcd。**

```py
import numpy as np

a = np.array((3, 6, 24, 56, 79, 144))

gcd_value = np.gcd.reduce(a)

print("Input Array:\n",a)
print("The GCD value of the elements of the array is:",gcd_value)

```

### 输出

```py
Input Array:
 [  3   6  24  56  79 144]
The GCD value of the elements of the array is: 1

```

为了找到单个 NumPy 数组的所有元素的 GCD，使用了 **`reduce`** 函数，该函数将 **`gcd`** 方法应用于数组的每个元素。

该功能的其余工作是相同的。

可以计算两个 NumPy 数组的 GCD 吗？让我们看看🙂

### 两个数字数组的 gcd

```py
import numpy as np

a = np.array((12 , 24 , 99))

b = np.array((44 , 66 , 27))

c = np.gcd(a , b)

print("Array 1:\n",a)
print("Array 2:\n",b)

print("GCD values:\n",c)

```

### 输出

```py
Array 1:
 [12 24 99]
Array 2:
 [44 66 27]
GCD values:
 [4 6 9]

```

太神奇了！我们刚刚计算了两个 NumPy 数组的 GCD。

这里， **`np.gcd()`** 函数选取两个数组中相同位置的元素，并计算它们的 GCD。例如，在上面的代码片段中，选择了 12 和 44，并计算了它们的 GCD。类似地，计算两个数组的下一个元素及其 GCD。

**`np.gcd()`** 的输出是一个 NumPy 数组，存储在上面代码片段中的变量 **`c`** 中。

这就是关于 NumPy gcd 函数的全部内容。

## 摘要

在本文中，我们学习了函数的语法，并练习了不同类型的示例，使我们的理解更加清晰。这些 NumPy 函数使用起来非常简单，你知道这就是 NumPy 库的强大之处。它有很多数学函数，当我们对大量数据进行数学计算时，这些函数使事情变得简单。

在这里继续探索关于其他 Python 主题的精彩文章[。](https://www.askpython.com/)

## 参考

[num py documentation–num py gcd](https://numpy.org/doc/stable/reference/generated/numpy.gcd.html)