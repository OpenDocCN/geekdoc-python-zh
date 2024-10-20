# NumPy LCM–返回两个数的最小公倍数

> 原文:# t0]https://www . aspython . com/python-modules/num py/numpy-LCM

读者你好！欢迎来到另一个关于 NumPy 数学函数的教程。在本教程中，我们将通过例子详细研究 NumPy lcm 函数。

你一定在数学课上计算过两个数的 **lcm(最小公倍数)**。计算它很有趣🙂

现在，让我们看看如何使用 Python 编程语言计算 lcm。让我们开始吧。

## 什么是 NumPy lcm？

NumPy lcm 是 NumPy 库的数学函数之一，用于计算两个输入数字的 lcm。这里，输入的数字必须是正的。

让我们看看函数的语法。

## NumPy lcm 的语法

```py
numpy.lcm(x1 , x2)

```

这里，x1 和 x2 可以是**个单一的数字**，也可以是**个数字数组**。

**注意:**数字 **x1** 和 **x2** 不能是浮点数字。

## 使用 NumPy lcm

让我们编写一些代码来处理 numpy.lcm()方法，并看看如何使用它。

### 单个数字的 NumPy lcm

```py
import numpy as np

print("The lcm of 3 and 15 is:",np.lcm(3 , 15))

print("The lcm of 12 and 44 is:",np.lcm(12 , 44))

print("The lcm of 3 and 9 is:",np.lcm(3 , 9))

print("The lcm of 120 and 200 is:",np.lcm(120,200))

```

### 输出

```py
The lcm of 3 and 15 is: 15
The lcm of 12 and 44 is: 132
The lcm of 3 and 9 is: 9
The lcm of 120 and 200 is: 600

```

输出非常明显。现在，让我们看看如何计算两个 NumPy 数组的 lcm。

### 带有 NumPy 数组的 NumPy lcm

```py
import numpy as np

a = np.array((12,44,78,144,10000))

b = np.array((24,54,18,120,100))

print("Input Arrays:\n",a,"\n",b)

print("The lcm values:\n",np.lcm(a , b))

```

### 输出

```py
Input Arrays:
 [   12    44    78   144 10000] 
 [ 24  54  18 120 100]
The lcm values:
 [   24  1188   234   720 10000]

```

从两个 NumPy 数组中各选取一个元素，并计算它们的 lcm。

这就是关于 NumPy lcm 函数的全部内容，这个函数很容易理解，也很容易使用。敬请关注更多此类文章。

## 参考

[NumPy 文件–num py LCM](https://numpy.org/doc/stable/reference/generated/numpy.lcm.html)