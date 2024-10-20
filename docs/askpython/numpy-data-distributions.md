# 要知道的 5 种 NumPy 数据分布

> 原文：<https://www.askpython.com/python-modules/numpy/numpy-data-distributions>

读者朋友们，你们好！在本文中，我们将关注 Python 中的 **5 NumPy 数据分布**。所以，让我们开始吧！！🙂

首先，数据分布使我们对数据的分布有一个概念。也就是说，它表示数据范围内所有可能值的列表，还表示这些数据值在分布中的频率。

[Python NumPy 模块](https://www.askpython.com/python-modules/numpy/python-numpy-arrays#the-numpy-array-object)为我们提供了 random 类，帮助我们为随机化的数据值随机生成数据分布。

## 数字数据分布

让我们使用下面的 NumPy 数据分布。

1.  **Zipf 分配**
2.  **帕累托分布**
3.  **瑞利分布**
4.  **指数分布**
5.  **具有 choice()函数的随机分布**

* * *

### 1。Zipf 分布

Zipf NumPy 数据分布基于 Zipf 定律，即第 x 个最常见元素是该范围中最常见元素的 1/x 倍。

Python **random.zipf()** 函数使我们能够在一个数组上实现 zipf 分布。

**语法:**

```py
random.zipf(a,size)

```

*   **a** :分布参数
*   **size** :合成数组的尺寸。

**举例:**

```py
from numpy import random

data = random.zipf(a=2, size=(2, 4))

print(data)

```

**输出:**

```py
[[   2   24    1    1]
 [   4 1116    4    4]]

```

* * *

### 2。帕累托分布

它遵循帕累托定律，即 20%的因素促成了 80%的结果。pareto()函数使我们能够在随机化的数字上实现 Pareto 数据分布。

看看下面的语法！

```py
random.pareto(a,size)

```

*   **a** :形状
*   **size** :合成数组的尺寸。

**举例:**

```py
from numpy import random

data = random.pareto(a=2, size=(2, 4))

print(data)

```

**输出:**

```py
[[2.33897169 0.40735475 0.39352079 2.68105791]
 [0.02858458 0.60243598 1.17126724 0.36481641]]

```

* * *

### 3。瑞利分布

有了**瑞利分布**，我们就可以在信号处理中用概率密度来定义和理解分布。

看看下面的语法！

```py
random.rayleigh(scale,size)

```

*   **标度**:标准偏差值基本上决定了一个数据分布的平坦性。
*   **size** :输出数组的尺寸。

**举例:**

```py
from numpy import random

data = random.rayleigh(scale=2, size=(2, 4))

print(data)

```

**输出:**

```py
[[3.79504431 2.24471025 2.3216389  4.01435725]
 [3.1247996  1.08692756 3.03840615 2.35757077]]

```

* * *

### 4。指数分布

**指数分布**使我们能够了解到下一个事件发生的时间范围。也就是说，任何动作的发生率取决于概率得分。比如成功的框架 v/s 失败率——成功/失败。

**语法:**

```py
random.exponential(scale, size)

```

*   **标度**:动作发生次数的倒数。默认值= 1.0
*   **大小**:输出数组的大小。

**举例:**

```py
from numpy import random

data = random.exponential(scale=2, size=(2, 4))

print(data)

```

**输出:**

```py
[[0.56948472 0.08230081 1.39297867 5.97532969]
 [1.51290257 0.95905262 4.40997749 7.25248917]]

```

* * *

### 5。具有 choice()函数的随机分布

随机分布表示遵循概率密度值的某些特征的一组随机数据。random 类为我们提供了 **choice()函数**，它使我们能够基于一组概率值定义随机数。

概率范围在 0 和 1 之间——0 表示该数字不会出现，1 表示该数字在集合中一定会出现。

**语法:**

```py
random.choice(array, p, size)

```

*   **数组**:需要发生随机数据分布的元素。数组元素的个数应该等于 p 的计数。
*   **p** :随机数据分布中每个数组元素出现的概率得分。p 的所有值之和必须等于 1。
*   **大小**:二维/一维数组的大小。

**举例:**

```py
from numpy import random

data = random.choice([1,3,5,7], p=[0.1, 0.3, 0.2, 0.4], size=(2, 2))

print(data)

```

**输出:**

```py
[[7 7]
 [1 3]]

```

* * *

## 结论

到此，我们就结束了这个话题。如果你遇到任何问题，欢迎在下面评论。

更多与 Python 编程相关的帖子，请继续关注我们。

在那之前，学习愉快！！🙂