# 反转布尔数组的元素

> 原文：<https://www.askpython.com/python-modules/numpy/invert-elements-of-a-boolean-array>

在这篇文章中，我们将学习如何反转一个布尔数组的元素，该数组包含布尔值，如 True 或 False。

## Python 中的布尔数组是什么？

布尔数组是一个有布尔值的数组，比如真或假，或者可能是 1 或 0。使用 dtype = bool 可以形成布尔数组。除了 0、无、假或空字符串之外，其他都被认为是真的。

```py
import numpy as np

arr_bool = np.array([1, 1.1, 0, None, 'a', '', True, False], dtype=bool)
print(arr_bool)

```

**输出:**

```py
[ True  True False False  True False  True False]

```

## 反转布尔数组元素的方法

以下是在 Python 中反转布尔数组元素的方法。

### 使用 np.invert()函数

使用内置的 np。invert()函数可以反转一个布尔数组的元素。

```py
import numpy as np
arr = np.array((True, True, False, True, False))
arr_inver = np.invert(arr)
print(arr_inver)

```

**输出:**

```py
[False False  True False  True]

```

### 使用 if-else 方法

在这个方法中，我们将检查数组中每个元素的索引值。如果该值为零，它将被更改为 1，反之亦然。此外，如果值为 True，它将被更改为 False。

```py
arr = ((0, 1, 0, 1))
a1 = list(arr)

for x in range(len(a1)):
    if(a1[x]):
        a1[x] = 0
    else:
        a1[x] = 1

print(a1)

```

**输出:**

```py
[1, 0, 1, 0]

```

## 结论

总之，我们学习了在 python 中反转布尔数组元素的不同方法。Numpy 是一个灵活的 python 库，并提供了多种功能。