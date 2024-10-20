# NumPy 集合运算知道！

> 原文：<https://www.askpython.com/python/numpy-set-operations>

读者朋友们，你们好！在本文中，我们将学习 Python 中的通用 NumPy 集合操作。所以，让我们开始吧！🙂

* * *

## 有用的数字集合运算

在本文中，我们将讨论 5 种有用的 numpy 集合运算。

1.  `numpy.unique(array)`
2.  `numpy.union1d(array,array)`
3.  `numpy.intersect1d(array,array,assume_unique)`
4.  `np.setdiff1d(arr1, arr2, assume_unique=True)`
5.  `np.setxor1d(arr1, arr2, assume_unique=True)`

让我们逐个检查这些操作。

### 1。NumPy 数组中的唯一值

这个 numpy 集合操作帮助我们从 Python 中的数组元素集合中找到唯一值。`numpy.unique()`函数跳过所有重复的值，只表示数组中唯一的元素

**语法:**

```py
numpy.unique(array)

```

**举例:**

在这个例子中，我们使用 unique()函数从数组集中选择并显示唯一的元素。因此，它跳过重复值 30，并且只选择它一次。

```py
import numpy as np
arr = np.array([30,60,90,30,100])
data = np.unique(arr)
print(data)

```

**输出:**

```py
[ 30  60  90 100]

```

* * *

### 2。在 NumPy 数组上设置 union 操作

[NumPy](https://www.askpython.com/python-modules/numpy/python-numpy-module) 为我们提供了通用的`union1d()`函数，可以对两个数组进行联合运算。

也就是说，它将两个数组中的值相加并表示出来。这个过程完全忽略了重复值，并且在数组的联合集中只包含一个重复元素。

**语法:**

```py
numpy.union1d(array,array)

```

**举例:**

```py
import numpy as np
arr1 = np.array([30,60,90,30,100])
arr2 = np.array([1,2,3,60,30])

data = np.union1d(arr1,arr2)

print(data)

```

**输出:**

```py
[  1   2   3  30  60  90 100]

```

* * *

### 3。在 NumPy 数组上设置交集操作

`intersect1d() function`使我们能够对数组执行交集操作。也就是说，它从两个数组中选择并表示公共元素。

**语法:**

```py
numpy.intersect1d(array,array,assume_unique)

```

*   assume_unique:如果设置为 TRUE，则包含交集运算的重复值。将其设置为 FALSE 将导致交集运算忽略重复值。

**举例:**

这里，由于我们已经将`assume_unique`设置为真，所以已经执行了包括重复值的交集操作，即它从两个数组中选择公共值，包括那些公共元素的重复。

```py
import numpy as np
arr1 = np.array([30,60,90,30,100])
arr2 = np.array([1,2,3,60,30])

data = np.intersect1d(arr1, arr2, assume_unique=True)

print(data)

```

**输出:**

```py
[30 30 60]

```

* * *

## 4.使用 NumPy 数组查找不常见的值

使用`setdiff1d()`函数，我们可以根据传递给函数的参数找到并表示第一个数组中不存在于第二个数组中的所有元素。

```py
import numpy as np
arr1 = np.array([30,60,90,30,100])
arr2 = np.array([1,2,3,60,30])

data = np.setdiff1d(arr1, arr2, assume_unique=True)

print(data)

```

**输出:**

```py
[ 90 100]

```

* * *

## 5。对称差异

使用`setxor1d()`函数，我们可以计算数组元素之间的对称差。也就是说，它选择并表示两个数组中不常见的所有元素。因此，它省略了数组中的所有公共值，并表示两个数组的不同值。

**举例:**

```py
import numpy as np
arr1 = np.array([30,60,90,30,100])
arr2 = np.array([1,2,3,60,30])

data = np.setxor1d(arr1, arr2, assume_unique=True)

print(data)

```

**输出:**

```py
[  1   2   3  90  100]

```

* * *

## 结论

到此，我们就结束了这个话题。如果你遇到任何问题，欢迎在下面评论。更多与 Python 编程相关的帖子，请继续关注我们。

在那之前，学习愉快！！🙂