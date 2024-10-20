# 搜索数组的 5 个技巧

> 原文：<https://www.askpython.com/python/search-numpy-array>

读者朋友们，你们好！在本文中，我们将详细讨论 5 种使用条件搜索 NumPy 数组的技术。

所以，让我们开始吧！🙂

一个 [NumPy 数组](https://www.askpython.com/python-modules/numpy/python-numpy-arrays)在一个连续的结构中存储相似类型的元素。我们经常遇到需要在动态运行时查看数组的最大和最小元素的情况。NumPy 为我们提供了一组函数，使我们能够搜索应用了特定条件的特定元素。

## 如何在 NumPy 数组中搜索特定的元素？

让我们详细看看用于搜索 NumPy 数组的 5 个函数:

1.  **arg max()函数**
2.  **nanargmax()函数**
3.  **arg min()函数**
4.  **nargmin()函数**
5.  **使用 where()函数搜索**

* * *

### 1。NumPy argmax()函数

使用 NumPy **argmax()函数**，我们可以轻松地获取并显示数组结构中最大元素的索引。

这样，最大元素的索引就是 argmax()函数的结果值。

**语法:**

```py
numpy.argmax() function

```

**举例:**

```py
import numpy as np
data = np.array([[66, 99, 22,11,-1,0,10],[1,2,3,4,5,0,-1]])
res =  np.argmax(data) 
print(data)
print("Max element's index:", res)

```

**输出:**

在上面的例子中，我们创建了两个相同数据类型的数组。此外，应用 argmax()函数从所有元素中获取 max 元素的索引。因为 99 是最大的元素，所以结果索引值显示为 1。

```py
[[66 99 22 11 -1  0 10]
 [ 1  2  3  4  5  0 -1]]
Max element's index: 1

```

* * *

### 2。NumPy nanargmax()函数

使用 **nanargmax()函数**，我们可以轻松处理数组中出现的 NAN 或 NULL 值。也就是说，它不会被区别对待。NAN 值对搜索值的功能没有影响。

**语法:**

```py
numpy.nanargmax()

```

**举例:**

在下面的示例中，数组元素包含使用 numpy.nan 函数传递的空值。此外，我们现在使用 nanargmax()函数来搜索 NumPy 数组，并从数组元素中找到最大值，而不让 NAN 元素影响搜索。

```py
import numpy as np
data = np.array([[66, 99, 22,np.nan,-1,0,10],[1,2,3,4,np.nan,0,-1]])
res =  np.nanargmax(data) 
print(data)
print("Max element's index:", res)

```

**输出:**

```py
[[66\. 99\. 22\. nan -1\.  0\. 10.]
 [ 1\.  2\.  3\.  4\. nan  0\. -1.]]
Max element's index: 1

```

* * *

### 3。NumPy argmin()函数

使用 **argmin()函数**，我们可以搜索 NumPy 数组，并在更大范围内获取数组中最小元素的索引。它搜索数组结构中的最小值，并返回该值的索引。因此，通过索引，我们可以很容易地获得数组中的最小元素。

**语法:**

```py
numpy.argmin() function

```

**举例:**

```py
import numpy as np
data = np.array([[66, 99, 22,11,-1,0,10],[1,2,3,4,5,0,-1]])
res =  np.argmin(data) 
print(data)
print("Min element's index:", res)

```

**输出:**

如下所示，有两个索引占据了最低的元素，即[-1]。但是，argmin()函数返回数组值中最小元素的第一个匹配项的索引。

```py
[[66 99 22 11 -1  0 10]
 [ 1  2  3  4  5  0 -1]]
Min element's index: 4

```

* * *

### 4。NumPy where()函数

使用 **where()函数**，我们可以轻松地在 NumPy 数组中搜索任何元素的索引值，这些元素匹配作为参数传递给函数的条件。

**语法:**

```py
numpy.where(condition)

```

**举例:**

```py
import numpy as np
data = np.array([[66, 99, 22,11,-1,0,10],[1,2,3,4,5,0,-1]])
res =  np.where(data == 2) 
print(data)
print("Searched element's index:", res)

```

**输出:**

在这个例子中，我们从数组中搜索了一个值等于 2 的元素。此外，where()函数返回数组索引及其数据类型。

```py
[[66 99 22 11 -1  0 10]
 [ 1  2  3  4  5  0 -1]]
Searched element's index: (array([1], dtype=int64))

```

* * *

### 5。NumPy nanargmin()函数

使用 **nanargmin()函数**，我们可以轻松地搜索 NumPy 数组，找到数组元素中最小值的索引，而不必担心数组元素中的 NAN 值。空值对元素的搜索没有影响。

**语法:**

```py
numpy.nanargmin()

```

**举例:**

```py
import numpy as np
data = np.array([[66, 99, np.nan,11,-1,0,10],[1,2,3,4,5,0,-1]])
res =  np.nanargmin(data) 
print(data)
print("Searched element's index:", res)

```

**输出:**

```py
[[66\. 99\. nan 11\. -1\.  0\. 10.]
 [ 1\.  2\.  3\.  4\.  5\.  0\. -1.]]
Searched element's index: 4

```

* * *

## 结论

如果你遇到任何问题，请随时在下面评论。更多与 Python 编程相关的帖子，请继续关注我们。

在那之前，学习愉快！！🙂