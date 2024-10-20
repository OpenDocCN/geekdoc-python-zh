# NumPy 中的 3 种简单排序技术

> 原文：<https://www.askpython.com/python/sorting-techniques-in-numpy>

读者朋友们，你们好！在本文中，我们将详细关注 NumPy 中的 **3 排序技术。**

所以，让我们开始吧！🙂

**[Python NumPy 库](https://www.askpython.com/python-modules/numpy/python-numpy-module)** 为我们提供了各种函数来创建数组和操作数组结构中相似类型的元素。除此之外，NumPy 还为我们提供了各种函数，使我们能够对数组结构中的元素进行排序。

## NumPy 中的排序技术

我们将在 NumPy 中学习下面的排序技术。

1.  **NumPy sort()函数**
2.  **NumPy argsort()函数**
3.  **NumPy lexsort()函数**

所以，让我们开始吧！

### 1。NumPy sort()函数

为了对数组结构中出现的各种元素进行排序，NumPy 为我们提供了 **sort()** 函数。使用 sort()函数，我们可以对元素进行排序，并分别按照升序到降序对它们进行分离。

看看下面的语法！

**语法:**

```py
numpy.sort(array, axis)

```

参数“轴”指定了需要执行排序的方式。因此，当我们设置 axis = NONE 时，排序以传统方式进行，得到的数组是单行元素。另一方面，如果我们设置 axis = 1，排序是以行的方式进行的，也就是说，每一行都是单独排序的。

**例 1:**

在这个例子中，我们已经创建了一个数组，我们还使用 [sort()函数](https://www.askpython.com/python/list/python-sort-list)和 **axis = NONE** 对数组进行了排序，也就是说，它按升序对元素进行排序。

```py
import numpy as np
data = np.array([[22, 55], [0, 10]])
res = np.sort(data, axis = None)        
print ("Data before sorting:", data)
print("Data after sorting:", res)

```

**输出:**

```py
Data before sorting: [[22 55]
 [ 0 10]]
Data after sorting: [ 0 10 22 55]

```

**例 2:**

在这个例子中，我们已经创建了一个数组，并使用 sort()函数对其进行了排序，这里我们设置 axis = 1，即已经执行了按行排序。

```py
import numpy as np
data = np.array([[66, 55, 22], [0, 10, -1]])
res = np.sort(data, axis = 1)        
print ("Data before sorting:", data)
print("Row wise sorting:", res)

```

**输出:**

```py
Data before sorting: [[66 55 22]
 [ 0 10 -1]]
Row wise sorting: [[22 55 66]
 [-1  0 10]]

```

* * *

### 2。NumPy argsort()

除了 sort()方法之外，我们还有用作 NumPy 中排序技术的 **argsort()** 函数，它返回排序元素的索引的**数组。从这些排序的索引值中，我们可以得到按升序排序的数组元素。**

因此，用 argsort()函数，我们可以对数组值进行排序，并得到与单独数组相同的索引值。

**举例:**

```py
import numpy as np
data = np.array([66, 55, 22,11,-1,0,10])
res_index = np.argsort(data)        
print ("Data before sorting:", data)
print("Sorted index values of the array:", res_index)

x = np.zeros(len(res_index), dtype = int)
for i in range(0, len(x)):
    x[i]= data[res_index[i]]
print('Sorted array from indexes:', x)

```

**输出:**

在上面的例子中，我们对数据值执行了 argsort()函数，并获得了元素的排序索引值。此外，我们利用相同的数组索引值来获得排序后的数组元素。

```py
Data before sorting: [66 55 22 11 -1  0 10]
Sorted index values of the array: [4 5 6 3 2 1 0]
Sorted array from indexes: [-1  0 10 11 22 55 66]

```

* * *

### 3。NumPy lexsort()函数

lexsort()函数使我们能够使用键序列(即按列)对数据值进行排序。使用 **lexsort()** 函数，我们一次一个地对两个数组进行排序。结果，我们得到排序元素的索引值。

```py
import numpy as np
data = np.array([66, 55, 22,11,-1,0,10])
data1 = np.array([1,2,3,4,5,0,-1])
res_index = np.lexsort((data1, data))        
print("Sorted index values of the array:", res_index)

```

**输出:**

```py
Sorted index values of the array: [4 5 6 3 2 1 0]

```

* * *

## 结论

如果你遇到任何问题，欢迎在下面评论。更多与 Python 编程相关的帖子，请继续关注我们。

在那之前，学习愉快！！