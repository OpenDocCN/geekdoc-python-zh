# Python 中数组是可变的吗？

> 原文：<https://www.askpython.com/python/array/are-arrays-mutable-in-python>

我们可以将 Python 对象分为两大类，即可变对象和不可变对象。可变对象是那些一旦被创建就可以被改变或修改的对象，而不可变对象一旦被创建就不能被改变。数组属于可变对象的范畴。在本文中，我们将学习数组及其可变性，以及可以对数组执行的操作。所以让我们开始吧！

## Python 中的数组是什么？

数组是 Python 中存储相似类型对象集合的数据结构。数组中的对象由一组正整数索引。它们可以是多维的，对科学计算非常有用。

**例如:**

```py
import numpy as np
list=[1,2,3,4]
arr = np.array(list)
print(arr)

```

**输出:**

```py
[1 2 3 4]

```

在上面的例子中，我们从一个列表中创建了一个一维数组。

您可以通过以下方法访问数组元素。

```py
import numpy as np
list=[1,2,3,4]
arr = np.array(list)
print("First element of array is:",arr[0]) 
print("Last element of array is:",arr[-1])

```

**输出:**

```py
First element of array is: 1
Last element of array is: 4

```

现在我们来看看数组的可变属性。

## 数组的可变属性

现在，我们将通过例子来看看我们能在数组中做出什么样的改变。

### 在数组中插入元素

insert 函数帮助你在数组中插入元素。该函数有两个参数，一个是要插入元素的索引位置，另一个是元素的值。

```py
import array as np

a = np.array('i', [1, 2, 3])

#using insert function
a.insert(1, 4)
print(a)

```

**输出:**

```py
array('i', [1, 4, 2, 3])

```

### 修改数组中的元素

您可以借助以下代码修改数组中的元素。

```py
import array as np

a = np.array('i', [1, 2, 3])

#using insert function
a[1]=9
print(a)

```

**输出:**

```py
array('i', [1, 9, 3])

```

您需要指定要修改的元素的索引位置。

### 弹出数组中的元素

pop()函数将帮助你弹出一个元素。您需要指定想要弹出的元素的索引位置。该函数的作用类似于删除操作。

```py
import array as np

a = np.array('i', [1, 2, 3])

#using pop function
a.pop(1)
print(a)

```

**输出:**

```py
array('i', [1, 3])

```

### 从数组中删除或移除元素

remove()函数将帮助你从一个数组中移除元素。您必须指定要删除的元素的值。

```py
import array as np

a = np.array('i', [1, 2, 3])

#using remove function
a.remove(3)
print(a)

```

**输出:**

```py
array('i', [1, 2])

```

### 反转数组

简单的 reverse()函数将帮助你反转一个数组。

```py
import array as np

a = np.array('i', [1, 2, 3])

#using remove function
a.reverse()
print(a)

```

**输出:**

```py
array('i', [3, 2, 1])

```

### 结论

总之，我们知道数组是可变的，可以在创建后修改或变更。理解基本的数组操作非常重要，因为数组在科学计算中非常有用。