# 在 Python 中反转数组——10 个例子

> 原文：<https://www.askpython.com/python/array/reverse-an-array-in-python>

## 介绍

在本教程中，我们将回顾 Python 中反转数组的不同方法。 **Python** 语言没有附带**数组**数据结构支持。相反，它有易于使用的内置列表结构，并提供了一些执行操作的方法。

我们可以通过导入像 **Array** 或 **NumPy** 这样的模块来继续使用 Python 中的典型数组。我们的教程将分为三个部分，每个部分都处理 Python 中的单个数组类型的反转。他们是，

*   在 Python 中反转数组列表，
*   在 Python 中反转数组模块的数组，
*   在 Python 中反转 NumPy 数组。

现在让我们进入正题。

## 在 Python 中反转列表数组

正如我们已经讨论过的，Python 中的**列表**和**数组**是相似的。两者的主要区别在于数组只允许相同数据类型的项，而列表允许它们不同。

因为 Python 不支持传统的数组，所以我们可以用列表来描述它们，并尝试反转它们。让我们看看完成这项任务的不同方法，

### 1.在 Python 中使用列表切片反转数组

我们可以使用**切片**方法反转一个列表数组。这样，我们实际上以与原始列表相反的顺序创建了一个新列表。让我们看看如何:

```py
#The original array
arr = [11, 22, 33, 44, 55]
print("Array is :",arr)

res = arr[::-1] #reversing using list slicing
print("Resultant new reversed array:",res)

```

**输出**:

```py
Array is : [1, 2, 3, 4, 5]
Resultant new reversed array: [5, 4, 3, 2, 1]

```

### 2.使用 reverse()方法

Python 还提供了一个内置的方法`reverse()`,直接在原来的地方颠倒列表项的顺序。

**注**:这样，我们就改变了实际列表的顺序。因此，原始订单丢失。

```py
#The original array
arr = [11, 22, 33, 44, 55]
print("Before reversal Array is :",arr)

arr.reverse() #reversing using reverse()
print("After reversing Array:",arr)

```

**输出**:

```py
Before reversal Array is : [11, 22, 33, 44, 55]
After reversing Array: [55, 44, 33, 22, 11]

```

### 3.使用 reversed()方法

我们还有另一个方法，`reversed()`,当它与一个列表一起传递时，返回一个 iterable，它只包含列表中逆序的项。如果我们在这个 iterable 对象上使用`list()`方法，我们会得到一个新的列表，其中包含我们的反转数组。

```py
#The original array
arr = [12, 34, 56, 78]
print("Original Array is :",arr)
#reversing using reversed()
result=list(reversed(arr))
print("Resultant new reversed Array:",result)

```

**输出**:

```py
Original Array is : [12, 34, 56, 78]
Resultant new reversed Array: [78, 56, 34, 12]

```

## 在 Python 中反转数组模块的数组

即使 Python 不支持数组，我们也可以使用**数组模块**来创建不同数据类型的类似数组的对象。尽管这个模块对数组的数据类型有很多限制，但它在 Python 中被广泛用于处理数组数据结构。

现在，让我们看看如何在 Python 中反转用 array 模块创建的数组。

### 1.使用 reverse()方法

与列表类似，`reverse()`方法也可以用于直接反转数组模块的 Python 中的数组。它在原始位置反转数组，因此不需要额外的空间来存储结果。

```py
import array

#The original array
new_arr=array.array('i',[2,4,6,8,10,12])
print("Original Array is :",new_arr)

#reversing using reverse()
new_arr.reverse()
print("Reversed Array:",new_arr)

```

**输出**:

```py
Original Array is : array('i', [2, 4, 6, 8, 10, 12])
Resultant new reversed Array: array('i', [12, 10, 8, 6, 4, 2])

```

### 2.使用 reversed()方法

同样，`reversed()`方法在通过数组传递时，返回一个元素顺序相反的 iterable。看看下面的例子，它展示了我们如何使用这个方法来反转一个数组。

```py
import array

#The original array
new_arr=array.array('i',[10,20,30,40])
print("Original Array is :",new_arr)

#reversing using reversed()
res_arr=array.array('i',reversed(new_arr))
print("Resultant Reversed Array:",res_arr)

```

**输出**:

```py
Original Array is : array('i', [10, 20, 30, 40])
Resultant Reversed Array: array('i', [40, 30, 20, 10])

```

## 在 Python 中反转 NumPy 数组

`Numpy`模块允许我们在 Python 中使用**数组**数据结构，这些数据结构真的**快**，并且只允许相同的数据类型数组。

这里，我们将反转用 NumPy 模块构建的 Python 中的数组。

### 1.使用 flip()方法

NumPy 模块中的`flip()`方法反转 NumPy 数组的顺序并返回 NumPy 数组对象。

```py
import numpy as np

#The original NumPy array
new_arr=np.array(['A','s','k','P','y','t','h','o','n'])
print("Original Array is :",new_arr)

#reversing using flip() Method
res_arr=np.flip(new_arr)
print("Resultant Reversed Array:",res_arr)

```

**输出**:

```py
Original Array is : ['A' 's' 'k' 'P' 'y' 't' 'h' 'o' 'n']
Resultant Reversed Array: ['n' 'o' 'h' 't' 'y' 'P' 'k' 's' 'A']

```

### 2.使用 flipud()方法

`flipud()`方法是 **NumPy** 模块中的另一个方法，它上下翻转一个数组。它还可以用于在 Python 中反转 NumPy 数组。让我们看看如何在一个小例子中使用它。

```py
import numpy as np

#The original NumPy array
new_arr=np.array(['A','s','k','P','y','t','h','o','n'])
print("Original Array is :",new_arr)

#reversing using flipud() Method
res_arr=np.flipud(new_arr)
print("Resultant Reversed Array:",res_arr)

```

**输出**:

```py
Original Array is : ['A' 's' 'k' 'P' 'y' 't' 'h' 'o' 'n']
Resultant Reversed Array: ['n' 'o' 'h' 't' 'y' 'P' 'k' 's' 'A']

```

### 3.使用简单切片

正如我们之前对列表所做的那样，我们可以使用**切片**来反转用 Numpy 构建的 Python 中的数组。我们创建一个新的 **NumPy** 数组对象，它以相反的顺序保存条目。

```py
import numpy as np

#The original NumPy array
new_arr=np.array([1,3,5,7,9])
print("Original Array is :",new_arr)

#reversing using array slicing
res_arr=new_arr[::-1]
print("Resultant Reversed Array:",res_arr)

```

**输出**:

```py
Original Array is : [1 3 5 7 9]
Resultant Reversed Array: [9 7 5 3 1]

```

## 结论

因此，在本教程中，我们学习了如何使用各种方法或技术在 Python 中反转数组。希望能给大家一个清晰的认识。

## 参考

*   [https://www.askpython.com/python/python-numpy-arrays](https://www.askpython.com/python-modules/numpy/python-numpy-arrays)
*   [https://www . ask python . com/python/array/python-array-examples](https://www.askpython.com/python/array/python-array-examples)