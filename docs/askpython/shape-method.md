# Python shape 方法:识别 Python 对象的维度

> 原文：<https://www.askpython.com/python-modules/pandas/shape-method>

Python shape 方法返回一个`[tuple](https://www.askpython.com/python/tuple/python-tuple)`,表示应用它的 Python 对象的尺寸。这些应用了`shape`方法的 Python 对象通常是一个`numpy.array`或一个`pandas.DataFrame`。由`shape`方法返回的元组中的元素数量等于 Python 对象中的维数。每个`tuple`元素代表对应于 Python 对象的维度的元素数量。

## 熊猫:形状方法

**[熊猫](https://www.askpython.com/python-modules/pandas/python-pandas-module-tutorial)** 中的`shape`方法返回一个代表`DataFrame`维度的`tuple`，即**(行&列)**。

### 1.检查数据框的尺寸

```py
# Import Pandas Python module
import pandas as pd 

# Create a Python list
ls =[['A','B','C','D'], ['e' ,'f' ,'g' ,'h'], [11, 22, 33, 44]]

# Create a Pandas DataFrame from the above list
df = pd.DataFrame(ls)

# Print the DataFrame
print(df)

# Check the dimensions of the DataFrame
print(df.shape)

```

**输出:**

```py
    0   1   2   3 
0   A   B   C   D 
1   e   f   g   h 
2  11  22  33  44 
(3, 4)

```

`shape`方法返回了一个元组 **(3，4)** ，其中有两个元素描述了 DataFrame 具有三行四列的两个维度。

### 2.检查空数据框的尺寸

```py
# Import Pandas Python module
import pandas as pd 

# Create an empty Pandas DataFrame
df = pd.DataFrame()

# Print the DataFrame
print(df)

# Check the dimensions of the empty DataFrame
print(df.shape)

```

**输出:**

```py
Empty DataFrame 
Columns: [] 
Index: [] 
(0, 0)

```

`shape`方法返回了一个元组 **(0，0)** ，其中包含两个元素，描述了 DataFrame 具有零行零列的两个维度。

## NumPy:形状方法

**NumPy** 中的`shape`方法返回一个代表`numpy array`尺寸的`tuple`。

### 1.检查 numpy 数组的维数

```py
# Import Python NumPy module
import numpy as np

# Define a numpy array with zero dimensions
arr = np.array([[[1,2] ,[3,5]], [[2,3] ,[4,7]], [[3,4] ,[5,8]]])

# Print the numpy array
print(arr)

# Check the dimensions of arr
print(arr.shape)

```

**输出:**

```py
[[[1 2 3] 
  [3 5 6]]] 
(1, 2, 3)

```

`shape`方法返回了一个元组 **(1，2，3)** ，它有三个元素，表示数组有三个维度，每个维度分别有一个、两个和三个元素。

### 2.检查零维 numpy 数组的维数

```py
# Import Python NumPy module
import numpy as np

# Define a numpy array with zero dimensions
arr = np.array(0)

# Print the numpy array
print(arr)

# Check the dimensions of arr
print(arr.shape)

```

**输出:**

```py
0 
()

```

`shape`方法返回了一个空元组 **()** ，其中包含零个元素，表示数组的维数为零。

### 3.检查一个一维但没有元素的 numpy 数组的维数

```py
# Import Python NumPy module
import numpy as np

# Define a numpy array from an empty list
arr = np.array([])

# Print the numpy array
print(arr)

# Check the dimensions of arr
print(arr.shape)

```

**输出:**

```py
[] 
(0,)

```

`shape`方法返回了一个元组 **(0，)**，其中一个元素表示数组只有一个包含零个元素的维度。

## 总结

在本教程中，我们学习了如何使用 Python 中的`shape`方法来找出 Python 对象的维度(NumPy 数组或 Pandas DataFrame)。