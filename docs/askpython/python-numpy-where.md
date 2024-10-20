# Python numpy.where()方法的最终指南

> 原文：<https://www.askpython.com/python-modules/numpy/python-numpy-where>

嘿，伙计们！在本文中，我们将关注 Python numpy.where()方法的**工作方式。**

* * *

## numpy.where()函数的工作原理

[Python NumPy 模块](https://www.askpython.com/python-modules/numpy/python-numpy-arrays)包含许多内置函数来创建和操作数组元素。

`numpy.where() function`用于根据某个**条件**返回数组元素

**语法:**

```py
numpy.where(condition,a,b)

```

*   `condition`:需要提及对阵列应用的操作条件。
*   `a`:如果满足条件，即条件为真，则函数产生一个。
*   `b`:如果不满足条件，函数返回该值。

**例 1:**

```py
import numpy as np 

data = np.array([[10,20,30], [40,50,60], [0,1,2]]) 

print(np.where(data<20,True,False)) 

```

在上面的例子中，对于所有数据值为 **< 20** 的数组元素，这些数据值都被替换为 **True** 。并且，对于数据值为 **> 20** 的所有数组元素，即不满足条件的值被替换为**假**。

**输出:**

```py
[[ True False False]
 [False False False]
 [ True  True  True]]

```

**例 2:**

```py
import numpy as np 

data = np.array([[10,20,30], [40,50,60], [0,1,2]]) 

data_set = np.where(data<20) 
print("Data elements less than 20:\n")

print(data[data_set]) 

```

在上面的例子中，我们已经显示了所有小于 20 的数组元素。

**输出:**

```py
Data elements less than 20:

[10  0  1  2]

```

* * *

## 具有多个条件的 Python numpy.where()函数

多个条件可以与`numpy.where() function`一起应用，以针对多个条件操作数组元素。

**语法:**

```py
numpy.where((condition1)&(condition2))
                  OR
numpy.where((condition1)|(condition2))

```

**例 1:**

```py
import numpy as np 

data = np.array([[10,20,30], [40,50,60], [0,1,2]]) 

data_set = np.where((data!=20)&(data<40)) 

print(data[data_set]) 

```

在本例中，我们显示了所有数据值小于 40 且不等于 20 的数组元素。

**输出:**

```py
[10 30  0  1  2]

```

**例 2:**

```py
import numpy as np 

data = np.array([[10,20,30], [40,50,60], [0,1,2]]) 

data_set = np.where((data<20)|(data>40)) 

print(data[data_set]) 

```

在上面这段代码中，满足上述任一条件的所有数据值都会显示出来，即小于 20 的数组元素和大于 40 的数组元素都会显示出来。

**输出**:

```py
[10 50 60  0  1  2]

```

* * *

## 使用 numpy.where()函数替换数组值

使用 numpy.where()函数，我们可以根据特定条件的满足情况替换这些值。

**语法:**

```py
numpy.where(condition,element1,element2)

```

**举例:**

```py
import numpy as np 

data = np.random.randn(2,3)
print("Data before manipulation:\n")
print(data)
data_set = np.where((data>0),data,0) 
print("\nData after manipulation:\n")
print(data_set) 

```

在本例中，我们用 0 替换了所有数据值小于 0 的数组元素，即不满足上述条件。

**输出:**

```py
Data before manipulation:

[[ 0.47544941 -0.35892873 -0.28972221]
 [-0.9772084   1.04305061  1.84890316]]

Data after manipulation:

[[0.47544941 0\.         0\.        ]
 [0\.         1.04305061 1.84890316]]

```

* * *

## 结论

因此，在本文中，我们已经理解了 Python numpy.where()函数跨各种输入的工作方式。

* * *

## 参考

*   Python numpy.where()函数— JournalDev
*   [Python numpy.where()函数—文档](https://numpy.org/doc/stable/reference/generated/numpy.where.html)