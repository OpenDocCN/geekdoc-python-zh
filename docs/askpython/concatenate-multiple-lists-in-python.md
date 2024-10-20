# Python 中连接多个列表的方法

> 原文：<https://www.askpython.com/python/list/concatenate-multiple-lists-in-python>

在本文中，我们将了解在 Python 中连接多个列表的各种技术。Python 列表为我们提供了一种存储数据和对数据进行操作的方法。

## Python 中连接多个列表的技术

以下任一技术可用于将两个或多个列表连接在一起:

*   **通过使用 itertools 模块**
*   **通过使用 Python 的'+'运算符**
*   **通过使用 Python 的' * '运算符**

* * *

### 1.使用 Python itertools.chain()方法

**Python itertools 模块**为我们提供了 itertools.chain()方法，将多个列表串联在一起。

**`itertools.chain()`** 方法接受不同的可迭代数据，如列表、字符串、元组等，并从中提供元素的线性序列。

该函数的工作与输入数据的数据类型无关。

**语法:**

```py
itertools.chain(list1, list2, ...., listN)

```

**举例:**

```py
import itertools 

x = [10, 30, 50, 70] 
y = [12, 16, 17, 18] 
z = [52, 43, 65, 98] 

opt = list(itertools.chain(x,y,z)) 

print ("Concatenated list:\n",str(opt)) 

```

**输出:**

```py
Concatenated list:
 [10, 30, 50, 70, 12, 16, 17, 18, 52, 43, 65, 98]

```

* * *

### 2.使用 Python“*”运算符

Python **`'*' operator`** 提供了一种更有效的方法来对输入列表进行操作并将它们连接在一起。

它在提供的**索引位置**表示并**展开**数据元素。

**语法:**

```py
[*input_list1, *input_list2, ...., *inout_listN]

```

如前所述，*input_list1，*input_list2 等将包含该列表中给定索引处的元素，按上述顺序排列。

**举例:**

```py
x = [10, 30, 50, 70] 
y = [12, 16, 17, 18] 
z = [52, 43, 65, 98] 

opt = [*x, *y, *z] 

print ("Concatenated list:\n",str(opt)) 

```

**输出:**

```py
Concatenated list:
 [10, 30, 50, 70, 12, 16, 17, 18, 52, 43, 65, 98]

```

* * *

### 3.使用 Python“+”运算符

Python **`'+' operator`** 可以用来将列表连接在一起。

**语法:**

```py
list1 + list2 + .... + listN

```

**举例:**

```py
x = [10, 30, 50, 70] 
y = [12, 16, 17, 18] 
z = [52, 43, 65, 98] 

opt = x+y+z

print ("Concatenated list:\n",str(opt)) 

```

**输出:**

```py
Concatenated list:
 [10, 30, 50, 70, 12, 16, 17, 18, 52, 43, 65, 98]

```

* * *

## 结论

因此，在本文中，我们揭示了用 Python 连接多个列表的不同方法。

* * *

## 参考

**Python 中连接列表的方法**