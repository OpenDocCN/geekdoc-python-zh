# Python 中遍历列表的方法

> 原文：<https://www.askpython.com/python/list/iterate-through-list-in-python>

在本教程中，我们将学习如何在 Python 中遍历 list。 [Python List](https://www.askpython.com/python/list/python-list) 基本上是一个`ordered data structure`，它使我们能够存储和操作其中的数据。

在 Python 中，可以引用以下任何一种方法来迭代列表:

*   **使用 Python range()方法**
*   **列表理解**
*   **使用 Python 枚举()方法**
*   **通过使用 for 循环**
*   **通过使用 while 循环**
*   **使用 Python NumPy 模块**
*   **使用λ函数**

* * *

## 1.使用 range()方法在 Python 中遍历列表

Python 的`range()`方法可以与 for 循环结合使用，在 Python 中遍历和迭代一个列表。

range()方法基本上返回一个`sequence of integers`，即它从提供的起始索引到参数列表中指定的结束索引构建/生成一个整数序列。

**语法:**

```py
range (start, stop[, step])

```

*   `start`(上限):该参数用于为将要生成的整数序列提供起始值/索引。
*   `stop`(下限):该参数用于提供要生成的整数序列的结束值/索引。
*   `step`(可选):提供要生成的序列中每个整数之间的差值。

**range()函数**生成从起始值到结束/停止值的整数序列，但不包括序列中的结束值，即**不包括结果序列**中的停止号/值。

**举例:**

```py
lst = [10, 50, 75, 83, 98, 84, 32]

for x in range(len(lst)): 
	print(lst[x]) 

```

在上面的代码片段中，使用 range()函数迭代列表，该函数遍历 **0(零)到定义的列表长度**。

**输出:**

```py
10
50
75
83
98
84
32

```

* * *

## 2.使用 for 循环遍历 Python 中的列表

[Python for loop](https://www.askpython.com/python/python-for-loop) 可以用来直接遍历列表。

**语法:**

```py
for var_name in input_list_name:

```

**举例:**

```py
lst = [10, 50, 75, 83, 98, 84, 32] 

for x in lst: 
	print(x) 

```

**输出:**

```py
10
50
75
83
98
84
32

```

* * *

## 3.列表理解在 Python 中遍历列表

Python 列表理解是一种生成具有特定属性或规范的元素列表的不同方式，即它可以识别输入是否是列表、字符串、元组等。

**语法:**

```py
[expression/statement for item in input_list]

```

**举例:**

```py
lst = [10, 50, 75, 83, 98, 84, 32] 

[print(x) for x in lst] 

```

**输出:**

```py
10
50
75
83
98
84
32

```

* * *

## 4.用 while 循环遍历 Python 中的列表

[Python while 循环](https://www.askpython.com/python/python-while-loop)也可以用来以类似 for 循环的方式迭代列表。

**语法:**

```py
while(condition) :
	Statement
        update_expression

```

**举例:**

```py
lst = [10, 50, 75, 83, 98, 84, 32]

x = 0

# Iterating using while loop 
while x < len(lst): 
	print(lst[x]) 
	x = x+1

```

**输出:**

```py
10
50
75
83
98
84
32

```

* * *

## 5.Python NumPy 遍历 Python 中的列表

[Python NumPy 数组](https://www.askpython.com/python-modules/numpy/python-numpy-arrays)也可以用来高效地迭代一个列表。

Python numpy.arange()函数创建一个统一的整数序列。

【numpy.arange()函数的语法:

```py
numpy.arange(start, stop, step)

```

*   `start`:该参数用于为将要生成的整数序列提供起始值/索引。
*   `stop`:该参数用于提供要生成的整数序列的结束值/索引。
*   `step`:提供待生成序列中每个整数之间的差值。

`numpy.nditer(numpy_array)`是一个为我们提供遍历 NumPy 数组的迭代器的函数。

**举例:**

```py
import numpy as np

n = np.arange(16)

for x in np.nditer(n): 
	print(x) 

```

在上面的代码片段中， **np.arange(16)** 创建一个从 0 到 15 的整数序列。

**输出:**

```py
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15

```

* * *

## 6.Python enumerate()方法迭代 Python 列表

Python enumerate()函数可用于以优化的方式迭代列表。

`enumerate()`函数**将一个计数器添加到列表或任何其他可迭代对象中，并通过函数将其作为枚举对象**返回。

因此，**减少了在迭代操作**时保持元素计数的开销。

**语法:**

```py
enumerate(iterable, start_index)

```

*   `start_index`:为迭代 iterable 记录计数器的元素的索引。

**举例:**

```py
lst = [10, 50, 75, 83, 98, 84, 32]

for x, res in enumerate(lst): 
	print (x,":",res) 

```

**输出:**

```py
0 : 10
1 : 50
2 : 75
3 : 83
4 : 98
5 : 84
6 : 32

```

* * *

## 7.使用 lambda 函数迭代 Python 列表

Python 的 lambda 函数基本上都是匿名函数。

**语法:**

```py
lambda parameters: expression

```

*   `expression`:待评估的 iterable。

lambda 函数和 Python map()函数可用于轻松迭代列表。

Python `map()`方法接受一个函数作为参数，并返回一个列表。

map()方法的输入函数被 iterable 的每个元素调用，它返回一个新的列表，其中分别包含该函数返回的所有元素。

**举例:**

```py
lst = [10, 50, 75, 83, 98, 84, 32] 

res = list(map(lambda x:x, lst))

print(res) 

```

在上面的代码片段中， **lambda x:x 函数**作为 map()函数的输入被提供。**lambda x:x 将接受 iterable 的每个元素并返回它**。

input_list ( **lst** )作为 map()函数的第二个参数提供。因此，map()函数将把 **lst** 的每个元素传递给 lambda x:x 函数并返回这些元素。

**输出:**

```py
[10, 50, 75, 83, 98, 84, 32]

```

* * *

## 结论

在这篇文章中。我们已经揭示了迭代 Python 列表的各种技术。

* * *

## 参考

*   遍历 Python 列表——journal dev