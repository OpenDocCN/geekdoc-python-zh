# Python slice()函数

> 原文：<https://www.askpython.com/python/built-in-methods/python-slice-function>

*Python slice()函数*根据传递给它的参数，从用户指定的输入索引集中返回一个切片对象。

因此，它使用户能够切片任何序列，如[列表](https://www.askpython.com/python/list/python-list)、[元组](https://www.askpython.com/python/tuple/python-tuple)、[字符串](https://www.askpython.com/python/string/python-string-functions)等。

**语法:**

```py
slice(Stop)
slice(Start, Stop[, Step)
```

*   **Start** :(可选)一个整数，指定开始切片过程的索引。
*   **Stop** :一个整数，指定 slice()方法的结束索引。
*   **步骤**:(可选)一个整数，指定切片过程的步骤。

**slice()函数返回的值:**

切片的物体。

* * *

## 对 slice()函数的基本理解

**举例:**

```py
print("Printing arguments passed to the slice().... ")
input = slice(4)  
print(input.start)
print(input.stop)
print(input.step)

input = slice(1,4,6)  
print(input.start)
print(input.stop)
print(input.step)

```

**输出:**

```py
Printing arguments passed to the slice().... 
None
4
None
1
4
6
```

* * *

## 带字符串的 Python slice()

Python slice()函数可以以两种不同的方式与字符串一起使用:

*   *带正索引的 slice()函数*
*   *带负索引的 slice()函数*

### 1.具有正索引的 slice()函数

**举例:**

```py
input='Engineering'
result=input[slice(1,6)]
print(result)

```

**输出:**

```py
ngine
```

### 2.具有负索引的 slice()函数

**举例:**

```py
input='Engineering'
result=input[slice(-5,-1)]
print(result)

```

**输出:**

```py
erin
```

* * *

## 带列表的 Python slice()

**举例**:

```py
input_list = slice(1, 5) 
my_list = ['Safa', 'Aman', 'Raghav', 'Raman', 'JournalDev', 'Seema']
print(my_list[input_list])

```

**输出:**

```py
['Aman', 'Raghav', 'Raman', 'JournalDev']
```

* * *

## 带元组的 Python slice()

**举例:**

```py
input_tuple = slice(1, 5)  
my_tuple = ['Safa', 'Aman', 'Raghav', 'Raman', 'JournalDev', 'Seema']
print(my_tuple[input_tuple])

```

**输出:**

```py
['Aman', 'Raghav', 'Raman', 'JournalDev']
```

* * *

## 使用 Python slice()扩展索引

一个*速记方法*可以用来提供 Python slice()的功能。

**语法:**

```py
input[start:stop:step]
```

**举例:**

```py
my_tuple = ['Safa', 'Aman', 'Raghav', 'Raman', 'JournalDev', 'Seema']
result = my_tuple[1:3] 
print(result)

```

**输出:**

```py
['Aman', 'Raghav']
```

* * *

## 删除 Python 切片

**del 关键字**可用于删除特定输入元素上应用的切片。

**举例:**

```py
my_tuple = ['Safa', 'Aman', 'Raghav', 'Raman', 'JournalDev', 'Seema']

del my_tuple[:2]
print(my_tuple)

```

**输出:**

```py
['Raghav', 'Raman', 'JournalDev', 'Seema']
```

* * *

## 结论

因此，在本文中，我们已经了解了 Python slice()函数的基本功能。

* * *

## 参考

*   [Python slice()文档](https://docs.python.org/3/c-api/slice.html)
*   Python slice()函数