# Python 字符串连接()函数

> 原文：<https://www.askpython.com/python/string/python-string-join-method>

在本文中，我们将看看 **Python 字符串连接()函数**。顾名思义，它用于将字符串连接在一起，并为字符串类型的数据工作。

* * *

## 了解 Python 字符串连接()方法

[Python String](https://www.askpython.com/python/string) 有各种内置函数来处理 String 类型的数据。

`join()`方法主要用于**通过另一组分隔符/字符串元素**连接输入字符串。它接受 **iterables** 如 [set](https://www.askpython.com/python/set) ， [list](https://www.askpython.com/python/list/python-list) ， [tuple](https://www.askpython.com/python/tuple/python-tuple) ，string 等和另一个 **string** (可分离元素)作为参数。

join()函数返回一个字符串，**将 iterable 的元素与作为参数传递给函数的分隔符字符串**连接起来。

**语法:**

```py
separator-string.join(iterable)

```

**例 1:**

```py
inp_str='JournalDev'
insert_str='*'
res=insert_str.join(inp_str)
print(res)

```

**输出:**

```py
J*o*u*r*n*a*l*D*e*v

```

**例 2:**

```py
inp_str='PYTHON'
insert_str='#!'
res=insert_str.join(inp_str)
print(res)

```

**输出:**

```py
P#!Y#!T#!H#!O#!N

```

嘿，伙计们！需要考虑的最重要的一点是， **join()函数只对字符串类型的输入值**起作用。如果我们输入**非字符串类型**的任何一个参数，就会引发一个`TypeError exception`。

**举例:**

```py
inp_str=200  #non-string type input
insert_str='S' 
res=insert_str.join(inp_str)
print(res)

```

在上面的示例中，分隔符字符串(即 insert_str)被赋予了一个整数值。因此，它会引发一个 TypeError 异常。

**输出:**

```py
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
<ipython-input-11-ef2dcbcf6abf> in <module>
      1 inp_str=200  #non-string type input
      2 insert_str='S'
----> 3 res=insert_str.join(inp_str)
      4 print(res)

TypeError: can only join an iterable

```

**以** `list` **为可迭代**的 Python string join()方法:

**语法:**

```py
separator-string.join(list)

```

**举例:**

```py
inp_lst=['10','20','30','40']
sep='@@'
res=sep.join(inp_lst)
print(res)

```

在上面的示例中，分隔符字符串“@@”被连接到输入列表的每个元素，即 inp_lst。

**输出:**

```py
[email protected]@[email protected]@[email protected]@40

```

**Python join()方法有** `set` **一个 iterable** :

**语法:**

```py
separator-string.join(set)

```

**举例:**

```py
inp_set=('10','20','30','40')
sep='**'
sep1='<'
res=sep.join(inp_set)
print(res)
res1=sep1.join(inp_set)
print(res1)

```

在上面的例子中，分隔符字符串“**”和“

**输出:**

```py
10**20**30**40
10<20<30<40

```

**Python join()方法用** `dictionary` **作为 iterable:**

Python string join()方法也可以作为 iterable 应用于[字典](https://www.askpython.com/python/dictionary/python-dictionary-dict-tutorial)。

但是，需要注意的重要一点是， **join()方法只作用于字典**数据结构的键，而不是与键相关联的值。

**语法:**

```py
separator-string.join(dict)

```

**例 1:**

```py
inp_dict={'Python':'1','Java':'2','C++':'3'}
sep='##'
res=sep.join(inp_dict)
print(res)

```

从上面的例子可以看出，join()方法只考虑 dict 的键来进行操作。它完全忽略了格言的价值。

**输出:**

```py
Python##Java##C++

```

**例 2:**

```py
inp_dict={'Python':1,'Java':2,'C++':3}
sep='##'
res=sep.join(inp_dict)
print(res)

```

在上面的例子中，dict 中的值是非字符串类型的。不过，这不会导致代码执行出错，因为 join()方法只处理字典的键。

**输出:**

```py
Python##Java##C++

```

**例 3:**

```py
inp_dict={1:'Python',2:'Java',3:'C++'}
sep='##'
res=sep.join(inp_dict)
print(res)

```

上面的代码返回一个 **TypeError** ，因为与字典相关联的键值是非字符串类型的。

**输出:**

```py
TypeError                                 Traceback (most recent call last)
<ipython-input-34-bb7356c41bc8> in <module>
      1 inp_dict={1:'Python',2:'Java',3:'C++'}
      2 sep='##'
----> 3 res=sep.join(inp_dict)
      4 print(res)

TypeError: sequence item 0: expected str instance, int found

```

* * *

## Python numpy.join()方法

[Python NumPy 模块](https://www.askpython.com/python-modules/numpy/python-numpy-arrays)内置了处理数组中字符串数据的函数。

`numpy.core.defchararray.join(sep-string,inp-arr)`用于将数组的元素与作为参数传递的分隔符字符串连接起来。

它接受包含字符串类型元素和分隔符字符串的数组作为参数，**返回包含由输入分隔符字符串(分隔符)**分隔的元素的数组。

**语法:**

```py
numpy.core.defchararray.join(separator-string,array)

```

**例 1:**

```py
import numpy as np
inp_arr=np.array(["Python","Java","Ruby","Kotlin"])
sep=np.array("**")
res=np.core.defchararray.join(sep,inp_arr)
print(res)

```

在上面的例子中，我们已经使用`numpy.array()`方法从传递的列表元素中生成了一个数组。此外，通过使用 join()函数，它将字符串“**”连接到数组的每个元素。

**输出:**

```py
['P**y**t**h**o**n' 'J**a**v**a' 'R**u**b**y' 'K**o**t**l**i**n']

```

**例 2:**

```py
import numpy as np
inp_arr=np.array(["Python","Java","Ruby","Kotlin"])
sep=np.array(["**","++","&&","$$"])
res=np.core.defchararray.join(sep,inp_arr)
print(res)

```

在上面的例子中，我们为数组的每个元素使用了不同的单独的字符串。唯一的条件是数组中可分离字符串(分隔符)的数量应该与输入数组中元素的数量相匹配。

**输出:**

```py
['P**y**t**h**o**n' 'J++a++v++a' 'R&&u&&b&&y' 'K$$o$$t$$l$$i$$n']

```

* * *

## Python Pandas str.join()方法

[Python Pandas 模块](https://www.askpython.com/python-modules/pandas/python-pandas-module-tutorial)有内置的 pandas.str.join()方法，用提供的分隔符连接数据集的元素。

`pandas.str.join()`方法处理数据集或输入序列的特定列(数据)值，并返回带有分隔符字符串或分隔符的连接数据项的序列。

**语法:**

```py
Series.str.join(delimiter or separator-string)

```

**输入。csv 文件:Book1.csv**

![Input csv file-Book1](img/2bc47b2e441c3014168c458c5ab388b5.png)

**Input csv file-Book1**

**举例:**

```py
import pandas
info=pandas.read_csv("C:\\Book1.csv")
info["Name"]=info["Name"].str.join("||")
print(info)

```

在上面的例子中，我们已经使用了`pandas.read_csv()`方法来读取数据集的内容。此外，我们将分隔符字符串，即“||”连接到输入数据集的列“Name”的数据值。

**输出:**

```py
           Name  Age
0        J||i||m   21
1  J||e||n||n||y   22
2     B||r||a||n   24
3  S||h||a||w||n   12
4  R||i||t||i||k   26
5     R||o||s||y   24
6  D||a||n||n||y   25
7  D||a||i||s||y   15
8        T||o||m   27

```

* * *

## 摘要

*   join()方法用于使用字符串分隔符元素**连接字符串类型的元素或 iterables。**
*   参数:iterable 元素和分隔符必须是`string type`。
*   而且，Python join()方法还可以应用于集合、列表、字典等可迭代对象。
*   如果分隔符或输入 iterable 包含非字符串类型的元素，join()方法会引发一个`TypeError exception`。

* * *

## 结论

因此，在本文中，我们已经了解了 Python String join()方法对不同可重复项(如集合、列表、元组、字典等)的工作原理。

* * *

## 参考

*   Python 字符串 join()方法–journal dev
*   Python 字符串函数