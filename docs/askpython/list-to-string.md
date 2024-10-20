# Python 列表到字符串

> 原文：<https://www.askpython.com/python/list/list-to-string>

在本教程中，我们将 Python 列表到字符串的转换。一个 [Python 列表](https://www.askpython.com/python/list/python-list)用于表示元素以供操作。它基本上代表了同质元素的集合。

[Python 字符串](https://www.askpython.com/python/string/python-string-functions)也用于以字符形式收集元素作为输入。

列表中的元素可以通过下列方法之一转换为字符串:

*   **通过使用 join()方法**
*   **通过使用列表理解**
*   **使用 for 循环进行迭代**
*   **通过使用 map()方法**

* * *

### 1.使用 join()方法将 Python 列表转换为字符串

Python join()方法可用于在 Python 中将列表转换为字符串。

`join()`方法接受 iterables 作为参数，比如[列表](https://www.askpython.com/python/list/python-list)、[元组](https://www.askpython.com/python/tuple/python-tuple)、[字符串](https://www.askpython.com/python/string/python-string-functions)等。此外，它返回一个新的字符串，该字符串包含从 iterable 作为参数连接的元素。

**注意**:join()方法的强制条件是传递的 iterable 应该包含 string 元素。如果 iterable 包含一个整数，它会引发一个**类型错误异常**。

**语法:**

```py
string.join(iterable)

```

**举例:**

```py
inp_list = ['John', 'Bran', 'Grammy', 'Norah'] 
out_str = " "
print("Converting list to string using join() method:\n")
print(out_str.join(inp_list)) 

```

在上面的例子中，join()方法接受 **inp_list** 作为参数，并将列表的元素连接到 **out_str** ，从而返回一个字符串作为输出。

**输出:**

```py
Converting list to string using join() method:

John Bran Grammy Norah

```

* * *

### 2.列表理解和 join()方法将 Python 列表转换成字符串

Python List Comprehension 从现有列表中创建元素列表。它还使用 for 循环以元素模式遍历 iterable 的项。

Python List Comprehension 和 join()方法可以用来将列表转换成字符串。list comprehension 将逐个元素地遍历元素，join()方法将列表的元素连接成一个新的字符串，并将其表示为输出。

**举例:**

```py
inp_list = ['John', 'Bran', 'Grammy', 'Norah'] 

res = ' '.join([str(item) for item in inp_list]) 
print("Converting list to atring using List Comprehension:\n")
print(res) 

```

**输出:**

```py
Converting list to atring using List Comprehension:

John Bran Grammy Norah

```

* * *

### 3.用 map()函数实现 Python 列表到字符串的转换

Python 的 map()函数可用于将列表转换为字符串。

`map()`函数接受函数和可迭代对象，如列表、元组、字符串等。接下来，map()函数用提供的函数映射 iterable 的元素。

**语法:**

```py
map(function, iterable)

```

**举例:**

```py
inp_list = ['John', 'Bran', 'Grammy', 'Norah'] 

res = ' '.join(map(str, inp_list)) 
print("Converting list to string using map() method:\n")
print(res) 

```

在上面的代码片段中， **map(str，inp_list)** 函数接受 **str** 函数和 **inp_list** 作为参数。它将输入 iterable( list)的每个元素映射到给定的函数，并返回元素列表。此外，join()方法用于将输出设置为字符串形式。

**输出:**

```py
Converting list to string using map() method:

John Bran Grammy Norah

```

* * *

### 4.使用 for 循环将 Python 列表转换为字符串的迭代

在这种技术中，输入列表的**元素被逐一迭代，并被添加到一个新的空字符串**中。因此，将列表转换为字符串。

**举例:**

```py
inp_str = ['John', 'Bran', 'Grammy'] 
st = "" 	
for x in inp_str:
  st += x

print(st)

```

**输出:**

```py
JohnBranGrammy

```

* * *

## 将字符列表转换为字符串

即使是列表形式的一组字符也可以用与上面相同的方式转换成字符串。下面是一个例子，演示如何将一列字符转换成一个字符串。

**举例:**

```py
inp_str = ['J', 'o', 'u', 'r', 'n', 'a', 'l', 'd', 'e', 'v']
st = ""
for x in inp_str: 
  st += x

print(st)

```

**输出:**

```py
Journaldev

```

* * *

## 结论

因此，在本文中，我们研究了将 Python 列表转换为字符串的不同技术和方法。

* * *

## 参考

*   [stack overflow——Python 中列表到字符串的转换](https://stackoverflow.com/questions/5618878/how-to-convert-list-to-string)