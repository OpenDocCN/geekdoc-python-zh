# Python 字符串查找()

> 原文：<https://www.askpython.com/python/string/python-string-find>

Python string **find()** 方法用于检查一个字符串是否是另一个字符串的子字符串。

通常，当我们想要检查一个字符串是否包含另一个字符串时，`find()`对于解决这个问题非常有用。

让我们通过一些简单的例子来理解如何使用它。

* * *

## Python 字符串 Find()的语法

格式:

```py
ret = str.find(sub, start, end)

```

这里， **find()** 在一个字符串对象`str`上被调用并返回一个整数。这将检查子串`sub`是否位于可选参数`start`和`end`之间。

**注**:含`start`，`end`为*不含*。

`find()`不接受关键字参数，所以你必须只提到位置参数。

如果位置之间不存在子串，那么`ret`就是`-1`。否则，它是子串的第一个**匹配在输入字符串上的位置。**

让我们看一个简单的例子。

```py
test_string = "Hello from AskPython"

# We will check if the substring
# 'AskPython' exists in test_string
print(test_string.find('AskPython', 0, len(test_string)) )

```

**输出** : 11

由于子串存在，所以返回子串第一个字符的位置(' AskPython '中的' A ')，即 11。

我们也可以使用负索引来表示从字符串末尾的偏移量。举个例子，

```py
test_string = "Hello from AskPython"

# Will search from the first to the 
# second last position
print(test_string.find('AskPython', 0, -1))

# Will search from index 5 to len(str) - 1
print(test_string.find('from', 5, -1))

```

**输出**

```py
-1
6

```

对于第一个实例，输出是负的，因为 find()方法不能完全匹配字符串“AskPython”。

原因是因为-1，find 字符串只搜索到**askphyto**，这意味着我们要搜索的东西并没有真正找到。

### 使用不带开始/结束参数的 Python 字符串 find()

如果我们希望搜索整个字符串，我们可以将`start`和`end`参数留空。这是许多程序员使用 Python string find()最广泛的方式。

我们可以将前面的例子改写为:

```py
test_string = "Hello from AskPython"
print(test_string.find('AskPython'))

```

这将给出与以前相同的输出。

### 使用只带有开始参数的 find()

我们只能通过省略`end`参数来指定起始位置，直到字符串结束。

```py
test_string = "Hello from AskPython"

# Will print 11
print(test_string.find('AskPython', 0))

# Will print -1, as the search starts
# from position 12, i.e from 'k'
print(test_string.find('AskPython', 12))

```

**输出**

```py
11
-1

```

* * *

## 结论

在本文中，我们学习了搜索子字符串的`str.find()`方法。

## 参考

*   【String.find()的 Python 文档

* * *