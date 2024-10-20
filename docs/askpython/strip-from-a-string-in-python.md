# 用 Python 从字符串中剥离

> 原文：<https://www.askpython.com/python/string/strip-from-a-string-in-python>

让我们看看如何在 Python 中从字符串中剥离。

Python 为我们提供了不同的方法来从一个 [Python 字符串](https://www.askpython.com/python/string/python-string-functions)中删除尾部字符，比如换行符、空格和制表符。这叫做**从管柱上剥离**。

* * *

## 如何在 Python 中从字符串中剥离

我们可以使用以下任何一种方法从字符串中剥离:

*   **strip()**–这将删除前导的**和**结尾的空格(" ")、制表符(" \t ")和换行符(" \n ")，并返回经过修剪的字符串。
*   **rst rip()**–这将删除任何**尾随的**空格、制表符和换行符，并返回经过修剪的字符串。由于我们只在右边修剪，这被恰当地称为`rstrip()`。
*   **lstrip()**–我们只修剪前导字符并返回修剪后的字符串。因为这只会修剪最左边的字符，所以称为`lstrip()`。

有字符串方法，所以我们在字符串对象上调用它们。它们不带任何参数，因此调用它们的语法是:

```py
# Strip from both left and right
my_string.strip()

# Strip only from the right
my_string.rstrip()

# Strip only from the left
my_string.lstrip()

```

让我们举一个例子来形象化地说明这一点:(我们放置一个名为“ **_ENDOFSTRING** ”的字符串结束标记，以查看是否删除了尾随空格和制表符。

```py
my_string = "\t  Hello from JournalDev\t   \n"

print("Original String (with tabs, spaces, and newline)", my_string + "_ENDOFSTRING")

print("After stripping leading characters using lstrip(), string:", my_string.lstrip() + "_ENDOFSTRING")

print("After stripping trailing characters using rstrip(), string:", my_string.rstrip() + "_ENDOFSTRING")

print("After stripping both leading and trailing characters using strip(), string:", my_string.strip() + "_ENDOFSTRING")

```

**输出**

```py
Original String (with tabs, spaces, and newline)          Hello from JournalDev    
_ENDOFSTRING
After stripping leading characters using lstrip(), string: Hello from JournalDev           
_ENDOFSTRING
After stripping trailing characters using rstrip(), string:       Hello from JournalDev_ENDOFSTRING
After stripping both leading and trailing characters using strip(), string: Hello from JournalDev_ENDOFSTRING

```

注意，在`lstrip()`的情况下，尾随字符(连同换行符)仍然存在，而在`rstrip()`和`strip()`中它们被删除了。

* * *

## 结论

在本文中，我们学习了如何使用各种方法从 Python 中的字符串中剥离。

* * *

## 参考

*   JournalDev 关于修剪字符串的文章

* * *