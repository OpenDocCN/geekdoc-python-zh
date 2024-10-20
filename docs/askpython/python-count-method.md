# 如何使用 Python count()函数

> 原文：<https://www.askpython.com/python/string/python-count-method>

嗨，伙计们！在本文中，我们将重点关注带有字符串和列表的 Python count()方法。

* * *

## 1.带字符串的 Python count()函数

[Python String](https://www.askpython.com/python/string/python-string-functions) 有一个内置的函数——String . count()方法来计算特定输入字符串中某个字符或子字符串的出现次数。

`string.count() method`接受一个字符或子串作为参数，并返回输入子串在字符串中出现的次数。

**语法:**

```py
string.count(string, start_index,end_index)

```

*   `substring(mandatory)`:输入字符串中需要统计出现次数的字符串。
*   `start_index(optional)`:开始搜索子字符串的索引。
*   `end_index(optional)`:需要停止搜索子字符串的索引。

**举例:**

```py
inp_str = "JournalDev -- AskPython @ JournalDev"
str_cnt = inp_str.count("JournalDev")
print(str_cnt)

```

**输出:**

```py
2

```

**例 2:**

```py
inp_str = "Python Java Python Kotlin"
str_cnt = inp_str.count("Python", 0 , 6)
print(str_cnt)

```

在上面的例子中，我们传递了' **Python** '作为要搜索的子字符串，并计算了在**索引 0–索引 6** 之间的存在。

**输出:**

```py
1

```

**例 3:**

```py
inp_str = "Python Java Python Kotlin"
str_len=len(inp_str)
str_cnt = inp_str.count("Python", 5 , str_len )
print(str_cnt)

```

在这里，我们搜索子字符串–'**Python '**,并计算它在**索引 5 到字符串末尾**之间的出现次数，这就是为什么我们将字符串的长度作为 end_index 参数传递。

**输出:**

```py
1

```

* * *

### Python 字符串计数()方法:TypeError

Python string.count()只接受单个子字符串作为参数。如果我们试图传递多个子字符串作为参数，就会引发`TypeError exception`。

**举例:**

```py
inp_str = "Python Java Python Kotlin"
str_cnt = inp_str.count('Python', 'Java')
print(str_cnt)

```

**输出:**

```py
TypeError                                 Traceback (most recent call last)
<ipython-input-40-6084d1350592> in <module>
      1 inp_str = "Python Java Python Kotlin"
----> 2 str_cnt = inp_str.count('Python', 'Java')
      3 print(str_cnt)

TypeError: slice indices must be integers or None or have an __index__ method

```

* * *

## 2.Python 列表计数()函数

[Python list](https://www.askpython.com/python/list/python-list) 有一个 list.count()方法来计算列表中特定元素的出现次数。

`list.count() method`对输入列表中出现的特定值/数据项进行计数。

**语法:**

```py
list.count(value)

```

**例 1:**

```py
inp_lst = ['Apple','Banana','Apple','Grapes','Jackfruit','Apple']

lst_cnt = inp_lst.count('Apple')
print(lst_cnt)

```

**输出:**

```py
3

```

**例 2:**

```py
inp_lst = [ ['Rat','Cat'], ['Heat','Beat'], ['Rat','Cat'] ]

lst_cnt = inp_lst.count(['Rat','Cat'])
print(lst_cnt)

```

在上面的例子中，我们基本上统计了嵌套列表['Rat '，' Cat']在列表中的出现次数。

**输出:**

```py
2

```

* * *

## Python count()函数一目了然！

*   Python `string.count() function`用于计算特定字符串中输入子字符串的出现次数。
*   如果我们试图将多个子字符串作为参数传递，string.count()方法会引发一个`TypeError exception`。
*   `list.count() function`检查特定元素在特定列表中出现的次数。

* * *

## 结论

因此，在本文中，我们已经了解了内置 Python count 函数对字符串和列表的处理。

* * *

## 参考

*   Python 字符串计数()函数–journal dev