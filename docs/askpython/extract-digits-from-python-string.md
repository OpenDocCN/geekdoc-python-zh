# 从 Python 字符串中提取数字的两种简单方法

> 原文：<https://www.askpython.com/python/string/extract-digits-from-python-string>

读者朋友们，你们好！在本文中，我们将关注从 Python 字符串中提取数字的**方法。那么，让我们开始吧。**

* * *

## 1.利用 isdigit()函数从 Python 字符串中提取数字

Python 为我们提供了`string.isdigit()`来检查字符串中数字的存在。

如果输入字符串中包含数字字符，Python [isdigit()](https://www.askpython.com/python/string/python-string-isdigit-function) 函数返回 **True** 。

**语法**:

```py
string.isdigit()

```

我们不需要给它传递任何参数。作为输出，它根据字符串中是否存在数字字符返回 True 或 False。

**例 1:**

```py
inp_str = "Python4Journaldev"

print("Original String : " + inp_str) 
num = ""
for c in inp_str:
    if c.isdigit():
        num = num + c
print("Extracted numbers from the list : " + num) 

```

在本例中，我们使用 for 循环逐个字符地迭代输入字符串。只要 isdigit()函数遇到一个数字，它就会将它存储到一个名为“num”的字符串变量中。

因此，我们看到的输出如下所示

**输出:**

```py
Original String : Python4Journaldev
Extracted numbers from the list : 4

```

现在，我们甚至可以使用 Python list comprehension 将迭代和 idigit()函数合并成一行。

这样，数字字符被存储到列表“num”中，如下所示:

**例 2:**

```py
inp_str = "Hey readers, we all are here be 4 the time!"

print("Original string : " + inp_str) 

num = [int(x) for x in inp_str.split() if x.isdigit()] 

print("The numbers list is : " + str(num)) 

```

**输出:**

```py
Original string : Hey readers, we all are here be 4 the time!
The numbers list is : [4]

```

* * *

## 2.使用正则表达式库提取数字

[Python 正则表达式库](https://www.askpython.com/python/regular-expression-in-python)称为“**正则表达式库**”，使我们能够检测特定字符的存在，如数字、一些特殊字符等。从一串。

在执行任何进一步的步骤之前，我们需要将 regex 库导入到 python 环境中。

```py
import re

```

此外，我们从字符串中提取数字字符。部分 **'\d+'** 将帮助 findall()函数检测任何数字的存在。

**举例:**

```py
import re
inp_str = "Hey readers, we all are here be 4 the time 1!"

print("Original string : " + inp_str) 

num = re.findall(r'\d+', inp_str) 

print(num)

```

因此，如下所示，我们将从字符串中获得所有数字字符的列表。

**输出:**

```py
Original string : Hey readers, we all are here be 4 the time 1!
['4', '1']

```

* * *

## 结论

到此，我们就结束了这个话题。如果你遇到任何问题，欢迎在下面评论。

我建议大家尝试使用数据结构来实现上面的例子，比如[列表](https://www.askpython.com/python/list/python-list)、[字典](https://www.askpython.com/python/dictionary/python-dictionary-dict-tutorial)等等。

更多与 Python 相关的帖子，敬请关注，在此之前，祝你学习愉快！！🙂