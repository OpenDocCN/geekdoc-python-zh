# Python 中的字符串操作

> 原文：<https://www.pythonforbeginners.com/basics/string-manipulation-in-python>

Python 字符串是处理文本数据最有效的工具之一。在本文中，我们将讨论 python 字符串和 Python 中字符串操作的基础知识。

## Python 中的字符串是什么？

一个 [python 字符串](https://www.pythonforbeginners.com/basics/python-string-methods-for-string-manipulation)是一个按顺序排列的字符列表。字符是你可以在键盘上一键输入的任何东西，
像一个字母、一个数字或一个反斜杠。

字符串可以有空格:

```py
"hello world".
```

空字符串是包含 0 个字符的字符串。

Python 字符串是不可变的

Python 将所有由引号
(" "或' ')分隔的内容都识别为字符串。

## Python 中的字符串操作

为了操作字符串，我们可以使用 Pythons 的一些内置方法。

### 创建字符串

要创建具有给定字符的字符串，您可以在将字符用双引号或单引号括起来后将其赋给变量，如下所示。

```py
word = "Hello World"

>>> print word
Hello World
```

### 访问字符串中的字符

要访问字符串中的字符，我们可以使用 [python 索引](https://www.pythonforbeginners.com/strings/string-indexing-in-python)操作符[ ]即方括号来访问字符串中的字符，如下所示。

```py
word = "Hello World"
letter=word[0]

>>> print letter
H
```

### 查找字符串的长度

要找到一个字符串的长度，我们可以使用 len()函数。len()函数将一个字符串作为输入参数，并返回字符串的长度，如下所示。

```py
word = "Hello World"

>>> len(word)
11
```

### 在字符串中查找字符

要查找字符串中某个字符的索引，我们可以使用 find()方法。在字符串上调用 find()方法时，该方法将该字符作为其输入参数，并返回该字符第一次出现的索引，如下所示。

```py
 >>> word = "Hello World" 
>>> print word.find("H") # find the word H in the string
0
```

您还可以在 python 中执行字符串操作，以查找字符串中某个字符的出现频率。为此，我们可以使用 count()方法。在字符串上调用 count()方法时，该方法将一个字符作为其输入参数，并返回该字符的频率，如下所示。

```py
>>> word = "Hello World"
>>> print word.count('l') # count how many times l is in the string
3
```

还可以使用 index()方法在字符串中查找字符或子字符串的索引。在字符串上调用 index()方法时，该方法将一个字符或子字符串作为其输入参数，并返回该字符或子字符串第一次出现的索引，如下所示。

```py
>>> word = "Hello World"
>>> print word.index("World") # find the letters World in the string
6
```

### 计算字符串中的空格数

要计算字符串中的空格数，可以将空格字符传递给 count()方法，如下所示。

```py
s = "Count, the number of spaces"

>>> print s.count(' ')
8
```

### [字符串切片](https://www.pythonforbeginners.com/strings/string-slicing-in-python)

要在 Python 中执行字符串操作，可以使用语法 string _ name[start _ index:end _ index]来获取字符串的子串。这里，切片操作为我们提供了一个子字符串，其中包含从字符串 string_name 的 start_index 到 end_index-1 的字符。

请记住，python 和许多其他语言一样，是从 0 开始计数的！！

```py
word = "Hello World"

print word[0] #get one char of the word
print word[0:1] #get one char of the word (same as above)
print word[0:3] #get the first three char
print word[:3] #get the first three char
print word[-3:] #get the last three char
print word[3:] #get all but the three first char
print word[:-3] #get all but the three last character

word = "Hello World"

word[start:end] # items start through end-1
word[start:] # items start through the rest of the list
word[:end] # items from the beginning through end-1
word[:] # a copy of the whole list
```

### Python 中的拆分字符串

您可以使用 split()方法来分割字符串，以便在 Python 中执行字符串操作。在字符串上调用 split()方法时，该方法将一个字符作为其输入参数。执行后，它在指定字符处拆分字符串，并返回子字符串列表，如下所示。

```py
word = "Hello World"

>>> word.split(' ') # Split on whitespace
['Hello', 'World']
```

在上面的例子中，我们在空格字符处分割了字符串。

### 检查字符串是以字符开头还是以字符结尾

要检查字符串是以特定字符开头还是结尾，可以分别使用 starts with()或 ends with()方法。

对字符串调用 startswith()方法时，该方法将一个字符作为输入参数。如果字符串以给定字符开头，则返回 True。否则，它返回 False。

在字符串上调用 endswith()方法时，它接受一个字符作为输入参数。如果字符串以给定字符结尾，则返回 True。否则，它返回 False。您可以在下面的示例中观察到这一点。

```py
word = "hello world"

>>> word.startswith("H")
True

>>> word.endswith("d")
True

>>> word.endswith("w")
False
```

### 多次重复字符串

您可以使用乘法运算符多次重复一个字符串。当我们将任何给定的字符串或字符乘以一个正数 N 时，它会重复 N 次。您可以在下面的示例中观察到这一点。

```py
print "."* 10 # prints ten dots

>>> print "." * 10
.......... 
```

### [在 Python 中替换字符串中的子串](https://www.pythonforbeginners.com/basics/remove-substring-from-string-in-python)

还可以使用 replace()方法将一个子字符串替换为另一个子字符串。当在字符串上调用 replace()方法时，将被替换的子字符串作为第一个输入参数，将替换字符串作为第二个输入参数。执行后，它用替换字符串替换指定的子字符串，并返回修改后的字符串。您可以使用 replace()方法在 Python 中执行字符串操作，如下所示。

```py
word = "Hello World"

>>> word.replace("Hello", "Goodbye")
'Goodbye World'
```

### 更改大小写字符串

可以使用 upper()、lower()和 title()方法将 string 转换成大写、小写和 titlecase。

upper()方法在字符串上调用时，会将字符串变为大写并返回修改后的字符串。

lower()方法在字符串上调用时，会将字符串变为小写，并返回修改后的字符串。

title()方法在字符串上调用时，会将字符串更改为 titlsecase 并返回修改后的字符串。

还可以使用 capital()和 swapcase()方法将字符串大写或交换字符串中字符的大小写。

在字符串上调用 capitalize()方法时，会将字符串的第一个字符大写，并返回修改后的字符串。

在字符串上调用 swapcase()方法时，会将小写字符转换为大写字符，反之亦然。执行后，它返回修改后的字符串。

您可以在下面的例子中观察这些用例。

```py
string = "Hello World"

>>> print string.upper()
HELLO WORLD

>>> print string.lower()
hello world

>>> print string.title()
Hello World

>>> print string.capitalize()
Hello world

>>> print string.swapcase()
hELLO wORLD
```

### [在 Python 中反转一个字符串](https://www.pythonforbeginners.com/basics/how-to-reverse-a-string-in-python)

要反转字符串，可以使用 reversed()函数和 join()方法。

reversed()函数将一个字符串作为其输入参数，并以相反的顺序返回一个包含输入字符串字符的列表。

join()方法在分隔符字符串上调用时，将字符列表作为其输入参数，并使用分隔符连接列表中的字符。执行后，它返回结果字符串。

要使用 reversed()函数和 join()方法反转一个字符串，我们将首先使用 reversed()函数创建一个逆序字符列表。然后，我们将使用一个空字符串作为分隔符，并在空字符串上调用 join()方法，将字符列表作为输入参数。在执行 join()方法之后，我们将得到一个新的反向字符串，如下所示。

```py
string = "Hello World"

>>> print ''.join(reversed(string))
dlroW olleH
```

### 在 Python 中剥离字符串

Python 字符串具有 strip()、lstrip()、rstrip()方法，用于移除字符串两端的任何字符。

strip()方法在字符串上调用时，将一个字符作为其输入参数，并从字符串的开头(左)和结尾(右)删除该字符。如果没有指定要删除的字符，则空白字符将被删除。

在字符串上调用 lstrip()方法时，将一个字符作为其输入参数，并从字符串的开头(左侧)删除该字符。

在字符串上调用 rstrip()方法时，该方法将一个字符作为其输入参数，并从字符串的末尾(右侧)删除该字符。

```py
word = "Hello World"
```

通过将“\n”作为输入参数传递给 rstrip()方法，可以从字符串末尾去掉换行符。

```py
>>> print word.strip('\n')
Hello World

strip() #removes from both ends
lstrip() #removes leading characters (Left-strip)
rstrip() #removes trailing characters (Right-strip)

>>> word = " xyz "

>>> print word
xyz

>>> print word.strip()
xyz

>>> print word.lstrip()
xyz

>>> print word.rstrip()
xyz
```

### [在 Python 中连接字符串](https://www.pythonforbeginners.com/concatenation/string-concatenation-and-formatting-in-python)

要在 Python 中连接字符串，请使用“+”运算符，如下所示。

```py
"Hello " + "World" # = "Hello World"
"Hello " + "World" + "!"# = "Hello World!"
```

如上所述，join()方法在分隔符字符串上调用时，将字符列表作为其输入参数，并使用分隔符连接列表中的字符。执行后，它返回结果字符串。

```py
>>> print ":".join(word) # #add a : between every char
H:e:l:l:o: :W:o:r:l:d

>>> print " ".join(word) # add a whitespace between every char
H e l l o W o r l d 
```

### 测试

Python 中的字符串可以测试真值。

返回类型将是布尔值(真或假)

```py
word = "Hello World"

word.isalnum() #check if all char are alphanumeric 
word.isalpha() #check if all char in the string are alphabetic
word.isdigit() #test if string contains digits
word.istitle() #test if string contains title words
word.isupper() #test if string contains upper case
word.islower() #test if string contains lower case
word.isspace() #test if string contains spaces
word.endswith('d') #test if string endswith a d
word.startswith('H') #test if string startswith H
```

## 结论

在本文中，我们讨论了在 Python 中执行字符串操作的不同方法。要了解更多关于 python 编程的知识，你可以阅读这篇关于 Python 中的[列表理解的文章。你可能也会喜欢这篇关于如何](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python)[用 python](https://codinginfinite.com/python-chat-application-tutorial-source-code/) 构建聊天机器人的文章。