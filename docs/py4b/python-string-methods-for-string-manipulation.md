# 用于字符串操作的 Python 字符串方法

> 原文：<https://www.pythonforbeginners.com/basics/python-string-methods-for-string-manipulation>

在分析文本数据时，字符串操作是最基本的技能。Python 有许多用于字符串操作的内置方法。在本文中，我们将研究用于字符串操作的最常用的 python 字符串方法。

## 将单词大写的 Python 字符串方法

为了在 python 中大写一个字符串的第一个字母，我们使用了 `capitalize()`方法。`capitalize()`方法返回一个新字符串，其中字符串的第一个字母大写。在此过程中，不会对原始字符串进行任何更改。

示例:

```py
myString="python"
print("Original String:")
print(myString)
newString=myString.capitalize()
print("New modified string:")
print(newString)
print("original string after modification:")
print(myString)
```

输出:

```py
Original String:
python
New modified string:
Python
original string after modification:
python
```

在输出中，我们可以看到新字符串的第一个字母已经被修改，但是调用该方法的字符串没有任何变化。

## Python 字符串方法大写每个单词的第一个字符

要将每个单词的第一个字符转换成大写字母，我们可以使用 title()方法。当在字符串上调用时，它将输入字符串中每个单词的第一个字符大写，并返回一个新字符串和结果。它不会影响原始字符串。

示例:

```py
myString="Python is a great language" 
newString=myString.title()
print("Original string is:")
print(myString)
print("Output is:")
print(newString)
```

输出:

```py
Original string is:
Python is a great language
Output is:
Python Is A Great Language
```

## Python 中如何将字符串转换成小写？

`casefold()`方法在 python 字符串上调用时返回一个新字符串，并将原始字符串的每个字母转换成小写。它不会改变原来的字符串。如果文本包含大写或小写字母的不规则使用，这个 python 字符串方法可用于预处理文本。

示例:

```py
myString="PytHon"
print("Original String:")
print(myString)
newString=myString.casefold()
print("New modified string:")
print(newString)
print("original string after modification:")
print(myString)
```

输出:

```py
Original String:
PytHon
New modified string:
python
original string after modification:
PytHon
```

另一种将字符串转换成小写的方法是`lower()`方法。它还将文本字符串中的字母转换成小写，并返回一个新字符串。

示例:

```py
myString="PytHon"
print("Original String:")
print(myString)
newString=myString.lower()
print("New modified string:")
print(newString)
print("original string after modification:")
print(myString)
```

输出:

```py
Original String:
PytHon
New modified string:
python
original string after modification:
PytHon
```

## Python 中如何把字符串转换成大写？

我们可以使用`upper()`方法将一个输入字符串转换成大写。当对任何字符串调用`upper()` 方法时，它返回一个所有字母都大写的新字符串。它不会改变原来的字符串。

示例:

```py
myString="PytHon"
print("Original String:")
print(myString)
newString=myString.upper()
print("New modified string:")
print(newString)
print("original string after modification:")
print(myString)
```

输出:

```py
Original String:
PytHon
New modified string:
PYTHON
original string after modification:
PytHon
```

还有另一个名为`swapcase()`的方法，它交换输入字符串中每个字母的大小写并返回一个新的字符串。它不会对调用它的输入字符串进行任何更改。

示例:

```py
myString="PytHon"
print("Original String:")
print(myString)
newString=myString.swapcase()
print("New modified string:")
print(newString)
print("original string after modification:")
print(myString)
```

输出:

```py
Original String:
PytHon
New modified string:
pYThON
original string after modification:
PytHon
```

## python 中如何拆分字符串？

为了在 python 中拆分字符串，我们使用了`split()`方法。 [Python split](https://www.pythonforbeginners.com/dictionary/python-split) 方法采用一个可选的分隔符，并在分隔符出现的地方分割输入字符串，并返回一个包含字符串分割部分的列表。

示例:

```py
myString="I am A Python String"
print("Original String:")
print(myString)
newList=myString.split()
print("New List:")
print(newList)
print("when 'A' is declared as separator:")
aList=myString.split("A")
print(aList)
print("original string after modification:")
print(myString)
```

输出:

```py
Original String:
I am A Python String
New List:
['I', 'am', 'A', 'Python', 'String']
when 'A' is declared as separator:
['I am ', ' Python String']
original string after modification:
I am A Python String
```

如果我们想把一个字符串拆分一定的次数，我们可以用`rsplit()`方法代替`split()`方法。`rsplit()`方法采用了一个名为`maxsplit` 的额外参数，它是字符串被拆分的次数。输入字符串在从字符串右侧开始的`maxsplit` 处被分割，由`rsplit()`方法返回一个包含输入字符串的`maxsplit+1`片段的列表。如果没有值传递给`maxsplit` 参数，`rsplit()`方法的工作方式与`split()` 方法相同。

示例:

```py
myString="I am A Python String"
print("Original String:")
print(myString)
newList=myString.rsplit()
print("New List without maxsplit:")
print(newList)
print("when maxsplit is set at 2:")
aList=myString.rsplit(maxsplit=2)
print(aList)
print("original string after modification:")
print(myString)
```

输出:

```py
Original String:
I am A Python String
New List without maxsplit:
['I', 'am', 'A', 'Python', 'String']
when maxsplit is set at 2:
['I am A', 'Python', 'String']
original string after modification:
I am A Python String
```

## Python 中如何串联字符串？

既然我们已经看到了如何拆分字符串，那么我们可能需要在 python 中执行[字符串连接](https://www.pythonforbeginners.com/concatenation/string-concatenation-and-formatting-in-python)。我们可以使用 `"+"`操作符和`join()`方法连接两个字符串。

在使用`"+"`操作符时，我们只是使用`"+"`操作符添加不同的字符串，并将其分配给一个新的字符串。在这里，我们可以在单个语句中使用`"+"`操作符连接任意数量的字符串。

示例:

```py
myString1="I am a "
print ("first string is:")
print(myString1)
myString2="Python String"
print("Second String is:")
print(myString2)
myString=myString1+myString2
print("Conactenated string is:")
print(myString)
```

输出:

```py
myString1="I am a "
print ("first string is:")
print(myString1)
myString2="Python String"
print("Second String is:")
print(myString2)
myString=myString1+myString2
print("Conactenated string is:")
print(myString)
```

我们还可以使用 python 中的 join 方法连接字符串。Join 方法在一个作为分隔符的字符串上被调用，一个列表或任何其他可迭代的字符串被传递给它进行连接。它返回一个包含 iterable 中单词的新字符串，由分隔符字符串分隔。

示例:

```py
myStringList=["I","am","a","python","string"]
print ("list of string is:")
print(myStringList)
separator=" "#space is used as separator
myString=separator.join(myStringList)
print("Concatenated string is:")
print(myString)
```

输出:

```py
list of string is:
['I', 'am', 'a', 'python', 'string']
Concatenated string is:
I am a python string
```

## Python 中如何修剪字符串？

字符串的开头或结尾可能包含额外的空格。我们可以使用 python 字符串方法删除这些空格，即`strip()`、`lstrip()`和`rstrip()`。

方法从输入字符串的开头删除空格，并返回一个新的字符串。

从字符串末尾删除空格并返回一个新的字符串。

方法从输入字符串的开头和结尾删除空格，并返回一个新的字符串。

示例:

```py
myString="          Python          " 
lstring=myString.lstrip()
rstring=myString.rstrip()
string =myString.strip()
print("Left Stripped string is:",end="")
print(lstring)
print("Right Stripped string is:",end="")
print(rstring)
print("Totally Stripped string is:",end="")
print(string)
```

输出:

```py
Left Stripped string is:Python          
Right Stripped string is:          Python
Totally Stripped string is:Python 
```

## Python 字符串方法在换行符处拆分字符串。

通过使用 python 中的`splitlines()` 方法，我们可以将一个字符串转换成一列句子。该函数在换行符或换行符处拆分输入字符串，并返回一个包含输入字符串所有片段的新列表。

示例:

```py
myString="Python is a great language.\n I love python" 
slist=myString.splitlines()
print("Original string is:")
print(myString)
print("Output is:")
print(slist)
```

输出:

```py
Original string is:
Python is a great language.
I love python
Output is:
['Python is a great language.', ' I love python']
```

## 结论

在本文中，我们看到了 python 字符串方法在 python 中操作字符串数据。我们已经看到了如何使用不同的方法来拆分、剥离和连接字符串。我们还看到了如何改变字符串中字母的大小写。请继续关注更多内容丰富的文章。