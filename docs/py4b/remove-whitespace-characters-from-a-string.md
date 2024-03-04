# 从字符串中删除空白字符

> 原文：<https://www.pythonforbeginners.com/basics/remove-whitespace-characters-from-a-string>

python 中的字符串广泛用于处理文本数据。在这篇文章中，我们将研究从字符串中删除空白字符的不同方法。我们还将实现这些示例，以便更好地理解这个概念。

## 什么是空白字符？

空白字符是诸如空格、制表符和换行符之类的字符。在 python 中，定义了一个包含所有空白字符的字符串常量 **string.whitespace** 。这些字符是由“”表示的空格、由“\t”表示的制表符、由“\n”表示的换行符、由“\r”表示的回车符、由“\v”表示的垂直制表符和由“\f”表示的换页符。

现在我们来看看从字符串中删除这些空白字符的不同方法。

## 使用 for 循环删除空白字符

从字符串中删除空白字符的最简单的方法是使用 for 循环。在这个方法中，我们将首先创建一个新的空字符串。之后，对于输入字符串中的每个字符，我们将检查它是否是空白字符。如果是，我们将丢弃它。否则，我们将使用如下的[字符串连接](https://www.pythonforbeginners.com/concatenation/string-concatenation-and-formatting-in-python)操作将字符添加到新创建的字符串中。

```py
import string

input_string = """This is PythonForBeginners.com
Here, you   can read python \v tutorials for free."""

new_string = ""
for character in input_string:
    if character not in string.whitespace:
        new_string = new_string + character

print("THe original string is:")
print(input_string)
print("Output String is:")
print(new_string) 
```

输出:

```py
THe original string is:
This is PythonForBeginners.com
Here, you   can read python  tutorials for free.
Output String is:
ThisisPythonForBeginners.comHere,youcanreadpythontutorialsforfree.
```

## 使用 split()方法删除空白字符

我们可以使用 split()和 join()方法来删除空格，而不是遍历输入字符串的每个字符。

在字符串上调用 split()方法时，该方法在空格处拆分字符串，并返回子字符串列表。我们可以使用 join()方法连接所有的子字符串。

join()方法在分隔符字符串上调用时，接受包含字符串的列表、元组或其他可迭代对象，并用分隔符字符串连接可迭代对象中的所有字符串。这里，我们将使用空字符串作为分隔符字符串。这样，我们可以将 split()方法创建的所有子字符串连接起来，以创建输出字符串。

```py
import string

input_string = """This is PythonForBeginners.com
Here, you   can read python \v tutorials for free."""
str_list = input_string.split()
new_string = "".join(str_list)
print("The original string is:")
print(input_string)
print("Output String is:")
print(new_string) 
```

输出:

```py
The original string is:
This is PythonForBeginners.com
Here, you   can read python  tutorials for free.
Output String is:
ThisisPythonForBeginners.comHere,youcanreadpythontutorialsforfree.
```

## 使用正则表达式删除空白字符

在 python 中，我们可以使用 regex 模块中的 sub()方法用一个模式替换另一个模式。sub()方法接受三个输入参数。第一个输入是需要替换的模式。第二个输入参数是必须放在字符串中的新模式。第三个输入参数是输入字符串。它返回一个带有修改值的新字符串。

这里，我们将用原始字符串中的空字符串替换中的空格字符。匹配所有空格字符的正则表达式是“\s+”。新的模式将只是一个由""表示的空字符串。使用这些模式和 sub()方法，我们可以从输入字符串中删除空白字符，如下所示。

```py
import re
import string

input_string = """This is PythonForBeginners.com
Here, you   can read python \v tutorials for free."""

new_string = re.sub(r"\s+", "", input_string)
print("The original string is:")
print(input_string)
print("Output String is:")
print(new_string) 
```

输出:

```py
The original string is:
This is PythonForBeginners.com
Here, you   can read python  tutorials for free.
Output String is:
ThisisPythonForBeginners.comHere,youcanreadpythontutorialsforfree.
```

## 使用翻译表删除空白字符

除了正则表达式，我们还可以使用翻译表和 translate()方法来删除字符串中的所有空白字符。

翻译表就是包含旧字符到新字符映射的字典。在我们的程序中，我们将把每个空格字符映射到一个空字符串。这里，我们将使用空格字符的 ASCII 值作为键，空字符串作为它们的关联值。使用 order()函数可以找到每个字符的 ASCII 值。可以如下创建转换表。

```py
import string

translation_table = {ord(x): "" for x in string.whitespace}
print("The translation table is:")
print(translation_table) 
```

输出:

```py
The translation table is:
{32: '', 9: '', 10: '', 13: '', 11: '', 12: ''}
```

创建翻译表后，我们可以使用 translate()方法从输入字符串中删除空白字符。在字符串上调用 translate()方法时，该方法将翻译表作为输入，并使用翻译表替换其中的字符。

我们将使用上面创建的转换表用空字符串替换空白字符，如下所示。

```py
 import string

input_string = """This is PythonForBeginners.com
Here, you   can read python \v tutorials for free."""
translation_table = {ord(x): "" for x in string.whitespace}

new_string = input_string.translate(translation_table)
print("The original string is:")
print(input_string)
print("Output String is:")
print(new_string)
```

输出:

```py
The original string is:
This is PythonForBeginners.com
Here, you   can read python  tutorials for free.
Output String is:
ThisisPythonForBeginners.comHere,youcanreadpythontutorialsforfree.
```

## 结论

在本文中，我们讨论了从字符串中删除空白字符的不同方法。要了解更多关于 python 编程的知识，你可以阅读这篇关于[列表理解](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python)的文章。你可能也会喜欢这篇关于 Python 中[链表的文章。](https://www.pythonforbeginners.com/lists/linked-list-in-python)