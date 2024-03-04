# 如何从 Python 字符串中删除标点符号

> 原文：<https://www.pythonforbeginners.com/python-strings/how-to-remove-punctuation-from-a-python-string>

在数据分析任务中，我们经常会遇到需要处理的文本数据，以便从数据中提取有用的信息。在文本处理过程中，我们可能必须从数据中提取或删除某些文本以使其有用，或者我们可能还需要用其他文本替换某些符号和术语以提取有用的信息。在这篇文章中，我们将学习标点符号，并看看从 python 字符串中删除标点符号的方法。

## 什么是标点符号？

英语语法中有几个符号，包括逗号、连字符、问号、破折号、感叹号、冒号、分号、圆括号、方括号等，这些都被称为标点符号。这些在英语中用于语法目的，但是当我们在 python 中执行文本处理时，我们通常不得不从字符串中省略标点符号。现在我们将看到在 Python 中从字符串中删除标点符号的不同方法。

## 使用 for 循环删除字符串中的标点符号

在这个方法中，首先我们将创建一个包含输出字符串的空 python 字符串。然后，我们将简单地遍历 python 字符串的每个字符，并检查它是否是标点符号。如果字符将是一个标点符号，我们将离开它。否则，我们将使用[字符串连接](https://www.pythonforbeginners.com/concatenation/string-concatenation-and-formatting-in-python)将其包含在输出字符串中。

例如，在下面给出的代码中，我们将每个标点符号保存在一个名为`punctuation`的字符串中。我们使用 for 循环遍历输入字符串`myString`，然后检查该字符是否出现在标点符号字符串中。如果不存在，该字符将包含在输出字符串`newString`中。

```py
 punctuation= '''!()-[]{};:'"\, <>./[[email protected]](/cdn-cgi/l/email-protection)#$%^&*_~'''
print("The punctuation marks are:")
print(punctuation)
myString= "Python.:F}or{Beg~inn;ers"
print("Input String is:")
print(myString)
newString=""
for x in myString:
    if x not in punctuation:
        newString=newString+x
print("Output String is:")
print(newString) 
```

输出

```py
The punctuation marks are:
!()-[]{};:'"\, <>./[[email protected]](/cdn-cgi/l/email-protection)#$%^&*_~
Input String is:
Python.:F}or{Beg~inn;ers
Output String is:
PythonForBeginners
```

## 使用正则表达式移除 python 字符串中的标点符号

我们还可以使用正则表达式在 python 中删除字符串中的标点符号。为此，我们将使用 python 中的`re`模块，它提供了使用正则表达式处理字符串的函数。

在此方法中，我们将使用`re.sub()`方法用空字符串替换每个非字母数字或空格字符，因此所有标点符号都将被删除。

`sub()` 方法的语法是`re.sub(pattern1, pattern2,input_string)`，其中`pattern1`表示将被替换的字符模式。在我们的例子中，我们将提供一个模式来表示不是字母数字或空格字符的字符。`pattern2`是`pattern1`中的字符将被替换的最终模式。在我们的例子中,`pattern2`将是空字符串，因为我们只需从 python 字符串中删除标点符号。`input_string`是必须被处理以去除标点符号的字符串。

示例:

```py
 import re
myString= "Python.:F}or{Beg~inn;ers"
print("Input String is:")
print(myString)
emptyString=""
newString=re.sub(r'[^\w\s]',emptyString,myString)
print("Output String is:")
print(newString)
```

输出

```py
Input String is:
Python.:F}or{Beg~inn;ers
Output String is:
PythonForBeginners
```

## 使用 replace()方法移除 python 字符串中的标点符号

在字符串上调用 Python string replace()方法时，该方法将初始模式和最终模式作为参数，并返回一个结果字符串，其中初始模式中的字符被最终模式中的字符替换。

我们可以使用 replace()方法，通过用空字符串替换每个标点符号来删除 python 字符串中的标点符号。我们将逐个迭代整个标点符号，用文本字符串中的空字符串替换它。

`replace()`方法的语法是`replace(character1,character2)`，其中`character1`是将被参数`character2`中给定字符替换的字符。在我们的例子中，`character1`将包含标点符号，而`character2`将是一个空字符串。

```py
 punctuation= '''!()-[]{};:'"\, <>./[[email protected]](/cdn-cgi/l/email-protection)#$%^&*_~'''
myString= "Python.:F}or{Beg~inn;ers"
print("Input String is:")
print(myString)
emptyString=""
for x in punctuation:
    myString=myString.replace(x,emptyString)
print("Output String is:")
print(myString)
```

输出:

```py
Input String is:
Python.:F}or{Beg~inn;ers
Output String is:
PythonForBeginners
```

## 使用 translate()方法移除 python 字符串中的标点符号

`translate()` 方法根据作为参数提供给函数的翻译表，用新字符替换输入字符串中指定的字符。翻译表应该包含哪些字符必须被哪些字符替换的映射。如果表中没有任何字符的映射，该字符将不会被替换。

`translate()`方法的语法是 translate( `translation_dictionary`)，其中`translation_dictionary`将是一个 python 字典，包含输入字符串中的字符到它们将被替换的字符的映射。

要创建翻译表，我们可以使用`maketrans()`方法。该方法将字符串中需要替换的起始字符、结束字符和需要删除的字符以字符串的形式作为可选输入，并返回一个作为翻译表的 [python 字典](https://www.pythonforbeginners.com/dictionary/how-to-use-dictionaries-in-python/)。

`maketrans()`方法的语法是`maketrans(pattern1,pattern2,optional_pattern)`。这里的`pattern1`将是一个包含所有要被替换的字符的字符串。`pattern2`将是一个字符串，其中包含了`pattern1`中的字符将被替换的字符。这里`pattern1`的长度应该等于`pattern2`的长度。`optional_pattern`是包含必须从输入文本中删除的字符的字符串。在我们的例子中，`pattern1`和`pattern2`将是空字符串，而`optional_pattern`将是包含标点符号的字符串。

为了创建一个从 python 字符串中删除标点符号的转换表，我们可以将`maketrans()` 函数的前两个参数留空，并将标点符号包含在要排除的字符列表中。这样，所有的标点符号将被删除，输出字符串将获得。

例子

```py
punctuation= '''!()-[]{};:'"\, <>./[[email protected]](/cdn-cgi/l/email-protection)#$%^&*_~'''
myString= "Python.:F}or{Beg~inn;ers"
print("Input String is:")
print(myString)
emptyString=""
translationTable= str.maketrans("","",punctuation)
newString=myString.translate(translationTable)
print("Output String is:")
print(newString)
```

输出

```py
Input String is:
Python.:F}or{Beg~inn;ers
Output String is:
PythonForBeginners
```

## 结论

在本文中，我们看到了如何使用 for 循环、正则表达式和内置的字符串方法(如 replace()和 translate())在 [python 中删除字符串中的标点符号。请继续关注更多内容丰富的文章。](https://www.pythonforbeginners.com/loops/for-while-and-nested-loops-in-python)