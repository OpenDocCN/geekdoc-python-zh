# 在 Python 中查找字符串中出现的所有子字符串

> 原文：<https://www.pythonforbeginners.com/basics/find-all-occurrences-of-a-substring-in-a-string-in-python>

子串是字符串中一个或多个字符的连续序列。在本文中，我们将讨论在 python 中查找一个字符串中所有子字符串的不同方法。

## 使用 Python 中的 For 循环查找字符串中所有出现的子字符串

在 for 循环的帮助下，我们可以遍历字符串中的字符。要使用 for 循环在 python 中查找一个字符串中所有出现的子字符串，我们将使用以下步骤。

*   首先，我们将找到输入字符串的长度，并将其存储在变量`str_len`中。
*   接下来，我们将找到子串的长度，并将其存储在变量`sub_len`中。
*   我们还将创建一个名为`sub_indices`的列表来存储子字符串出现的起始索引。
*   之后，我们将使用 for 循环遍历输入字符串。
*   在迭代过程中，我们将检查从当前索引开始的长度为`sub_len`的子串是否等于输入子串。
*   如果是，我们将使用`append()`方法将当前索引存储在`sub_indices`列表中。当在`sub_indices`上被调用时，`append()`方法将当前索引作为其输入参数，并将其附加到`sub_indices`。

在执行 for 循环之后，我们将获得输入子串在`sub_indices`列表中所有出现的起始索引。您可以在下面的示例中观察到这一点。

```py
myStr = "I am pythonforbeginners. I provide free python tutorials for you to learn python."
substring = "python"
str_len = len(myStr)
sub_len = len(substring)
sub_indices = []
for i in range(str_len - sub_len):
    if myStr[i:i + sub_len] == substring:
        sub_indices.append(i)
print("The string is:", myStr)
print("The substring is:", substring)
print("The starting indices of the occurrences of {} in the string are:{}".format(substring, sub_indices))
```

输出:

```py
The string is: I am pythonforbeginners. I provide free python tutorials for you to learn python.
The substring is: python
The starting indices of the occurrences of python in the string are:[5, 40, 74]
```

在这里，您可以观察到我们已经获得了子字符串 python 在整个字符串中的起始索引。

## 使用 Python 中的 While 循环查找字符串中所有出现的子字符串

使用 for 循环迭代字符串中的每个字符在时间上是很昂贵的。它还给出输出中重叠子字符串的索引。为了减少执行时间并在 python 中找到不重叠的子字符串，我们可以使用 while 循环。为此，我们将使用以下步骤。

*   首先，我们将找到输入字符串的长度，并将其存储在变量`str_len`中。
*   接下来，我们将找到子串的长度，并将其存储在变量`sub_len`中。
*   我们还将创建一个名为`sub_indices`的空列表来存储子字符串出现的起始索引和一个初始化为 0 的变量`temp`。
*   之后，我们将使用 while 循环遍历输入字符串。
*   在迭代过程中，我们将检查从索引`temp`开始的长度为`sub_len`的子串是否等于输入子串。
*   如果是，我们将使用`append()`方法将`temp`存储在`sub_indices`列表中。然后，我们将温度增加`sub_len`。
*   如果我们在索引`temp`处没有找到所需的子串，我们将把`temp`加 1。
*   在 while 循环执行之后，我们将获得输入子串在`sub_indices`列表中所有出现的起始索引。

您可以在下面的示例中观察到这一点。

```py
myStr = "I am pythonforbeginners. I provide free python tutorials for you to learn python."
substring = "python"
str_len = len(myStr)
sub_len = len(substring)
sub_indices = []
temp = 0
while temp <= str_len - sub_len:
    if myStr[temp:temp + sub_len] == substring:
        sub_indices.append(temp)
        temp = temp + sub_len
    else:
        temp = temp + 1
print("The string is:", myStr)
print("The substring is:", substring)
print("The starting indices of the occurrences of {} in the string are:{}".format(substring, sub_indices))
```

输出:

```py
The string is: I am pythonforbeginners. I provide free python tutorials for you to learn python.
The substring is: python
The starting indices of the occurrences of python in the string are:[5, 40, 74]
```

在这个例子中，我们将只获得输入字符串的所有非重叠子字符串的出现。

## 使用 Python 中的 Find()方法在字符串中查找子字符串的所有匹配项

在 python 中，`find()`方法用于查找字符串中任何子串的第一次出现。 `find()`方法的语法如下。

```py
myStr.find(sub_string, start, end)
```

这里，

*   `myStr`是输入字符串，我们必须在其中找到`sub_string`的位置。
*   `start`和`end`参数是可选的。它们接受字符串的起始索引和结束索引，我们必须在它们之间搜索`sub_string`。

当我们在一个字符串上调用`find()`方法时，它将一个子字符串作为它的输入参数。执行后，如果找到子串，它将返回子串的起始索引。否则，它返回-1。您可以在下面的示例中观察到这一点。

```py
myStr = "I am pythonforbeginners. I provide free python tutorials for you to learn python."
print(myStr.find("python"))
print(myStr.find("Aditya"))
```

输出:

```py
5
-1
```

在这里，可以看到`find()`方法返回子串`python`的起始索引。另一方面，它为子串`aditya`返回-1，因为它不在`myStr`中。

为了使用 python 中的`find()`方法在一个字符串中找到一个子字符串的所有出现，我们将使用以下步骤。

*   首先，我们将找到输入字符串的长度，并将其存储在变量`str_len`中。
*   接下来，我们将找到子串的长度，并将其存储在变量`sub_len`中。我们还将创建一个名为`sub_indices`的列表来存储子字符串出现的起始索引。
*   之后，我们将使用 for 循环遍历输入字符串。
*   在迭代过程中，我们将调用输入字符串上的 `find()` 方法，将子字符串作为第一个输入参数，将当前索引作为第二个输入参数，将当前索引+ `sub_len`作为第三个输入参数。基本上，我们正在检查从 index 到 index+ `sub_len`的当前子串是否是我们正在搜索的字符串。
*   如果`find()` 方法返回一个非-1 的值，我们将把它附加到`sub_indices`。这是因为如果在字符串中找到子字符串，find()方法会返回该子字符串的起始索引。然后，我们将转移到 for 循环的下一次执行。
*   如果`find()`方法返回-1，我们将转到 for 循环的下一次执行。

在执行 for 循环之后，我们将获得输入子串在`sub_indices`列表中所有出现的起始索引。您可以在下面的示例中观察到这一点。

```py
myStr = "I am pythonforbeginners. I provide free python tutorials for you to learn python."
substring = "python"
str_len = len(myStr)
sub_len = len(substring)
sub_indices = []
for temp in range(str_len-sub_len):
    index = myStr.find(substring, temp, temp + sub_len)
    if index != -1:
        sub_indices.append(index)
    else:
        continue
print("The string is:", myStr)
print("The substring is:", substring)
print("The starting indices of the occurrences of {} in the string are:{}".format(substring, sub_indices))
```

输出:

```py
The string is: I am pythonforbeginners. I provide free python tutorials for you to learn python.
The substring is: python
The starting indices of the occurrences of python in the string are:[5, 40, 74]
```

同样，上述方法给出了输出中重叠序列的索引。要在 python 中找到一个字符串的非重叠子字符串，我们可以使用 while 循环和如下所示的`find()` 方法。

```py
myStr = "I am pythonforbeginners. I provide free python tutorials for you to learn python."
substring = "python"
str_len = len(myStr)
sub_len = len(substring)
sub_indices = []
temp = 0
while temp <= str_len - sub_len:
    index = myStr.find(substring, temp, temp + sub_len)
    if index != -1:
        sub_indices.append(index)
        temp = temp + sub_len
    else:
        temp = temp + 1
print("The string is:", myStr)
print("The substring is:", substring)
print("The starting indices of the occurrences of {} in the string are:{}".format(substring, sub_indices))
```

输出:

```py
The string is: I am pythonforbeginners. I provide free python tutorials for you to learn python.
The substring is: python
The starting indices of the occurrences of python in the string are:[5, 40, 74]
```

## Python 中使用 startswith()方法的字符串中的子字符串

在 python 中,`startswith()`方法用于查找一个字符串是否以某个子字符串开头。`startswith()`方法的语法如下。

```py
myStr.startswith(sub_string, start, end)
```

这里，

*   `myStr`是我们必须检查它是否以`sub_string`开头的输入字符串。
*   `start`和`end`参数是可选的。它们接受字符串的起始索引和结束索引，在它们之间我们必须检查字符串是否在索引`start`处以`sub_string`开始。

当我们在一个字符串上调用`startswith()`方法时，它将一个子字符串作为它的输入参数。执行后，如果字符串以子字符串开头，则返回`True`。否则，它返回 False。您可以在下面的示例中观察到这一点。

```py
myStr = "I am pythonforbeginners. I provide free python tutorials for you to learn python."
print(myStr.startswith("python"))
print(myStr.startswith("I am"))
```

输出:

```py
False
True
```

在这里，您可以观察到子串`python`的`startswith()`方法`returns`为 False。另一方面，它为子串`I am`返回`True`。这是因为`myStr`从 `I am`开始，而不是从`python`开始。

为了使用`startswith()`方法在 python 中找到一个字符串中所有出现的子字符串，我们将使用以下步骤。

*   首先，我们将找到输入字符串的长度，并将其存储在变量`str_len`中。
*   接下来，我们将找到子串的长度，并将其存储在变量`sub_len`中。我们还将创建一个名为`sub_indices`的列表来存储子字符串出现的起始索引。
*   之后，我们将使用 for 循环遍历输入字符串。
*   在迭代过程中，我们将调用输入字符串上的`startswith()`方法，将子字符串作为第一个输入参数，将当前索引作为第二个输入参数。
*   如果`startswith()`方法返回`False`，这意味着子串没有从当前索引开始。因此，我们将转到 for 循环的下一次执行。
*   如果`startswith()`方法返回`True`，则意味着子串从当前索引开始。因此，我们将把当前索引附加到`sub_indices`上。之后，我们将进入 for 循环的下一次迭代。

在执行 for 循环之后，我们将获得输入子串在`sub_indices`列表中所有出现的起始索引。您可以在下面的示例中观察到这一点。

```py
myStr = "I am pythonforbeginners. I provide free python tutorials for you to learn python."
substring = "python"
str_len = len(myStr)
sub_len = len(substring)
sub_indices = []
for temp in range(str_len-sub_len):
    index = myStr.startswith(substring, temp)
    if index:
        sub_indices.append(temp)
    else:
        continue
print("The string is:", myStr)
print("The substring is:", substring)
print("The starting indices of the occurrences of {} in the string are:{}".format(substring, sub_indices))
```

输出:

```py
The string is: I am pythonforbeginners. I provide free python tutorials for you to learn python.
The substring is: python
The starting indices of the occurrences of python in the string are:[5, 40, 74]
```

使用 for 循环的方法给出了子串在字符串中重叠出现的索引。要在 python 中查找子串的非重叠索引的索引，可以使用如下所示的`startswith()`方法和 while 循环。

```py
myStr = "I am pythonforbeginners. I provide free python tutorials for you to learn python."
substring = "python"
str_len = len(myStr)
sub_len = len(substring)
sub_indices = []
temp = 0
while temp <= str_len - sub_len:
    index = myStr.startswith(substring, temp)
    if index:
        sub_indices.append(temp)
        temp = temp + sub_len
    else:
        temp = temp + 1
print("The string is:", myStr)
print("The substring is:", substring)
print("The starting indices of the occurrences of {} in the string are:{}".format(substring, sub_indices))
```

输出:

```py
The string is: I am pythonforbeginners. I provide free python tutorials for you to learn python.
The substring is: python
The starting indices of the occurrences of python in the string are:[5, 40, 74]
```

## 使用 Python 中的正则表达式查找字符串中子字符串的所有匹配项

正则表达式为我们提供了一种在 Python 中操作文本数据的最有效的方法。我们还可以使用`re`模块中提供的`finditer()` 方法在 python 中找到一个字符串中所有出现的子字符串。`finditer()`方法的语法如下。

```py
re.finditer(sub_string, input_string)
```

这里，`input_string`是一个字符串，我们必须在其中搜索`sub_string`的出现。

`finditer()`方法将子字符串作为第一个输入参数，将原始字符串作为第二个参数。执行后，它返回一个迭代器，其中包含子字符串的匹配对象。匹配对象包含有关子字符串的起始和结束索引的信息。我们可以通过对匹配对象调用`start()` 方法和 `end()`方法来获得匹配对象的开始和结束索引。

为了使用`finditer()`方法在 python 中找到一个字符串中所有出现的子字符串，我们将使用以下步骤。

*   首先，我们将创建一个名为`sub_indices`的列表来存储子字符串出现的起始索引。
*   之后，我们将获得包含子串匹配对象的迭代器。
*   一旦我们得到迭代器，我们将使用 for 循环遍历匹配对象。
*   迭代时，我们将调用当前匹配对象上的`start()`方法。它将返回原始字符串中子串的起始索引。我们将把索引附加到`sub_indices`上。

在 for 循环执行之后，我们将获得输入字符串中给定子字符串的所有匹配项的起始索引。您可以在以下示例中观察到这一点。

```py
import re

myStr = "I am pythonforbeginners. I provide free python tutorials for you to learn python."
substring = "python"
sub_indices = []
match_objects = re.finditer(substring, myStr)
for temp in match_objects:
    index = temp.start()
    sub_indices.append(index)
print("The string is:", myStr)
print("The substring is:", substring)
print("The starting indices of the occurrences of {} in the string are:{}".format(substring, sub_indices))
```

输出:

```py
The string is: I am pythonforbeginners. I provide free python tutorials for you to learn python.
The substring is: python
The starting indices of the occurrences of python in the string are:[5, 40, 74]
```

除了使用 for 循环，您还可以使用 [list comprehension](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python) 在 python 中查找一个字符串中所有出现的子字符串，如下所示。

```py
import re

myStr = "I am pythonforbeginners. I provide free python tutorials for you to learn python."
substring = "python"
match_objects = re.finditer(substring, myStr)
sub_indices = [temp.start() for temp in match_objects]
print("The string is:", myStr)
print("The substring is:", substring)
print("The starting indices of the occurrences of {} in the string are:{}".format(substring, sub_indices))
```

输出:

```py
The string is: I am pythonforbeginners. I provide free python tutorials for you to learn python.
The substring is: python
The starting indices of the occurrences of python in the string are:[5, 40, 74]
```

在上面的例子中，我们首先使用`finditer()`方法获得了匹配对象。之后，我们使用列表理解和`start()`方法来寻找`myStr`中子串的起始索引。

## 使用 Python 中的使用 flashtext 模块在字符串中出现的所有子字符串

在 Python 中，您可以使用`flashtext`模块来查找一个字符串中所有出现的子字符串，而不是使用上面讨论的所有方法。您可以使用以下语句通过 PIP 安装`flashtext`模块。

```py
pip3 install flashtext
```

要使用 flashtext 模块在一个字符串中查找一个子字符串的所有出现，我们将使用以下步骤。

*   首先，我们将使用`KeywordProcessor()`函数创建一个关键字处理器对象。
*   创建关键字处理器后，我们将使用`add_keyword()`方法将子串添加到关键字处理器对象中。当在关键字处理器对象上调用`add_keyword()`方法时，它会将子字符串作为输入参数。
*   然后，我们将调用关键字处理器对象上的`extract_keywords()`方法。它返回一个元组列表。每个元组包含子串作为其第一个元素，子串的开始索引作为其第二个元素，结束索引作为其第三个元素。
*   最后，我们将创建一个名为`sub_indices`的空列表，并使用 for 循环从元组列表中提取子串的起始索引。

在执行 for 循环之后，我们将在`sub_indices`中获得输入字符串中给定子串的所有出现的起始索引。您可以在下面的示例中观察到这一点。

```py
from flashtext import KeywordProcessor

myStr = "I am pythonforbeginners. I provide free python tutorials for you to learn python."
substring = "python"
sub_indices = []
kwp = KeywordProcessor()
kwp.add_keyword(substring)
result_list = kwp.extract_keywords(myStr,span_info=True)
for tuples in result_list:
    sub_indices.append(tuples[1])
print("The string is:", myStr)
print("The substring is:", substring)
print("The starting indices of the occurrences of {} in the string are:{}".format(substring, sub_indices))
```

输出:

```py
The string is: I am pythonforbeginners. I provide free python tutorials for you to learn python.
The substring is: python
The starting indices of the occurrences of python in the string are:[40, 74]
```

除了在最后一步中使用 for 循环，还可以使用 list comprehension 在 python 中查找某个字符串中某个子字符串的所有匹配项，如下所示。

```py
from flashtext import KeywordProcessor

myStr = "I am pythonforbeginners. I provide free python tutorials for you to learn python."
substring = "python"
kwp = KeywordProcessor()
kwp.add_keyword(substring)
result_list = kwp.extract_keywords(myStr, span_info=True)
sub_indices = [tuples[1] for tuples in result_list]
print("The string is:", myStr)
print("The substring is:", substring)
print("The starting indices of the occurrences of {} in the string are:{}".format(substring, sub_indices))
```

输出:

```py
The string is: I am pythonforbeginners. I provide free python tutorials for you to learn python.
The substring is: python
The starting indices of the occurrences of python in the string are:[40, 74]
```

在这种方法中，您可以观察到关键字流程只提取子字符串的两个实例。这是由于关键字处理器搜索整个单词的原因。如果子字符串不是整个单词，它就不会包含在结果中。

## 结论

在本文中，我们讨论了在 Python 中查找一个字符串中所有子字符串的不同方法。在所有这些方法中，我建议您使用带有列表理解的正则表达式。它会在最快的时间内给你结果，因为这是最有效的方法。

要了解更多关于 python 编程的知识，您可以阅读这篇关于如何在 Python 中[删除列表中所有出现的字符的文章。您可能也喜欢这篇关于如何](https://www.pythonforbeginners.com/basics/remove-all-occurrences-of-a-character-in-a-list-or-string-in-python)[检查 python 字符串是否包含数字](https://www.pythonforbeginners.com/strings/check-if-a-python-string-contains-a-number)的文章。

请继续关注更多内容丰富的文章。

快乐学习！