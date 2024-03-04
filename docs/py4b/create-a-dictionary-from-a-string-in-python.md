# 用 Python 从字符串创建字典

> 原文：<https://www.pythonforbeginners.com/basics/create-a-dictionary-from-a-string-in-python>

字符串和字典是 Python 中最常用的两种数据结构。在 Python 中，我们使用字符串进行[文本分析。另一方面，字典用于存储键值对。在本文中，我们将讨论如何用 Python 从字符串创建字典。](https://www.pythonforbeginners.com/basics/text-analysis-in-python)

## 使用 for 循环从字符串创建字典

为了用 Python 从一个字符串创建一个字典，我们可以使用一个 [python for 循环](https://www.pythonforbeginners.com/loops/for-while-and-nested-loops-in-python)。在这种方法中，我们将创建一个字典，其中包含原始字符串中所有字符的计数。为了从给定的字符串创建字典，我们将使用以下步骤。

*   首先，我们将使用`dict()`函数创建一个空的 [python 字典](https://www.pythonforbeginners.com/dictionary/how-to-use-dictionaries-in-python/)。
*   接下来，我们将使用 for 循环遍历字符串中的字符。
*   在迭代时，我们将首先检查该字符是否已经存在于字典中。如果是，我们将增加与字典中的字符相关的值。否则，我们将把这个字符作为一个键，1 作为关联值赋给字典。

在执行 for 循环之后，我们将得到字典，其中包含作为键的字符串的字符和作为值的它们的频率。您可以在下面的示例中观察到这一点。

```py
myStr="Python For Beginners"
print("The input string is:",myStr)
myDict=dict()
for character in myStr:
    if character in myDict:
        myDict[character]+=1
    else:
        myDict[character]=1
print("The dictionary created from characters of the string is:")
print(myDict)
```

输出:

```py
The input string is: Python For Beginners
The dictionary created from characters of the string is:
{'P': 1, 'y': 1, 't': 1, 'h': 1, 'o': 2, 'n': 3, ' ': 2, 'F': 1, 'r': 2, 'B': 1, 'e': 2, 'g': 1, 'i': 1, 's': 1}
```

## 使用计数器方法从字符串中查找字典

为了从 python 中的字符串创建字典，我们还可以使用集合模块中定义的`Counter()`方法。`Counter()`方法将一个字符串作为输入参数，并返回一个计数器对象。计数器对象包含作为键的字符串的所有字符，以及作为相关值的它们的频率。您可以在下面的示例中观察到这一点。

```py
from collections import Counter
myStr="Python For Beginners"
print("The input string is:",myStr)
myDict=dict(Counter(myStr))
print("The dictionary created from characters of the string is:")
print(myDict)
```

输出:

```py
The input string is: Python For Beginners
The dictionary created from characters of the string is:
{'P': 1, 'y': 1, 't': 1, 'h': 1, 'o': 2, 'n': 3, ' ': 2, 'F': 1, 'r': 2, 'B': 1, 'e': 2, 'g': 1, 'i': 1, 's': 1}
```

## 使用 Dict.fromkeys()方法从字符串中查找字典

我们还可以使用`fromkeys()`方法从 python 中的一个字符串创建一个字典。`fromkeys()` 方法将一个字符串作为第一个输入参数，将一个默认值作为第二个参数。执行后，它返回一个字典，其中包含作为键的字符串的所有字符，以及作为每个键的关联值的输入值。如果您想为字典的所有键提供一个默认的关联值，您可以使用`fromkeys()`方法从一个字符串创建一个字典，如下所示。

```py
from collections import Counter
myStr="Python For Beginners"
print("The input string is:",myStr)
myDict=dict.fromkeys(myStr,0)
print("The dictionary created from characters of the string is:")
print(myDict)
```

输出:

```py
The input string is: Python For Beginners
The dictionary created from characters of the string is:
{'P': 0, 'y': 0, 't': 0, 'h': 0, 'o': 0, 'n': 0, ' ': 0, 'F': 0, 'r': 0, 'B': 0, 'e': 0, 'g': 0, 'i': 0, 's': 0}
```

如果不想向`fromkeys()`方法传递任何默认值，可以选择省略。在这种情况下，从字符串创建的字典的所有键都没有关联值。您可以在下面的示例中观察到这一点。

```py
from collections import Counter
myStr="Python For Beginners"
print("The input string is:",myStr)
myDict=dict.fromkeys(myStr)
print("The dictionary created from characters of the string is:")
print(myDict)
```

输出:

```py
The input string is: Python For Beginners
The dictionary created from characters of the string is:
{'P': None, 'y': None, 't': None, 'h': None, 'o': None, 'n': None, ' ': None, 'F': None, 'r': None, 'B': None, 'e': None, 'g': None, 'i': None, 's': None}
```

## 使用 Ordereddict.fromkeys()方法

除了字典，您还可以从 Python 中的字符串获得有序字典。如果您想为字典的所有键提供一个默认的关联值，您可以使用`fromkeys()`方法从一个字符串创建一个有序字典。`fromkeys()`方法将一个字符串作为第一个输入参数，将一个默认值作为第二个参数。执行后，它返回一个有序字典，其中包含作为键的字符串的所有字符。您可以在下面的示例中观察到这一点。

```py
from collections import OrderedDict
myStr="Python For Beginners"
print("The input string is:",myStr)
myDict=OrderedDict.fromkeys(myStr,0)
print("The dictionary created from characters of the string is:")
print(myDict)
```

输出:

```py
The input string is: Python For Beginners
The dictionary created from characters of the string is:
OrderedDict([('P', 0), ('y', 0), ('t', 0), ('h', 0), ('o', 0), ('n', 0), (' ', 0), ('F', 0), ('r', 0), ('B', 0), ('e', 0), ('g', 0), ('i', 0), ('s', 0)])
```

如果不想向 `fromkeys()`方法传递任何默认值，可以选择省略。在这种情况下，从字符串创建的有序字典的所有键都没有关联值。您可以在下面的示例中观察到这一点。

```py
from collections import OrderedDict
myStr="Python For Beginners"
print("The input string is:",myStr)
myDict=OrderedDict.fromkeys(myStr)
print("The dictionary created from characters of the string is:")
print(myDict)
```

输出:

```py
The input string is: Python For Beginners
The dictionary created from characters of the string is:
OrderedDict([('P', None), ('y', None), ('t', None), ('h', None), ('o', None), ('n', None), (' ', None), ('F', None), ('r', None), ('B', None), ('e', None), ('g', None), ('i', None), ('s', None)]) 
```

## 结论

在本文中，我们讨论了用 Python 从字符串创建字典的不同方法。要了解更多关于 python 编程的知识，你可以阅读这篇关于[字符串操作](https://www.pythonforbeginners.com/basics/string-manipulation-in-python)的文章。您可能也会喜欢这篇关于 [python simplehttpserver](https://www.pythonforbeginners.com/modules-in-python/how-to-use-simplehttpserver) 的文章。

我希望你喜欢阅读这篇文章。请继续关注更多内容丰富的文章。

快乐学习！