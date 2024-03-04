# 在 Python 中删除字符串中的逗号

> 原文：<https://www.pythonforbeginners.com/strings/remove-commas-from-string-in-python>

在 python 中，我们使用字符串来分析文本数据。在分析之前，我们需要对文本数据进行预处理。有时，我们可能需要从文本中删除像逗号或撇号这样的字符。在本文中，我们将讨论如何在 python 中删除给定字符串中的逗号。

## 使用 for 循环删除字符串中的逗号

从字符串中删除逗号最简单的方法是创建一个新的字符串，留下逗号。

在这种方法中，我们将首先创建一个空字符串`newString`。之后，我们将使用 for 循环逐个字符地遍历输入字符串。在遍历时，我们将检查当前字符是否是逗号。如果字符不是逗号，我们将使用[字符串连接](https://www.pythonforbeginners.com/concatenation/string-concatenation-and-formatting-in-python)操作将当前字符添加到`newString`中。一旦 for 循环的执行完成，我们将得到变量`newString`中没有逗号的输出字符串。

您可以在下面的示例中观察整个过程。

```py
myString = "This, is, a, string, that, has, commas, in, it."
print("Original String:", myString)
newString = ""
for character in myString:
    if character != ",":
        newString = newString + character
print("Output String:", newString)
```

输出:

```py
Original String: This, is, a, string, that, has, commas, in, it.
Output String: This is a string that has commas in it.
```

## 使用 replace()方法

我们可以使用`replace()`方法从任何给定的字符串中删除逗号，而不是遍历整个字符串来删除字符串中的逗号。

replace 方法的语法如下。

`str.replace(old_character, new_character, count)`

这里，

*   参数`old_character`用于将需要删除的字符传递给 replace 方法。我们将删除逗号“，”作为第一个输入参数。
*   参数`new_character`用于传递将替换`old_character`的字符。我们将传递一个空字符串作为`new_character`，以便删除逗号字符。
*   参数 count 用于决定字符串中有多少个`old_character`实例将被`new_character`替换。这是一个可选参数，我们可以让它为空。在这种情况下，字符串中所有的`old_character`实例都将被替换为`new_character`。
*   在用`new_character`替换了`old_character`的所有实例之后，`replace()`方法返回修改后的字符串。

我们可以使用`str.replace()`方法从给定的字符串中删除逗号，如下所示。

```py
myString = "This, is, a, string, that, has, commas, in, it."
print("Original String:", myString)
newString = myString.replace(",", "")
print("Output String:", newString)
```

输出:

```py
Original String: This, is, a, string, that, has, commas, in, it.
Output String: This is a string that has commas in it.
```

## 使用 re.sub()方法删除字符串中的逗号

正则表达式是操作字符串的一个很好的工具。在 python 中，我们也可以用它们来删除字符串中的逗号。为此，我们将使用`re.sub()`方法。

`re.sub()`方法的语法如下。

`re.sub(old_character, new_character, input_string)`

这里，

*   参数`old_character`用于将需要移除的字符传递给`re.sub()`方法。我们将删除逗号“，”作为第一个输入参数。
*   参数`new_character`用于传递将替换`old_character`的字符。我们将传递一个空字符串作为`new_character`，以便删除逗号字符。

*   `input_string`参数用于将需要修改的字符串传递给`re.sub()`方法。
*   执行后，该方法返回修改后的字符串。

我们可以使用`re.sub()`方法从给定的字符串中删除逗号，如下所示。

```py
import re

myString = "This, is, a, string, that, has, commas, in, it."
print("Original String:", myString)
newString = re.sub(",", "", myString)
print("Output String:", newString)
```

输出:

```py
Original String: This, is, a, string, that, has, commas, in, it.
Output String: This is a string that has commas in it.
```

## 结论

在本文中，我们讨论了在 python 中删除字符串中逗号的三种方法。要了解更多关于 python 中的字符串，可以阅读这篇关于 python 中的[正则表达式的文章。你可能也会喜欢这篇关于用 python](https://www.pythonforbeginners.com/regex/regular-expressions-in-python) 理解[列表的文章。](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python)