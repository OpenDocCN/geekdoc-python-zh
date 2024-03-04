# 在 Python 中检查字符串是否为空或空白

> 原文：<https://www.pythonforbeginners.com/basics/check-if-a-string-is-empty-or-whitespace-in-python>

Python 中的字符串用于处理文本数据。在对文本数据进行操作时，我们可能需要删除空字符串或空白。当我们打印空字符串或空白时，我们无法区分这两者。在本文中，我们将讨论 Python 中检查字符串是否为空或空白的不同方法。这将帮助您区分空字符串和空白。

## 使用等式运算符检查 Python 中的字符串是否为空或空白

要使用等号运算符检查一个字符串是否为空，我们只需将该字符串与另一个空字符串进行比较。如果结果为真，则输入字符串为空。否则不会。

```py
input_string=""
print("The input string is:",input_string)
if input_string=="":
    print("Input string is an empty string.")
```

输出:

```py
The input string is: 
Input string is an empty string.
```

要检查字符串是否只包含空白字符，可以将其与空字符串进行比较。如果结果为 False，字符串将包含空白字符。您可以在下面的示例中观察到这一点。

```py
input_string=" "
print("The input string is:",input_string)
if input_string=="":
    print("Input string is an empty string.")
else:
    print("Input string is a whitespace character.")
```

输出:

```py
The input string is:  
Input string is a whitespace character.
```

这里，包含空格以外的字符的输入字符串也将返回 False，如下所示。

```py
input_string="Aditya"
print("The input string is:",input_string)
if input_string=="":
    print("Input string is an empty string.")
else:
    print("Input string is a whitespace character.")
```

输出:

```py
The input string is: Aditya
Input string is a whitespace character.
```

在上面的例子中，你可以看到我们使用了字符串`“Aditya”`。然而，程序告诉我们字符串只包含空格。因此，该程序在逻辑上是不正确的。因此，只有当您确定输入字符串要么是空字符串，要么只包含空白字符时，才可以使用这种方法。

## 在 Python 中使用 len()函数检查字符串是否为空或空白

`len()`函数用于查找列表、字符串、元组等可迭代对象的长度。它将 iterable 对象作为其输入参数，并返回 iterable 对象的长度。

要使用 Python 中的 `len()` 函数检查字符串是空的还是空白，我们将使用以下步骤。

*   首先，我们将使用`len()` 函数找到输入字符串的长度。我们将把长度存储在变量`str_len`中。
*   现在，我们将检查输入字符串的长度是否为 0。
*   如果输入字符串的长度为 0，我们就说该字符串是空字符串。否则，我们会说字符串包含空白。

您可以在下面的示例中观察到这一点。

```py
input_string=""
print("The input string is:",input_string)
str_len=len(input_string)
if str_len==0:
    print("Input string is an empty string.")
else:
    print("Input string is a whitespace character.")
```

输出:

```py
The input string is: 
Input string is an empty string.
```

同样，如果输入字符串包含空格以外的字符，程序将给出错误的结果，如下所示。

```py
input_string = "Aditya"
print("The input string is:", input_string)
str_len = len(input_string)
if str_len == 0:
    print("Input string is an empty string.")
else:
    print("Input string is a whitespace character.")
```

输出:

```py
The input string is: Aditya
Input string is a whitespace character.
```

同样，你可以看到我们使用了字符串 `“Aditya”`。然而，程序告诉我们字符串只包含空格。因此，该程序在逻辑上是不正确的。因此，只有当您确定输入字符串要么是空字符串，要么只包含空白字符时，才可以使用这种方法。

## 在 Python 中使用 not 运算符查找字符串是否为空或空白

在 Python 中，我们还可以使用 not 操作符来检查字符串是空的还是空白。

在 Python 中，所有的可迭代对象，如字符串、列表和元组，当它们为空时，计算结果为 False。因此，空字符串的计算结果为 False。

为了使用 not 操作符检查一个字符串是空的还是空白的，我们将在输入字符串上使用 not 操作符。如果字符串为空，字符串将计算为 False。然后，not 运算符将结果转换为 True。

因此，如果输出为真，我们会说字符串是空的。否则，我们会说字符串包含空白。您可以在下面的示例中观察到这一点。

```py
input_string = ""
print("The input string is:", input_string)
if not input_string:
    print("Input string is an empty string.")
else:
    print("Input string is a whitespace character.")
```

输出:

```py
The input string is: 
Input string is an empty string.
```

同样，如果输入字符串包含空格以外的字符，程序会说该字符串只包含空格。因此，只有当您确定输入字符串要么是空字符串，要么只包含空白字符时，才可以使用这种方法。

## 在 Python 中使用 For 循环检查字符串是否为空或空白

python 中有六个空白字符，分别是空格`“ ”,`制表符 `“\t”`，换行符`“\n”`，竖线制表符`”\v”`，回车符`“\r”`，以及 `“\f”`回车符。我们可以使用这些空白字符的列表和 for 循环来检查一个字符串是空的还是 python 中的空白。为此，我们将使用以下步骤。

*   首先，我们将定义一个名为`whitespaces`的列表来存储空白字符。
*   然后，我们将定义一个变量`isWhiteSpace`，并将其初始化为 True。
*   现在，我们将使用等式运算符检查输入字符串是否为空字符串。
*   如果输入字符串是一个空字符串，我们将这样打印。
*   如果输入字符串不为空，我们将使用 for 循环迭代输入字符串的字符。
*   在 for 循环中，我们将使用成员操作符和`whitespaces`列表检查当前字符是否是空白字符。
*   如果当前字符不是空白字符，我们将赋值 False 给变量`isWhiteSpace`。然后，我们将使用[中断语句](https://www.pythonforbeginners.com/basics/break-and-continue-statements)来中断 for 循环。
*   在 for 循环之外，如果`isWhiteSpace`为真，我们将打印出该字符串只包含空白字符。
*   否则，我们将打印出字符串包含空格以外的字符。

您可以在下面的示例中观察整个过程。

```py
input_string = "    "
whitespaces = [" ", "\t", "\n", "\v", "\r", "\f"]
print("The input string is:", input_string)
isWhiteSpace = True

if input_string == "":
    print("Input string is an empty string.")
else:
    for character in input_string:
        if character in whitespaces:
            continue
        else:
            isWhiteSpace = False
            break
if isWhiteSpace:
    print("The string contains only whitespace characters.")
else:
    print("The string contains characters other than whitespaces.")
```

输出:

```py
The input string is: 
The string contains only whitespace characters.
```

## 在 Python 中使用列表理解来发现一个字符串是空的还是空白

[列表理解](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python)用于从现有的可迭代对象创建一个列表。然而，我们可以使用`all()` 函数和 list comprehension 来检查 Python 中的输入字符串是否为空或空白。

`all()` 函数将一个 iterable 对象作为它的输入参数。如果 iterable 对象的所有元素都计算为 True，则返回 True。否则，它返回 False。

为了使用 Python 中的列表理解来检查字符串是空的还是空白的，我们将使用以下步骤。

*   首先，我们将定义一个名为`whitespaces`的列表来存储空白字符。
*   然后，我们将使用等式运算符检查输入字符串是否为空字符串。
*   如果输入字符串是一个空字符串，我们将这样打印。
*   如果输入字符串不为空，我们将使用列表理解创建一个布尔值列表。
*   在列表理解中，如果输入字符串中的字符是空白字符，我们将在输出列表中包含 True。否则，我们将在输出列表中包含 False。
*   创建布尔值列表后，我们将把它传递给`all()` 函数。如果 `all()`函数返回 True，这意味着字符串只包含空白字符。因此，我们将打印相同的内容。
*   如果 all()函数返回 False，我们将打印出输入字符串包含空白字符以外的字符。

您可以在下面的示例中观察整个过程。

```py
input_string = "    "
whitespaces = [" ", "\t", "\n", "\v", "\r", "\f"]
print("The input string is:", input_string)
if input_string == "":
    print("Input string is an empty string.")
else:
    myList = [True for character in input_string if character in whitespaces]
    output = all(myList)
    if output:
        print("The string contains only whitespace characters.")
    else:
        print("The string contains characters other than whitespaces.")
```

输出:

```py
The input string is: 
The string contains only whitespace characters.
```

## 使用 strip()方法检查 Python 中的字符串是否为空或空白

`strip()`方法用于删除字符串中的前导或尾随空格。当`invoked()` 在字符串上时，从字符串中删除前导和尾随空格。执行后，它返回修改后的字符串。

为了检查 Python 中的字符串是空的还是空白，我们将使用以下步骤。

*   首先，我们将使用等式运算符检查字符串是否为空。
*   如果字符串为空，我们将这样打印。否则，我们将在字符串上调用`strip()`方法。
*   如果`strip()` 方法返回一个空字符串，我们可以说原始字符串只包含空白字符。因此，我们将这样打印。
*   如果`strip()`方法返回非空字符串，则输入字符串包含除空白字符以外的字符。因此，我们将打印该字符串包含除空白字符以外的字符。

您可以在下面的示例中观察整个过程。

```py
input_string = "   . "
print("The input string is:", input_string)
if input_string == "":
    print("Input string is an empty string.")
else:
    newStr = input_string.strip()
    if newStr == "":
        print("The string contains only whitespace characters.")
    else:
        print("The string contains characters other than whitespaces.")
```

输出:

```py
The input string is:    . 
The string contains characters other than whitespaces.
```

这种方法甚至适用于包含空格以外的字符的字符串。因此，你可以在任何情况下使用这种方法。

## 使用 isspace()方法检查 Python 中的字符串是否为空或空白

方法用来检查一个字符串是否只包含空白字符。在字符串上调用时，如果字符串只包含空白字符，则`isspace()`方法返回 True。否则，它返回 False。

为了使用`isspace()`方法在 Python 中检查一个字符串是空的还是空白，我们将使用以下步骤。

*   首先，我们将使用等式运算符检查字符串是否为空。
*   如果字符串为空，我们将这样打印。否则，我们将在字符串上调用`isspace()` 方法。
*   如果`isspace()`方法返回 True，这意味着输入字符串只包含空白字符。
*   如果`isspace()`方法返回 False，则输入字符串包含空白字符以外的字符。因此，我们将打印该字符串包含除空白字符以外的字符。

您可以在下面的示例中观察到这一点。

```py
input_string = "  A "
print("The input string is:", input_string)
if input_string == "":
    print("Input string is an empty string.")
elif input_string.isspace():
    print("The string contains only whitespace characters.")
else:
    print("The string contains characters other than whitespaces.")
```

输出:

```py
The input string is:   A 
The string contains characters other than whitespaces.
```

同样，这种方法甚至适用于包含非空白字符的字符串。因此，你可以在任何情况下使用这种方法。

## 使用正则表达式检查 Python 中的字符串是否为空或空白

[正则表达式](https://www.pythonforbeginners.com/regex/regular-expressions-in-python)用于在 python 中高效地操作字符串。我们还可以使用正则表达式来检查给定的字符串是空的还是空白。为此，我们将使用`search()`函数。

`search()`函数将一个字符串模式作为它的第一个输入参数，将一个字符串作为它的第二个输入参数。执行后，它返回一个匹配对象。如果作为第二个输入参数给定的输入字符串中的子字符串与作为第一个输入参数给定的模式相匹配，那么 match 对象不是 None。如果字符串中不存在该模式，匹配对象将为 None。

要使用`search()`函数检查给定的字符串是空的还是空白字符，我们将使用以下步骤。

*   首先，我们将使用等式运算符检查字符串是否为空。
*   如果字符串为空，我们将这样打印。否则，我们将使用`search()`函数来检查字符串是否只包含空白字符。
*   因为我们需要检查字符串是否只包含空格，所以我们将检查字符串中是否有任何非空格字符。为此，我们将把模式 `“\S”`作为第一个输入参数传递给`search()`函数。此外，我们将把输入字符串作为第二个输入参数传递给`search()`函数。
*   如果`search()`函数返回 None，则意味着字符串中没有非空白字符。因此，我们将打印出该字符串只包含空白字符。
*   如果`search()`函数返回一个匹配对象，我们就说这个字符串包含了除空白字符以外的字符。

您可以在下面的示例中观察到这一点。

```py
from re import search

input_string = "   "
pattern = "\\S"
print("The input string is:", input_string)
if input_string == "":
    print("Input string is an empty string.")
else:
    match_object = search(pattern, input_string)
    if match_object is None:
        print("The string contains only whitespace characters.")
    else:
        print("The string contains characters other than whitespaces.") 
```

输出:

```py
The input string is:    
The string contains only whitespace characters.
```

## 结论

在本文中，我们讨论了在 Python 中检查字符串是否为空或空白的不同方法。在所有三种方法中，使用等式运算符而不是运算符的方法以及`len()`函数在逻辑上是不正确的。只有当我们确定输入字符串要么是空字符串，要么只包含空白字符时，才能使用它们。

使用`strip()` 方法、`isspace()`方法和`re.search()` 函数的方法是稳健的。这些方法可以在所有情况下使用。因此，我建议您使用这些方法来检查 Python 中给定的字符串是空的还是空白。

要了解更多关于编程的知识，你可以阅读这篇关于[数据建模工具](https://www.codeconquest.com/blog/data-modeling-tools-you-must-try-in-2022/)的文章。你可能也会喜欢这篇关于机器学习中[回归的文章](https://codinginfinite.com/regression-in-machine-learning-with-examples/)。你也可以看看这篇关于[数据分析师和数据科学家](https://www.codeconquest.com/blog/data-analyst-vs-data-scientist-skills-education-responsibilities-and-salaries/)的文章，文章比较了数据分析师和数据科学家的工资、教育和工作职责。