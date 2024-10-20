# Python isidentifier()方法

> 原文：<https://www.askpython.com/python/string/python-isidentifier-method>

因此，今天在本教程中，我们将介绍**Python 的 isidentifier()方法**。

## 介绍

基本上，[标识符](https://docs.python.org/3.1/reference/lexical_analysis.html#identifiers)是用户给任何变量、类、对象、函数等的名字。这些名称对于唯一识别单个变量、类等非常重要。

因此，命名是任何变量、类、函数、对象等声明的重要部分。Python 对用户进行了限制，并为此命名过程提供了一些基本准则。

## 了解 Python isidentifier()方法

`isidentifier()`方法检查所提供的字符串是否有资格成为标识符，如果是，则相应地返回 **true** ，否则返回 **false** 。

下面给出了使用 Python `isidentifier()` 方法的语法。

```py
result = str.isidentifier()

```

这里，

*   **result** 存储方法返回的布尔值(真或假)，
*   **str** 是我们需要检查它是否是标识符的字符串。

## 使用 Python isidentifier()方法

现在我们已经对标识符的概念和 Python `isidentifier()`方法有了基本的了解，让我们举一些例子来理解这个方法的工作原理。

```py
string1 = "Askpython"
print(f"Is {string1} a valid identifier? ", string1.isidentifier())

string2 = "i" #an identifier may be of any length > 0
print(f"Is {string2} a valid identifier? ", string2.isidentifier())

string3 = "" #short length not allowed
print(f"Is {string3} a valid identifier? ", string3.isidentifier())

string4 = "_abcd1234" #an identifier may start with an underscore
print(f"Is {string4} a valid identifier? ", string4.isidentifier())

string5 = "1976" #strings starting with numbers are not identifiers
print(f"Is {string5} a valid identifier? ", string5.isidentifier())

```

**输出**:

```py
Is Askpython a valid identifier?  True
Is i a valid identifier?  True
Is  a valid identifier?  False
Is _abcd1234 a valid identifier?  True
Is 1976 a valid identifier?  False

```

这里，

*   对于**string 1**–‘ask python’是一个有效的标识符，因为它以一个字符开头，并且不包含任何特殊字符，
*   对于**string 2**–‘I’是一个有效的标识符，因为它不包含任何特殊字符，并且有足够的长度，
*   对于**string 3**–该字符串不包含任何字符，因此长度为 0。字符串中应该至少有一个字符有资格作为标识符，
*   对于**string 4**–它是一个有效的标识符，因为它以下划线(' _ ')开头，并且包含字符和数字，
*   对于**字符串 5**–‘1976’不是有效的标识符，因为它以数字开头。

## 结论

这就是本教程的内容。我们学习了内置的 Python `isidentifier()`方法。我们强烈建议读者浏览下面的参考链接。isidentifier()方法是一个 [Python 字符串](https://www.askpython.com/python/string)方法。

如有任何其他问题，请使用下面的评论随时联系。

## 参考

*   [String is identifier()](https://docs.python.org/3.3/library/stdtypes.html?highlight=isidentifier#str.isidentifier)–Python 文档，
*   [标识符和关键字](https://docs.python.org/3.1/reference/lexical_analysis.html#identifiers)–Python 文档，
*   支持非 ASCII 标识符-[PEP-3131](https://peps.python.org/pep-3131/)，
*   [python 化地检查变量名是否有效](https://stackoverflow.com/questions/36330860/pythonically-check-if-a-variable-name-is-valid)–stack overflow 问题。