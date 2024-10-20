# 在 Python 中用 str.isdigit()检查字符串是否为数字

> 原文：<https://www.pythoncentral.io/how-to-check-if-a-string-is-a-number-in-python-including-unicode/>

编辑 1:本文已经过修改，以显示`str.isdigit()`和所提供的定制解决方案之间的差异。
编辑 2:样本也支持 Unicode！

通常你会想检查 Python 中的字符串是否是一个数字。这种情况经常发生，例如用户输入、从数据库获取数据(可能返回一个字符串)或读取包含数字的文件。根据您期望的数字类型，您可以使用几种方法。比如解析字符串、使用正则表达式或者只是尝试将它转换成一个数字，看看会发生什么。您还会经常遇到用 Unicode 编码的非 ASCII 数字。这些可能是也可能不是数字。例如๒，在泰语中是 2。不过仅仅是版权符号，而显然不是一个数字。

请注意，如果您在 Python 2.x 中执行以下代码，您必须将编码声明为 UTF-8/Unicode -如下:
【Python】
#-*-编码:utf-8 -*-

下面的函数可以说是检查一个字符串是否为数字的最快最简单的方法之一。它支持 str 和 Unicode，并将在 Python 3 和 Python 2 中工作。

## 检查 Python 字符串是否为数字

```py

def is_number(s):

try:

float(s)

return True

except ValueError:

pass
try: 
导入 unicode data
unicode data . numeric(s)
返回 True 
 except (TypeError，ValueError): 
通过
返回 False 

```

当我们测试这个函数时，我们得到如下结果:

*   [Python 3.x](#custom-tab-0-python-3-x)
*   [Python 2.x](#custom-tab-0-python-2-x)

*   [Python 3.x](#)

[python]
>>> # Testing as a str
>>> print(is_number('foo'))
False
>>> print(is_number('1'))
True
>>> print(is_number('1.3'))
True
>>> print(is_number('-1.37'))
True
>>> print(is_number('1e3'))
True
>>>
>>> # Testing Unicode
>>> # 5 in Arabic
>>> print(is_number('٥'))
True
>>> # 2 in Thai
>>> print(is_number('๒'))
True
>>> # 4 in Japanese
>>> print(is_number('四'))
True
>>> # Copyright Symbol
>>> print(is_number('©'))
False
[/python]

*   [Python 2.x](#)

[python]
>>> # Testing as a str
>>> print(is_number('foo'))
False
>>> print(is_number('1'))
True
>>> print(is_number('1.3'))
True
>>> print(is_number('-1.37'))
True
>>> print(is_number('1e3'))
True
>>>
>>> # Testing Unicode
>>> # 5 in Arabic
>>> print(is_number(u'٥'))
True
>>> # 2 in Thai
>>> print(is_number(u'๒'))
True
>>> # 4 in Japanese
>>> print(is_number(u'四'))
True
>>> # Copyright Symbol
>>> print(is_number(u'©'))
False
[/python]

我们首先尝试简单地将它转换/转换为浮点型，看看会发生什么。当字符串不是严格的 Unicode 时，这将适用于`str`和 Unicode 数据类型。然而，我们也尝试使用`unicodedata`模块来转换字符串。`unicodedata.numeric`功能是做什么的？一些 Unicode 字符被赋予了数字属性，如果它们是数字的话。`unicodedata.numeric`将简单地返回字符的数值，如果它存在的话。如果没有数值属性，你可以给`unicodedata.numeric`分配一个默认值，否则默认情况下会产生一个`ValueError`。

## 检查 Python 字符串是否是 Python 中的数字(str.isdigit)

Python 有一个方便的内置函数`str.isdigit`，可以让你检查一个字符串是否是一个数字。

### 数字对数字

确保当您使用`str.isdigit`函数时，您真正检查的是一个数字，而不是一个任意的数字。根据 Python 文档，数字的定义如下:

> 如果字符串中的所有字符都是数字并且至少有一个字符，则返回 true，否则返回 false。数字包括十进制字符和需要特殊处理的数字，如兼容性上标数字。
> [https://docs.python.org/3/library/stdtypes.html#str.isdigit](https://docs.python.org/3/library/stdtypes.html#str.isdigit "Python built-in str.isdigit")

这里有一个例子:

```py

>>> print('foo'.isdigit())

False

>>> print('baz'.isdigit())

False

>>> print('1'.isdigit())

True

>>> print('36'.isdigit())

True

>>> print('1.3'.isdigit())

False

>>> print('-1.37'.isdigit())

False

>>> print('-45'.isdigit())

False

>>> print('1e3'.isdigit())

False

```