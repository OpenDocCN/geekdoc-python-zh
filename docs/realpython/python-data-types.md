# Python 中的基本数据类型

> 原文：<https://realpython.com/python-data-types/>

*立即观看**本教程有真实 Python 团队创建的相关视频课程。配合文字教程一起看，加深理解:[**Python 中的基本数据类型**](/courses/python-data-types/)

现在你知道了如何与 Python 解释器交互并执行 Python 代码。是时候深入研究 Python 语言了。首先讨论 Python 内置的基本数据类型。

**以下是你将在本教程中学到的内容:**

*   您将了解 Python 中内置的几种基本的**数字、字符串**和**布尔**类型。本教程结束时，您将熟悉这些类型的对象的外观，以及如何表示它们。
*   您还将大致了解 Python 的内置**函数。**这些是预先写好的代码块，你可以调用它们来做有用的事情。你已经看到了[内置的`print()`函数](https://realpython.com/courses/python-print/)，但是还有很多其他的。

**免费 PDF 下载:** [Python 3 备忘单](https://realpython.com/bonus/python-cheat-sheet-short/)

***参加测验:****通过我们的交互式“Python 中的基本数据类型”测验来测试您的知识。完成后，您将收到一个分数，以便您可以跟踪一段时间内的学习进度:*

*[参加测验](/quizzes/python-data-types/)

## 整数

在 Python 3 中，整数值的长度实际上没有限制。当然，它会受到系统内存容量的限制，但除此之外，任何整数都可以是您需要的长度:

>>>

```py
>>> print(123123123123123123123123123123123123123123123123 + 1)
123123123123123123123123123123123123123123123124
```

Python 将不带任何前缀的十进制数字序列解释为十进制数:

>>>

```py
>>> print(10)
10
```

以下字符串可以添加到整数值的前面，以表示基数不是 10:

| 前缀 | 解释 | 基础 |
| --- | --- | --- |
| `0b`(零+小写字母`'b'` )
`0B`(零+大写字母`'B'`) | 二进制的 | Two |
| `0o`(零+小写字母`'o'` )
`0O`(零+大写字母`'O'`) | 八进制的 | eight |
| `0x`(零+小写字母`'x'` )
`0X`(零+大写字母`'X'`) | 十六进制的 | Sixteen |

例如:

>>>

```py
>>> print(0o10)
8

>>> print(0x10)
16

>>> print(0b10)
2
```

有关非十进制整数值的更多信息，请参见下面的维基百科网站:[二进制](https://en.wikipedia.org/wiki/Binary_number)、[八进制](https://en.wikipedia.org/wiki/Octal)和[十六进制](https://en.wikipedia.org/wiki/Hexadecimal)。

Python 整数的底层类型被称为`int`，与用于指定它的基数无关:

>>>

```py
>>> type(10)
<class 'int'>
>>> type(0o10)
<class 'int'>
>>> type(0x10)
<class 'int'>
```

**注意:**这是一个很好的时机来说明，如果你想在 REPL 会话中显示一个值，你不需要使用`print()`函数。只需在`>>>`提示符下输入值，然后点击 `Enter` 就会显示出来:

>>>

```py
>>> 10
10
>>> 0x10
16
>>> 0b10
2
```

本系列教程中的许多例子都将使用这个特性。

请注意，这在脚本文件中不起作用。在脚本文件中，一个值单独出现在一行不会有任何作用。

[*Remove ads*](/account/join/)

## 浮点数

Python 中的`float`类型指定一个浮点数。`float`数值用小数点指定。可选地，后跟正整数或负整数的字符`e`或`E`可以被附加以指定[科学符号](https://en.wikipedia.org/wiki/Scientific_notation):

>>>

```py
>>> 4.2
4.2
>>> type(4.2)
<class 'float'>
>>> 4.
4.0
>>> .2
0.2

>>> .4e7
4000000.0
>>> type(.4e7)
<class 'float'>
>>> 4.2e-4
0.00042
```

> 深潜:浮点表示
> 
> 下面是关于 Python 如何在内部表示浮点数的更深入的信息。您可以在 Python 中轻松使用浮点数，而不必理解到这种程度，所以如果这看起来过于复杂，也不用担心。这里提供了一些信息，以防你感到好奇。
> 
> 根据 [IEEE 754](https://en.wikipedia.org/wiki/IEEE_754_revision) 标准，几乎所有平台都将 Python `float`值表示为 64 位“双精度”值。在这种情况下，一个浮点数可以拥有的最大值大约是 1.8 ⨉ 10 <sup>308</sup> 。Python 将通过字符串`inf`指示一个大于该数字的数字:
> 
> >>>
> 
> ```py
> `>>> 1.79e308
> 1.79e+308
> >>> 1.8e308
> inf` 
> ```
> 
> 非零数字最接近零的值大约是 5.0 ⨉ 10 <sup>-324</sup> 。任何比这更接近零的东西实际上都是零:
> 
> >>>
> 
> ```py
> `>>> 5e-324
> 5e-324
> >>> 1e-325
> 0.0` 
> ```
> 
> 浮点数在内部表示为二进制(基数为 2)分数。大多数十进制分数不能精确地表示为二进制分数，因此在大多数情况下，浮点数的内部表示是实际值的近似值。实际上，实际值和表示值之间的差异非常小，通常不会造成重大问题。

**延伸阅读:**有关 Python 中浮点表示法的更多信息以及相关的潜在缺陷，请参见 Python 文档中的[浮点运算:问题和限制](https://docs.python.org/3.6/tutorial/floatingpoint.html)。

## 复数

[复数](https://realpython.com/python-complex-numbers/)指定为`<real part>+<imaginary part>j`。例如:

>>>

```py
>>> 2+3j
(2+3j)
>>> type(2+3j)
<class 'complex'>
```

## 字符串

字符串是字符数据的序列。Python 中的[字符串类型](https://realpython.com/python-strings/)称为`str`。

字符串可以用单引号或双引号分隔。开始分隔符和匹配的结束分隔符之间的所有字符都是字符串的一部分:

>>>

```py
>>> print("I am a string.")
I am a string.
>>> type("I am a string.")
<class 'str'>

>>> print('I am too.')
I am too.
>>> type('I am too.')
<class 'str'>
```

Python 中的字符串可以包含任意多的字符。唯一的限制是你的机器的内存资源。字符串也可以是空的:

>>>

```py
>>> ''
''
```

如果您想包含一个引号字符作为字符串本身的一部分呢？你的第一反应可能是尝试这样的事情:

>>>

```py
>>> print('This string contains a single quote (') character.')
SyntaxError: invalid syntax
```

如你所见，这并不奏效。本例中的字符串以单引号开始，因此 Python 假设下一个单引号，即括号中的单引号，是字符串的一部分，是结束分隔符。最后一个单引号是个迷，导致了所示的[语法错误](https://realpython.com/invalid-syntax-python/)。

如果您想在字符串中包含任何一种类型的引号字符，最简单的方法是用另一种类型来分隔字符串。如果字符串包含单引号，请用双引号将其分隔开，反之亦然:

>>>

```py
>>> print("This string contains a single quote (') character.")
This string contains a single quote (') character.

>>> print('This string contains a double quote (") character.')
This string contains a double quote (") character.
```

### 字符串中的转义序列

有时，您希望 Python 以不同的方式解释字符串中的字符或字符序列。这可能以两种方式之一发生:

*   您可能希望取消某些字符通常在字符串中给出的特殊解释。
*   您可能希望对字符串中通常按字面理解的字符进行特殊解释。

您可以使用反斜杠(`\`)字符来完成此操作。字符串中的反斜杠字符表示它后面的一个或多个字符应该被特殊处理。(这被称为转义序列，因为反斜杠导致后续字符序列“转义”其通常含义。)

让我们看看这是如何工作的。

#### 抑制特殊字符含义

您已经看到了当您试图在字符串中包含引号字符时可能会遇到的问题。如果字符串由单引号分隔，则不能直接指定单引号字符作为字符串的一部分，因为对于该字符串，单引号有特殊的含义，它终止字符串:

>>>

```py
>>> print('This string contains a single quote (') character.')
SyntaxError: invalid syntax
```

在字符串中的引号字符前面指定一个反斜杠会“转义”它，并导致 Python 取消其通常的特殊含义。然后，它被简单地解释为文字单引号字符:

>>>

```py
>>> print('This string contains a single quote (\') character.')
This string contains a single quote (') character.
```

这同样适用于由双引号分隔的字符串:

>>>

```py
>>> print("This string contains a double quote (\") character.")
This string contains a double quote (") character.
```

以下是转义序列表，这些转义序列会导致 Python 禁止对字符串中的字符进行通常的特殊解释:

| 转义
序列 | 反斜杠后的
字符的通常解释 | “逃”的解释 |
| --- | --- | --- |
| `\'` | 用单引号开始分隔符终止字符串 | 文字单引号(`'`)字符 |
| `\"` | 用双引号开始分隔符终止字符串 | 文字双引号(`"`)字符 |
| `\<newline>` | 终止输入线 | [换行被忽略](https://stackoverflow.com/questions/48693600/what-does-the-newline-escape-sequence-mean-in-python) |
| `\\` | 引入转义序列 | 文字反斜杠(`\`)字符 |

通常，换行符终止行输入。所以在字符串中间按下 `Enter` 会让 Python 认为它是不完整的:

>>>

```py
>>> print('a

SyntaxError: EOL while scanning string literal
```

要将一个字符串拆分成多行，请在每个换行符前加一个反斜杠，换行符将被忽略:

>>>

```py
>>> print('a\
... b\
... c')
abc
```

要在字符串中包含文字反斜杠，请用反斜杠对其进行转义:

>>>

```py
>>> print('foo\\bar')
foo\bar
```

#### 对字符应用特殊含义

接下来，假设您需要创建一个包含制表符的字符串。一些文本编辑器可能允许您直接在代码中插入制表符。但是许多程序员认为这是一种糟糕的做法，原因有几个:

*   计算机可以区分制表符和一系列空格字符，但你不能。对于阅读代码的人来说，制表符和空格字符在视觉上是无法区分的。
*   一些文本编辑器被配置为通过将制表符扩展到适当的空格数来自动消除制表符。
*   一些 Python REPL 环境不会在代码中插入制表符。

在 Python(以及几乎所有其他常见的计算机语言)中，制表符可以由转义序列`\t`指定:

>>>

```py
>>> print('foo\tbar')
foo     bar
```

转义序列`\t`导致`t`字符失去其通常的含义，即字面上的`t`。相反，该组合被解释为制表符。

以下是导致 Python 应用特殊含义而不是字面解释的转义序列列表:

| 换码顺序 | “逃”的解释 |
| --- | --- |
| `\a` | ASCII 字符(`BEL`) |
| `\b` | ASCII 退格(`BS`)字符 |
| `\f` | ASCII 换页符(`FF`)字符 |
| `\n` | ASCII 换行(`LF`)字符 |
| `\N{<name>}` | 具有给定`<name>`的 Unicode 数据库中的字符 |
| `\r` | ASCII 回车(`CR`)字符 |
| `\t` | ASCII 水平制表符(`TAB`)字符 |
| `\uxxxx` | 带 16 位十六进制值的 Unicode 字符`xxxx` |
| `\Uxxxxxxxx` | 具有 32 位十六进制值的 Unicode 字符`xxxxxxxx` |
| `\v` | ASCII 垂直制表符(`VT`)字符 |
| `\ooo` | 具有八进制值的字符`ooo` |
| `\xhh` | 带十六进制值的字符`hh` |

示例:

>>>

```py
>>> print("a\tb")
a    b
>>> print("a\141\x61")
aaa
>>> print("a\nb")
a
b
>>> print('\u2192  \N{rightwards arrow}')
→ →
```

这种类型的转义序列通常用于插入不易从键盘生成或者不易阅读或打印的字符。

[*Remove ads*](/account/join/)

### 原始字符串

原始字符串文字的前面是`r`或`R`，这表示相关字符串中的转义序列不被翻译。反斜杠字符留在字符串中:

>>>

```py
>>> print('foo\nbar')
foo
bar
>>> print(r'foo\nbar')
foo\nbar

>>> print('foo\\bar')
foo\bar
>>> print(R'foo\\bar')
foo\\bar
```

### 三重引号字符串

在 Python 中还有另一种分隔字符串的方法。三重引号字符串由三个单引号或三个双引号组成的匹配组分隔。转义序列在三重引号字符串中仍然有效，但是单引号、双引号和换行符可以在不转义它们的情况下包含在内。这为创建包含单引号和双引号的字符串提供了一种便捷的方法:

>>>

```py
>>> print('''This string has a single (') and a double (") quote.''')
This string has a single (') and a double (") quote.
```

因为可以包含换行符而不用转义它们，所以这也允许多行字符串:

>>>

```py
>>> print("""This is a
string that spans
across several lines""")
This is a
string that spans
across several lines
```

在即将到来的 Python 程序结构教程中，您将看到如何使用三重引号字符串向 Python 代码添加解释性注释。

## 布尔类型、布尔上下文和“真实性”

Python 3 提供了一种[布尔数据类型](https://realpython.com/python-boolean/)。布尔类型的对象可能有两个值，`True`或`False`:

>>>

```py
>>> type(True)
<class 'bool'>
>>> type(False)
<class 'bool'>
```

正如您将在接下来的教程中看到的，Python 中的表达式通常在布尔上下文中计算，这意味着它们被解释为代表真或假。在布尔上下文中为真的值有时被称为“真”，而在布尔上下文中为假的值被称为“假”(你也可能看到“falsy”拼成“falsey”)

布尔类型对象的“真”是不言而喻的:等于`True`的布尔对象是真(true)，等于`False`的布尔对象是假(false)。但是非布尔对象也可以在布尔上下文中进行评估，并确定为真或假。

在接下来的关于 Python 中的运算符和表达式的教程中，当您遇到逻辑运算符时，您将会学到更多关于在布尔上下文中计算对象的知识。

## 内置函数

Python 解释器支持许多内置的函数:从 Python 3.6 开始有 68 个函数。在接下来的讨论中，当它们出现在上下文中时，您将会涉及其中的许多内容。

现在，接下来是一个简短的概述，只是为了给大家一个可用的感觉。更多细节参见关于内置函数的 [Python 文档。以下许多描述涉及到将在未来教程中讨论的主题和概念。](https://docs.python.org/3.6/library/functions.html)

### 数学

| 功能 | 描述 |
| --- | --- |
| [T2`abs()`](https://realpython.com/python-absolute-value/#using-the-built-in-abs-function-with-numbers) | 返回一个数字的绝对值 |
| `divmod()` | 返回整数除法的商和余数 |
| [T2`max()`](https://realpython.com/python-min-and-max/) | 返回 iterable 中给定参数或项目的最大值 |
| [T2`min()`](https://realpython.com/python-min-and-max/) | 返回 iterable 中最小的给定参数或项 |
| `pow()` | 对一个数字进行幂运算 |
| `round()` | 对浮点值进行舍入 |
| `sum()` | 对 iterable 的各项求和 |

[*Remove ads*](/account/join/)

### 类型转换

| 功能 | 描述 |
| --- | --- |
| `ascii()` | 返回包含对象的可打印表示形式的字符串 |
| `bin()` | 将整数转换为二进制字符串 |
| `bool()` | 将参数转换为布尔值 |
| `chr()` | 返回由整数参数给出的字符的字符串表示形式 |
| `complex()` | 返回由参数构造的复数 |
| `float()` | 返回由数字或字符串构造的浮点对象 |
| `hex()` | 将整数转换为十六进制字符串 |
| `int()` | 返回由数字或字符串构造的整数对象 |
| `oct()` | 将整数转换为八进制字符串 |
| `ord()` | 返回字符的整数表示形式 |
| `repr()` | 返回包含对象的可打印表示形式的字符串 |
| `str()` | 返回对象的字符串版本 |
| `type()` | 返回对象的类型或创建新的类型对象 |

### 可迭代程序和迭代器

| 功能 | 描述 |
| --- | --- |
| [T2`all()`](https://realpython.com/python-all/) | 如果 iterable 的所有元素都为真，则返回`True` |
| [T2`any()`](https://realpython.com/any-python/) | 如果 iterable 的任何元素为真，则返回`True` |
| [T2`enumerate()`](https://realpython.com/python-enumerate/) | 从 iterable 返回包含索引和值的元组列表 |
| [T2`filter()`](https://realpython.com/python-filter-function/) | 从 iterable 中筛选元素 |
| `iter()` | 返回迭代器对象 |
| [T2`len()`](https://realpython.com/len-python-function/) | 返回对象的长度 |
| [T2`map()`](https://realpython.com/python-map-function/) | 将函数应用于 iterable 的每一项 |
| `next()` | 从迭代器中检索下一项 |
| `range()` | 生成一系列整数值 |
| `reversed()` | 返回反向迭代器 |
| `slice()` | 返回一个`slice`对象 |
| [T2`sorted()`](https://realpython.com/python-sort/) | 从 iterable 返回一个排序列表 |
| [T2`zip()`](https://realpython.com/python-zip-function/) | 创建一个迭代器，从可迭代对象中聚合元素 |

### 复合数据类型

| 功能 | 描述 |
| --- | --- |
| `bytearray()` | 创建并返回一个`bytearray`类的对象 |
| `bytes()` | 创建并返回一个`bytes`对象(类似于`bytearray`，但不可变) |
| `dict()` | 创建一个`dict`对象 |
| `frozenset()` | 创建一个`frozenset`对象 |
| `list()` | 创建一个`list`对象 |
| `object()` | 创建新的无特征对象 |
| `set()` | 创建一个`set`对象 |
| `tuple()` | 创建一个`tuple`对象 |

### 类、属性和继承

| 功能 | 描述 |
| --- | --- |
| `classmethod()` | 返回函数的类方法 |
| `delattr()` | 从对象中删除属性 |
| `getattr()` | 返回对象的命名属性的值 |
| `hasattr()` | 如果对象具有给定的属性，则返回`True` |
| `isinstance()` | 确定对象是否是给定类的实例 |
| `issubclass()` | 确定一个类是否是给定类的子类 |
| `property()` | 返回类的属性值 |
| `setattr()` | 设置对象的命名属性的值 |
| [T2`super()`](https://realpython.com/python-super/) | 返回一个代理对象，该对象将方法调用委托给父类或同级类 |

### 输入/输出

| 功能 | 描述 |
| --- | --- |
| `format()` | 将值转换为格式化的表示形式 |
| `input()` | 从控制台读取输入 |
| `open()` | 打开一个文件并返回一个 file 对象 |
| `print()` | 打印到文本流或控制台 |

### 变量、引用和范围

| 功能 | 描述 |
| --- | --- |
| `dir()` | 返回当前本地范围内的名称列表或对象属性列表 |
| `globals()` | 返回表示当前全局符号表的字典 |
| `id()` | 返回对象的标识 |
| `locals()` | 更新并返回表示当前本地符号表的字典 |
| `vars()` | 返回[模块](https://realpython.com/python-modules-packages/)、类或对象的`__dict__`属性 |

### 杂项

| 功能 | 描述 |
| --- | --- |
| `callable()` | 如果对象显示为可调用，则返回`True` |
| `compile()` | 将源代码编译成代码或 AST 对象 |
| [T2`eval()`](https://realpython.com/python-eval-function/) | 计算 Python 表达式 |
| `exec()` | 实现 Python 代码的动态执行 |
| `hash()` | 返回对象的哈希值 |
| `help()` | 调用内置帮助系统 |
| `memoryview()` | 返回一个内存视图对象 |
| `staticmethod()` | 返回函数的静态方法 |
| `__import__()` | 由`import`语句调用 |

## 结论

在本教程中，您了解了 Python 提供的内置**数据类型**和**函数**。

到目前为止给出的例子都只处理和显示常量值。在大多数程序中，您通常会希望创建在程序执行时值会发生变化的对象。

进入下一个教程，学习 Python **变量。**

***参加测验:****通过我们的交互式“Python 中的基本数据类型”测验来测试您的知识。完成后，您将收到一个分数，以便您可以跟踪一段时间内的学习进度:*

*[参加测验](/quizzes/python-data-types/)*

*[« Interacting with Python](https://realpython.com/interacting-with-python/)[Basic Data Types in Python](#)[Variables in Python »](https://realpython.com/python-variables/)

*立即观看**本教程有真实 Python 团队创建的相关视频课程。配合文字教程一起看，加深理解:[**Python 中的基本数据类型**](/courses/python-data-types/)*******