# 使用 Python globals()函数

> 原文：<https://www.askpython.com/python/built-in-methods/python-globals>

嘿，伙计们！在本文中，我们将详细讨论 **Python globals()函数**。

那么，让我们开始吧。

* * *

## 什么是 Python globals()函数？

使我们能够在整个程序的特定代码中访问所有全局变量及其值的列表。

globals()函数表示一个包含当前全局符号表的 [Python 字典](https://www.askpython.com/python/dictionary/python-dictionary-dict-tutorial)。

你可能会想到一个问题..

什么是符号表？

**符号表**是表示关于不同符号的信息的表结构。符号可以是编程代码的任何属性，如变量、[关键字](https://www.askpython.com/python/python-keywords)、[函数](https://www.askpython.com/python/python-functions)等。

此外，符号表指定了这些提到的符号的名称以及它们的对象类型、它们在整个程序中的作用范围等。

符号表可分为以下类型:

*   **全局表**:以字典的形式表示全局变量的信息。
*   **局部表**:以字典的形式表示局部变量的信息。

`Python locals() function`表示并返回当前局部符号表。要了解更多关于局部变量和局部变量()函数的信息，请访问 [Python 局部变量()函数](https://www.askpython.com/python/built-in-methods/python-locals-function)。

理解了 globals()函数的工作原理后，现在让我们来理解下一节中提到的 globals()函数的结构。

* * *

## 全局()函数的语法

`globals() function`不接受任何参数。它返回表示当前符号表值的[字典](https://www.askpython.com/python/dictionary/python-dictionary-dict-tutorial)。

**语法:**

```py
globals()

```

现在，让我们通过下面的例子来理解 globals()函数的实现。

* * *

## 通过实例实现 Python globals()函数

在下面的示例中，我们调用了 be globals()函数来解释输出，如下所示:

**举例:**

```py
print(globals())

```

globals()函数从全局符号表中返回一个值字典，它提供了关于模块、文件名等的详细信息。

**输出:**

```py
{'__cached__': None, '__builtins__': <module 'builtins' (built-in)>, '__name__': '__main__', '__spec__': None, '__file__': 'main.py', '__doc__': None, '__loader__': <_frozen_importlib.SourceFileLoader object at 0x7f31606f4e80>, '__package__': None}

```

### 显示全局和局部范围变量

接下来，我们将在函数中定义一个全局和局部变量，并调用 globals()函数来了解结果。

**举例:**

```py
varG = 20 # global variable
def var():
    varL = 100 # local variable

print(globals())

```

Python globals()函数返回一个字典，该字典给出了关于全局变量(varG)及其值以及函数、文件名等的信息。

但是，如果您注意到，globals()函数并没有表示关于局部变量(varL)的信息。这项工作将由 locals()函数来完成。

**输出:**

```py
{'__file__': 'main.py', 'varG': 20, '__loader__': <_frozen_importlib.SourceFileLoader object at 0x7f7ff13e7e48>, '__cached__': None, '__doc__': None, '__package__': None, '__name__': '__main__', '__spec__': None, '__builtins__': <module 'builtins' (built-in)>, 'var': <function var at 0x7f7ff1436bf8>}

```

### 修改和操作变量

我们甚至可以修改全局变量的值，如下所示:

**举例:**

```py
var = 100

globals()['var'] = 12

print('var:', var)
print(globals())

```

这里，我们指定 var = 100。此外，如上所示，我们使用 globals()函数将变量“var”的值更改为 12。

如果我们分析输出，globals 函数返回全局符号表的 dict 和变量的更新值，即 var = 12

**输出:**

```py
var: 12
{'__file__': 'main.py', '__builtins__': <module 'builtins' (built-in)>, '__doc__': None, '__spec__': None, '__name__': '__main__', '__package__': None, '__loader__': <_frozen_importlib.SourceFileLoader object at 0x7f3f83a1ae10>, '__cached__': None, 'var': 12}

```

* * *

## 结论

到此，我们就结束了这个话题。如果你遇到任何问题，欢迎在下面评论。在那之前，学习愉快！！

* * *

## 参考

*   Python 全局变量— JournalDev