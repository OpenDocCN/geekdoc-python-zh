# Python 中的命名空间是什么？

> 原文：<https://www.pythonforbeginners.com/basics/what-is-namespace-in-python>

名称空间是一个非常棒的想法——让我们多做一些吧！名称空间是用于组织 python 程序中分配给对象的名称的结构。在本文中，我们将了解 python 中名称和命名空间的概念。

## Python 中的名字是什么？

名称或标识符是我们在 python 中创建对象时赋予对象的名称。

这个名字就是我们在程序中使用的变量名。在 python 中，我们可以声明变量名并将它们分配给对象，如下所示。

```py
myInt = 1117
myString = "PythonForBeginners"
```

这里，我们定义了两个名为 myInt 和 myString 的变量，它们分别指向一个整数和一个字符串对象。

## 什么是名称空间？

简而言之，名称空间是名称和名称所引用的对象的详细信息的集合。我们可以把名称空间看作是一个 [python 字典](https://www.pythonforbeginners.com/dictionary/how-to-use-dictionaries-in-python/)，它把对象名称映射到对象。字典的键对应于名称，值对应于 python 中的对象。

在 python 中，有四种类型的命名空间，即内置命名空间、全局命名空间、局部命名空间和封闭命名空间。我们将在接下来的章节中研究它们。

## Python 中的内置命名空间是什么？

内置命名空间包含内置函数和对象的名称。它是在启动 python 解释器时创建的，只要解释器运行就存在，当我们关闭解释器时就被销毁。它包含内置数据类型、异常和函数的名称，如 print()和 input()。我们可以访问内置名称空间中定义的所有名称，如下所示。

```py
builtin_names = dir(__builtins__)
for name in builtin_names:
    print(name) 
```

输出:

```py
ArithmeticError
AssertionError
AttributeError
BaseException
BlockingIOError
BrokenPipeError
BufferError
BytesWarning
ChildProcessError
ConnectionAbortedError
ConnectionError
ConnectionRefusedError
ConnectionResetError
DeprecationWarning
EOFError
Ellipsis
EnvironmentError
Exception
False
FileExistsError
FileNotFoundError
FloatingPointError
FutureWarning
GeneratorExit
IOError
ImportError
ImportWarning
IndentationError
IndexError
InterruptedError
IsADirectoryError
KeyError
KeyboardInterrupt
LookupError
MemoryError
ModuleNotFoundError
NameError
None
NotADirectoryError
NotImplemented
NotImplementedError
OSError
OverflowError
PendingDeprecationWarning
PermissionError
ProcessLookupError
RecursionError
ReferenceError
ResourceWarning
RuntimeError
RuntimeWarning
StopAsyncIteration
StopIteration
SyntaxError
SyntaxWarning
SystemError
SystemExit
TabError
TimeoutError
True
TypeError
UnboundLocalError
UnicodeDecodeError
UnicodeEncodeError
UnicodeError
UnicodeTranslateError
UnicodeWarning
UserWarning
ValueError
Warning
ZeroDivisionError
__build_class__
__debug__
__doc__
__import__
__loader__
__name__
__package__
__spec__
abs
all
any
ascii
bin
bool
breakpoint
bytearray
bytes
callable
chr
classmethod
compile
complex
copyright
credits
delattr
dict
dir
divmod
enumerate
eval
exec
exit
filter
float
format
frozenset
getattr
globals
hasattr
hash
help
hex
id
input
int
isinstance
issubclass
iter
len
license
list
locals
map
max
memoryview
min
next
object
oct
open
ord
pow
print
property
quit
range
repr
reversed
round
set
setattr
slice
sorted
staticmethod
str
sum
super
tuple
type
vars
zip 
```

在上面的例子中，我们可以看到 python 3.8 的内置名称空间中定义了 152 个名称。

## python 中的全局命名空间是什么？

全局命名空间是在程序或模块级别定义的。它包含模块或主程序中定义的对象的名称。程序启动时会创建一个全局名称空间，该名称空间会一直存在，直到程序被 python 解释器终止。从下面的例子可以理解全局名称空间的概念。

```py
myNum1 = 10
myNum2 = 10

def add(num1, num2):
    temp = num1 + num2
    return temp 
```

在上面的例子中，myNum1 和 myNum2 位于程序的全局名称空间中。

## Python 中的本地命名空间是什么？

局部命名空间是为类、函数、循环或任何代码块定义的。代码块或函数中定义的名字是局部的。不能在定义变量名的代码块或函数之外访问变量名。本地命名空间在代码块或函数开始执行时创建，并在函数或代码块终止时终止。从下面的例子可以理解本地名称空间的概念。

```py
myNum1 = 10
myNum2 = 10

def add(num1, num2):
    temp = num1 + num2
    return temp
```

这里，变量名 num1、num2 和 temp 是在函数 add 的本地名称空间中定义的。

## 什么是封闭名称空间？

正如我们所知，我们可以在另一个代码块或函数中定义一个代码块或函数，在任何函数中定义的函数或代码块都可以访问外部函数或代码块的名称空间。因此，外部命名空间被称为内部函数或代码块的命名空间的封闭命名空间。这将从下面的例子中变得更加清楚。

```py
myNum1 = 10
myNum2 = 10

def add(num1, num2):
    temp = num1 + num2

    def print_sum():
        print(temp)

    return temp 
```

在上面的示例中，add()函数的本地名称空间是 print_sum()函数的封闭名称空间，因为 print_sum()是在 add()函数中定义的。

## 结论

在本文中，我们讨论了 python 中的名称和命名空间。我们还看到了不同类型的名称空间及其功能。你可以在[列表理解](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python)中阅读这篇文章。请继续关注更多内容丰富的文章。