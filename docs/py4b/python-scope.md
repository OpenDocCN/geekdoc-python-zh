# Python 范围

> 原文：<https://www.pythonforbeginners.com/basics/python-scope>

在用 python 编程时，我们必须处理各种各样的结构，比如变量、函数、模块、库等等。在一些情况下，在一个地方使用的变量名也可能在不同的地方使用，而与前面的定义没有任何关系。在这篇关于 python 作用域的文章中，我们将尝试理解 python 解释器如何处理变量的定义。

## Python 中的作用域是什么？

当我们在程序中定义一个变量、函数或类名时，它只能在程序的某个区域中被访问。在这个区域中，名字一旦被定义，就可以用来标识一个对象、一个变量或一个函数，这个区域称为作用域。根据变量或函数名的定义，范围可以从单个代码块(如函数)扩展到整个运行时环境。

范围的概念与名称空间密切相关，范围是作为名称空间实现的。我们可以把名称空间看作是一个将对象名称映射到对象的 [python 字典](https://www.pythonforbeginners.com/dictionary/how-to-use-dictionaries-in-python/)。字典的键对应于名称，值对应于 python 中的对象。

在 python 中，有四种类型的作用域定义，即内置作用域、全局作用域、局部作用域和封闭作用域。我们将在接下来的章节中研究所有这些。

## Python 中的内置作用域是什么？

python 中的内置作用域包含内置对象和函数定义。它是使用 python 最新版本中的内置模块实现的。

每当我们启动 python 解释器时，内置模块就会自动加载到我们的运行时环境中。因此，我们可以在程序中访问模块中定义的所有函数和对象，而无需导入它们。

像 print()、abs()、input()、int()、float()、string()、sum()、max()、sorted()等类似的函数，在使用前不需要导入，都是在内置作用域中定义的。我们可以看看内置作用域中可用的函数和对象定义，如下所示。

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

内置作用域是在解释器加载后创建的，随着 python 解释器的关闭而被销毁。builtins 模块中定义的所有名称都在程序的内置范围内。

## 什么是全局范围？

我们用来编写代码的 python 脚本被 python 解释器称为 __main__ 模块。与 __main__ 模块相关联的作用域称为全局作用域。

对于任何 python 程序，只能有一个全局范围。一旦程序启动，全局范围就被创建，并随着 python 程序的终止而被销毁。

我们可以从下面的程序中理解全局范围的概念。

```py
myNum1 = 10
myNum2 = 10

def add(num1, num2):
    temp = num1 + num2

    def print_sum():
        print(temp)

    return temp 
```

在上面的程序中，myNum1 和 myNum2 在程序的全局范围内。存在于全局范围内的对象是在任何代码块之外定义的。

## 什么是局部范围？

python 程序中的局部作用域是为函数等代码块定义的。python 程序中的每个函数都有自己的局部作用域，在这个作用域中定义了所有的变量和对象名。

当函数被任何其他函数调用时，函数的局部范围被加载。一旦函数终止，与之相关的局部作用域也会终止。

为了理解局部范围的概念，请看下面的例子。

```py
myNum1 = 10
myNum2 = 10

def add(num1, num2):
    temp = num1 + num2

    def print_sum():
        print(temp)

    return temp 
```

在上面的程序中，变量 num1、num2 和 temp 存在于 add()函数的局部作用域中。这些名称只在执行 add()函数之前存在。

## Python 中的封闭作用域是什么？

每当一个函数被定义在任何其他函数内部时，内部函数的作用域就被定义在外部函数的作用域内部。因此，外部函数的范围被称为内部函数的封闭范围。

我们可以访问一个函数中所有的变量名，这个函数已经在它的封闭作用域中定义了。但是，我们不能访问在内部函数中定义的外部函数中的变量名。从下面的例子可以更清楚地看出这一点。

```py
myNum1 = 10
myNum2 = 10

def add(num1, num2):
    temp = num1 + num2

    def print_sum():
        print(temp)

    return temp 
```

这里，print_sum()函数存在于 add()函数的局部作用域中。因此，在 add()函数中定义的变量名 num1、num2 和 temp 可在 print_sum()函数的作用域中访问。

## 结论

在本文中，我们研究了 python 中的范围概念。我们还研究了不同的作用域类型及其示例。要阅读其他 python 概念，如[列表理解](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python)，敬请关注。