# 了解 Python exec()方法

> 原文：<https://www.askpython.com/python/built-in-methods/python-exec-method>

所以今天在本教程中，我们来了解一下中的**Python exec()方法。**

## Python exec()方法

基本上，Python `exec()`方法以字符串的形式执行传递的代码集。它非常有用，因为它实际上支持动态执行。下面给出了该方法的语法。

```py
exec(object, globals, locals)

```

这里，`object`可以是一个字符串，一个打开的文件对象，或者一个代码对象。

*   **For string**–该字符串被解析为一组 Python 语句，然后被执行(除非出现语法错误)。
*   **对于一个打开的文件**–文件被解析直到 EOF 并执行。
*   **对于一个代码对象**–它被简单地执行。

并且两个可选参数`globals`和`locals`必须是用于全局和局部变量的[字典](https://www.askpython.com/python/dictionary/python-dictionary-dict-tutorial)。

现在我们已经对`exec()`方法有了一个基本的概念，让我们通过一个例子来理解它的工作原理。

```py
>>> exec("print('Hey!')")
Hey!
>>> exec("print(6+4)")
10

```

从上面的代码片段可以清楚地看到，`print()`语句被`exec()`方法成功地执行了，我们得到了想要的结果。

## 使用 Python exec()方法

现在让我们直接进入一些例子，探索在有和没有`globals`和`locals`参数的情况下`exec()`方法在 Python 中是如何工作的。

### 1.没有全局和局部参数

在前面的例子中，我们只是在 Python 中执行了一些指令集，将对象参数传递给了`exec()`方法。但是，我们在当前范围内没有看到这些名称。

现在让我们在调用`exec()`方法之前，使用 [dir()](https://docs.python.org/3/library/functions.html#dir) 方法获得当前方法和名称的列表，其中包含了`math`模块。

```py
from math import *

exec("print(pow(2, 5))")

exec("print(dir())")

```

**输出:**

```py
32.0
['__annotations__', '__builtins__', '__cached__', '__doc__', '__file__', '__loader__', '__name__', '__package__', '__spec__', 'acos', 'acosh', 'asin', 'asinh', 'atan', 'atan2', 'atanh', 'ceil', 'copysign', 'cos', 'cosh', 'degrees', 'e', 'erf', 'erfc', 'exp', 'expm1', 'fabs', 'factorial', 'floor', 'fmod', 'frexp', 'fsum', 'gamma', 'gcd', 'hypot', 'inf', 'isclose', 'isfinite', 'isinf', 'isnan', 'ldexp', 'lgamma', 'log', 'log10', 'log1p', 'log2', 'modf', 'nan', 'pi', 'pow', 'radians', 'remainder', 'sin', 'sinh', 'sqrt', 'tan', 'tanh', 'tau', 'trunc']

```

正如你所看到的，包括`builtins`在内的各种方法，以及来自`math`模块的方法，现在都在当前的作用域中，可供 Python `exec()`方法使用。

当考虑执行**动态 Python 代码**时，这提出了一个很大的安全问题。用户可能包含一些模块来访问系统命令，这些命令甚至会使您的计算机崩溃。使用`globals`和`locals`参数，我们可以限制`exec()`超越我们想要访问的方法。

### 2.使用全局参数

现在让我们看看如何使用带有**全局**参数的 Python `exec()`方法。Python 允许我们只传递和指定我们希望`exec()`方法从**内置**模块中访问(以字典的形式)的方法。

```py
def squareNo(a):
    return a*a

exec('print(squareit(10))',{"__builtins__":{"squareit": squareNo, "print": print}})

exec("print(dir())")

```

**输出:**

```py
100
['__annotations__', '__builtins__', '__cached__', '__doc__', '__file__', '__loader__', '__name__', '__package__', '__spec__', 'squareNo']

```

在上面的代码中，我们传递了一个包含方法`squareNo()`(映射到一个自定义名称 **squareit** )和`print()`的字典。注意，使用**内置**方法中的任何其他方法都会引发一个`TypeError`。

### 3.带局部变量参数

当我们只传递`local`参数([字典](https://www.askpython.com/python/dictionary/python-dictionary-dict-tutorial))时，默认情况下，所有的**内置**方法也是可用的，除非我们显式排除它们。

看看下面的例子，这里虽然我们已经指定了`locals`字典，但是所有的**内置**和**数学**模块方法在当前作用域中都是可用的。

```py
from math import *
def squareNo(a):
    return a*a

#global And local parameters
exec('print(pow(4,3))', {"squareit": squareNo, "print": print})

exec("print(dir())")

```

**输出:**

```py
64
['__annotations__', '__builtins__', '__cached__', '__doc__', '__file__', '__loader__', '__name__', '__package__', '__spec__', 'acos', 'acosh', 'asin', 'asinh', 'atan', 'atan2', 'atanh', 'ceil', 'copysign', 'cos', 'cosh', 'degrees', 'e', 'erf', 'erfc', 'exp', 'expm1', 'fabs', 'factorial', 'floor', 'fmod', 'frexp', 'fsum', 'gamma', 'gcd', 'hypot', 'inf', 'isclose', 'isfinite', 'isinf', 'isnan', 'ldexp', 'lgamma', 'log', 'log10', 'log1p', 'log2', 'modf', 'nan', 'pi', 'pow', 'radians', 'remainder', 'sin', 'sinh', 'sqrt', 'squareNo', 'tan', 'tanh', 'tau', 'trunc']

```

因此，现在明确排除了**内置**。

```py
from math import *

def squareNo(a):
    return a*a

#explicitly excluding built-ins
exec('print(pow(4,3))', {"__builtins__": None},{"squareit": squareNo, "print": print})

exec("print(dir())")

```

**输出:**

```py
Traceback (most recent call last):
  File "C:/Users/sneha/Desktop/test.py", line 7, in <module>
    exec('print(pow(4,3))', {"__builtins__": None},{"squareit": squareNo, "print": print})
  File "<string>", line 1, in <module>
TypeError: 'NoneType' object is not subscriptable

```

在上面的代码中，限制`exec()`方法只能使用传递的(**局部变量**)方法实际上使得`pow()`方法不可访问。因此，在运行它时，我们得到了`TypeError`。

## Python 中的 exec()与 eval()

虽然`eval()`和`exec()`方法做的工作几乎相同，但是它们之间有两个主要的区别。

1.  **eval()** 只能执行一个表达式，而 **exec()** 可以用来执行动态创建的语句或程序，可以包括循环、`if-else`语句、函数和`class`定义。
2.  **eval()** 在执行一个特定的表达式后返回值，而 **exec()** 基本上什么也不返回，只是忽略这个值。

## 结论

今天就到这里吧。希望你对 Python `exec()`方法的工作和使用有一个清晰的理解。

如有任何与 Python `exec()`相关的问题，欢迎在下面的评论中提问。

## 参考

*   [exec 语句](https://docs.python.org/2.0/ref/exec.html)–Python 文档，
*   eval、exec 和 compile 有什么区别？–堆栈溢出问题，
*   python exec()–journal dev Post。