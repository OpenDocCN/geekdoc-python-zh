# 使用 singledispatch 的 Python 3 函数重载

> 原文：<https://www.blog.pythonlibrary.org/2016/02/23/python-3-function-overloading-with-singledispatch/>

Python 最近在 **Python 3.4** 中增加了对函数重载的部分支持。他们通过向名为 **singledispatch** 的 **functools** 模块添加一个简洁的小装饰器来实现这一点。这个装饰器将把你的常规函数转换成一个单一的调度通用函数。但是请注意，singledispatch 只基于第一个参数的类型发生。让我们来看一个例子，看看这是如何工作的！

```py

from functools import singledispatch

@singledispatch
def add(a, b):
    raise NotImplementedError('Unsupported type')

@add.register(int)
def _(a, b):
    print("First argument is of type ", type(a))
    print(a + b)

@add.register(str)
def _(a, b):
    print("First argument is of type ", type(a))
    print(a + b)

@add.register(list)
def _(a, b):
    print("First argument is of type ", type(a))
    print(a + b)

if __name__ == '__main__':
    add(1, 2)
    add('Python', 'Programming')
    add([1, 2, 3], [5, 6, 7])

```

这里我们从 functools 导入 singledispatch，并将其应用于一个简单的函数，我们称之为 **add** 。这个函数是我们的总括函数，只有当其他修饰函数都不处理传递的类型时才会被调用。你会注意到我们目前处理整数、字符串和列表作为第一个参数。如果我们用其他东西调用我们的 add 函数，比如字典，那么它将引发 NotImplementedError。

尝试自己运行代码。您应该会看到如下所示的输出:

```py

First argument is of type  3
First argument is of type  <class>PythonProgramming
First argument is of type  <class>[1, 2, 3, 5, 6, 7]
Traceback (most recent call last):
  File "overloads.py", line 30, in <module>add({}, 1)
  File "/usr/local/lib/python3.5/functools.py", line 743, in wrapper
    return dispatch(args[0].__class__)(*args, **kw)
  File "overloads.py", line 5, in add
    raise NotImplementedError('Unsupported type')
NotImplementedError: Unsupported type 
```

如您所见，代码完全按照宣传的那样工作。它根据第一个参数的类型调用适当的函数。如果该类型未被处理，那么我们将引发 NotImplementedError。如果您想知道我们当前正在处理什么类型，您可以将下面这段代码添加到文件的末尾，最好是在引发错误的那一行之前:

```py

print(add.registry.keys())

```

这将打印出类似这样的内容:

```py

dict_keys([, <class>, <class>, <class>]) 
```

这告诉我们，我们可以处理字符串、整数、列表和对象(默认)。singledispatch 装饰器还支持装饰器堆栈。这允许我们创建一个可以处理多种类型的重载函数。让我们来看看:

```py

from functools import singledispatch
from decimal import Decimal

@singledispatch
def add(a, b):
    raise NotImplementedError('Unsupported type')

@add.register(float)
@add.register(Decimal)
def _(a, b):
    print("First argument is of type ", type(a))
    print(a + b)

if __name__ == '__main__':
    add(1.23, 5.5)
    add(Decimal(100.5), Decimal(10.789))

```

这基本上告诉 Python，add 函数重载之一可以处理 float 和 decimal。十进制类型作为第一个参数。如果您运行这段代码，您应该会看到如下内容:

```py

First argument is of type  6.73
First argument is of type  <class>111.2889999999999997015720510
dict_keys([<class>, <class>, <class>, <class> 
```

您可能已经注意到了这一点，但是由于所有这些函数的编写方式，您可以将 decorators 堆叠起来，将前面的示例和这个示例中的所有情况处理到一个重载函数中。然而，在正常的重载情况下，每个重载会调用不同的代码，而不是做相同的事情。

* * *

### 包扎

函数重载在其他语言中已经存在很长时间了。很高兴看到 Python 也添加了它。当然，其他一些语言允许多分派而不是单分派，这意味着它们查看不止一种参数类型。希望 Python 能在未来的版本中增加这个功能。还要注意，可以用 singledispatch 注册 lambdas 和预先存在的函数。有关完整的详细信息，请参见文档。

* * *

### 相关阅读

*   [functools 模块](https://docs.python.org/3.5/library/functools.html)的 Python 文档
*   PEP 0443 - [单调度通用函数](https://www.python.org/dev/peps/pep-0443/)
*   单个调度通用功能[条](http://lukasz.langa.pl/8/single-dispatch-generic-functions/)
*   Python 3.4 [单调度通用函数](https://julien.danjou.info/blog/2013/python-3.4-single-dispatch-generic-function)