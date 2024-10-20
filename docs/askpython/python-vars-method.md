# python vars()–查找 __dict__ 属性

> 原文：<https://www.askpython.com/python/built-in-methods/python-vars-method>

大家好！在本文中，我们将看看如何使用 Python **vars()** 函数，该函数返回对象的`__dict__`属性。

这是一个罕见的函数，但它有助于了解如何使用它，因为它在某些情况下很有用。现在让我们来看看这些情况，使用说明性的例子！

* * *

## Python 变量的语法()

该函数接受一个对象`obj`，其形式如下:

```py
vars([obj])

```

这将返回`obj`的`__dict__`属性，该属性包含对象的所有可写属性。

这里，`obj`可以是任何模块/类/类的实例等。

这里有几个例子，取决于参数的类型和数量。

*   如果你不提供任何参数，Python `vars()`将像`locals()`方法一样，返回一个包含当前本地符号表的字典。
*   因为它返回`__dict__`属性，如果对象没有这个属性，它将引发一个`TypeError`异常。

现在让我们看一些与不同对象相关的例子。

* * *

## 使用不带任何参数的 Python **变量**()

如前所述，这将充当`locals()`方法，并返回本地符号表的字典。

```py
print(vars())

```

**输出**

```py
{'__name__': '__main__', '__doc__': None, '__package__': None, '__loader__': <_frozen_importlib_external.SourceFileLoader object at 0x108508390>, '__spec__': None, '__annotations__': {}, '__builtins__': <module 'builtins' (built-in)>, '__file__': '/Users/vijay/home/python_vars_function.py', '__cached__': None, 'Data': <class '__main__.Data'>, 'd': <__main__.Data object at 0x108565048>, 'math': <module 'math' from '/Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/lib-dynload/math.cpython-36m-darwin.so'>}

```

## 在类对象上使用 Python vars()

如果`obj`是一个类类型，或者是该类的一个实例，让我们看看`vars()`在这种情况下做了什么。

让我们创建一个示例类`MyClass`并在它的`__init__()`方法上定义一些属性。

```py
class MyClass:
    def __init__(self, i, j):
        self.a = i
        self.b = j

m = MyClass(10, 20)
print(vars(m))
print(vars(MyClass))

```

**输出**

```py
{'a': 10, 'b': 20}
{'__module__': '__main__', '__init__': <function MyClass.__init__ at 0x000001C24EA129D8>, '__dict__': <attribute '__dict__' of 'MyClass' objects>, '__weakref__': <attribute '__weakref__' of 'MyClass' objects>, '__doc__': None}

```

正如您所看到的，对于类实例，`vars()`返回了所有相关的属性`a`和`b`，以及它们的值。

而在类`MyClass`的情况下，它被封装在`main`模块下，具有`__init__`方法，以及类的`__dict__`属性。

`vars()`调用所有的邓德方法，如`__repr__`、`__dict__`等。

因此，如果您可以直接调用这个函数，而不是调用 dunder 方法，会更方便。(尽管没有任何区别)

类似地，你可以在其他对象和类上使用`vars()`，比如`str`和`list`。

```py
print(vars(str))
print(vars(list))

```

这将显示两个类的所有相关属性和实例方法。

## 在模块上使用 vars()

我们也可以在一个模块上使用这个函数，找出它包含的所有方法，以及其他相关信息，甚至文档字符串！

比如你要看内置模块`antigravity`，(就是一个复活节彩蛋！)你确实导入了，看看`vars(antigravity)`

```py
import antigravity

print(vars(antigravity))

```

**样本输出(最后几行)**

```py
 {'__name__': 'antigravity', '__doc__': None, '__package__': '', 'TimeoutError': <class 'TimeoutError'>, 'open': <built-in function open>, 'quit': Use quit() or Ctrl-Z plus Return to exit, 'exit': Use exit() or Ctrl-Z plus Return to exit, 'copyright': Copyright (c) 2001-2018 Python Software Foundation.
All Rights Reserved.

Copyright (c) 2000 BeOpen.com.
All Rights Reserved.

Copyright (c) 1995-2001 Corporation for National Research Initiatives.
All Rights Reserved.
}

```

如果你在一个没有`__dict__`属性的对象(比如`int`)上使用`vars()`，它将引发一个`TypeError`。

```py
print(vars(12))

```

```py
Traceback (most recent call last):
  File "vars_example.py", line 12, in <module>
    print(vars(12))
TypeError: vars() argument must have __dict__ attribute

```

* * *

## 结论

在本文中，我们研究了 Python vars()函数，如果您想快速获得任何类/模块/对象的属性和所有有代表性的方法，这个函数很有用。

* * *

## 参考

*   关于 Python vars()函数的 JournalDev 文章

* * *