# python 文档字符串

> 原文：<https://www.pythonforbeginners.com/basics/python-docstrings>

## 什么是 Docstring？

Python 文档字符串(或文档字符串)提供了一种将文档与 Python 模块、函数、类和方法相关联的便捷方式。

通过将字符串常量作为对象定义中的第一条语句来定义对象的文档。它是在源代码中指定的，像注释一样，用来记录特定的代码段。

与传统的源代码注释不同，docstring 应该描述函数做什么，而不是如何做。

所有函数都应该有一个 docstring。这允许程序在运行时检查这些注释，例如作为交互式帮助系统，或者作为元数据。

文档字符串可以通过对象的 __doc__ 属性来访问。

## Docstring 应该是什么样子？

文档字符串行应该以大写字母开始，以句点结束。第一行应该是简短的描述。

不要写对象的名字。如果文档字符串中有更多行，第二行应该是空白的，从视觉上将摘要与描述的其余部分分开。

下面几行应该是一个或多个段落，描述对象的调用约定，它的副作用等等。

## Docstring 示例

让我们展示一个多行 docstring 的例子:

```py
def my_function():
    """Do nothing, but document it.

    No, really, it doesn't do anything.
    """
    pass 
```

让我们看看打印出来后会是什么样子

```py
>>> print my_function.__doc__
Do nothing, but document it.

    No, really, it doesn't do anything. 
```

## 文档字符串的声明

以下 python 文件显示了 Python 源文件中的文档字符串声明:

```py
"""
Assuming this is file mymodule.py, then this string, being the
first statement in the file, will become the "mymodule" module's
docstring when the file is imported.
"""

class MyClass(object):
    """The class's docstring"""

    def my_method(self):
        """The method's docstring"""

def my_function():
    """The function's docstring""" 
```

## 如何访问文档字符串

下面是一个交互式会话，展示了如何访问文档字符串

```py
>>> import mymodule
>>> help(mymodule) 
```

假设这是文件 mymodule.py，那么在导入文件时，作为文件中第一条语句的这个字符串将成为 mymodule modules docstring。

```py
>>> help(mymodule.MyClass)
The class's docstring

>>> help(mymodule.MyClass.my_method)
The method's docstring

>>> help(mymodule.my_function)
The function's docstring 
```

## 更多阅读

*   [http://en.wikipedia.org/wiki/Docstring](https://en.wikipedia.org/wiki/Docstring "docstrings_wiki")
*   [http://docs . python . org/2/tutorial/control flow . html # tut-doc strings](https://docs.python.org/2/tutorial/controlflow.html#tut-docstrings "docstrings")
*   [http://onlamp.com/lpt/a/python/2001/05/17/docstrings.html](http://onlamp.com/lpt/a/python/2001/05/17/docstrings.html "onlamp_docstring")