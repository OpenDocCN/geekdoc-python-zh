# Python 中注释和文档字符串的区别

> 原文：<https://www.pythonforbeginners.com/comments/difference-between-comments-and-docstrings-in-python>

注释用于增加源代码的可读性和可理解性。python 注释可以是单行注释，也可以是使用单行注释或多行字符串常量编写的多行注释。文档字符串或文档字符串在 python 中也是多行字符串常量，但是它们有非常特殊的属性，不像 [python 注释](https://www.pythonforbeginners.com/comments/comments-in-python)。在本文中，我们将看看 python 中注释和文档字符串之间的区别。

## python 中的注释声明

python 中的单行注释使用#符号声明，如下所示。

```py
#This is a single line comment.
```

Python 基本上没有多行注释，但是我们可以使用多个单行注释在 python 中编写多行注释，如下所示。示例中给出的函数将一个数字及其平方作为键值对添加到一个 [python 字典](https://www.pythonforbeginners.com/dictionary/how-to-use-dictionaries-in-python/)中。

```py
#This is a multiline comment
#written using single line comments
def add_square_to_dict(x,mydict):
    a=x*x
    mydict[str(x)]=a
    return mydict
```

我们还可以使用多行字符串常量在 python 中实现多行注释，如下所示。这里声明了多行字符串，但没有将其赋给任何变量，因为没有为它分配内存，它的工作方式就像一个注释。

```py
"""This is a multiline comment 
written using multiline strings """
```

我们应该记住，使用#符号编写的注释不需要遵循缩进规则，但是使用多行字符串编写的注释必须遵循声明它们的块的缩进。

## python 中文档字符串的声明

docstring 是与任何 python 对象或模块相关联的字符串常量。该对象可以是类、方法或函数。docstring 的编写类似于使用多行字符串的多行注释，但它必须是对象定义中的第一条语句。

python 中类的 docstring 声明如下。

```py
class MyNumber():
    """This is the docstring of this class.

    It describes what this class does and all its attributes."""
    def __init__(self, value):
        self.value=value
```

方法的 docstring 声明如下。

```py
class MyNumber():
    """This is the docstring of this class.

    It describes what this class does and all its attributes."""
    def __init__(self, value):
        self.value=value
    def increment(self):
        """This is the docstring for this method.

        It describes what the method does, what are its calling conventions and
        what are its side effects"""
        self.value=self.value+1
        return self.value
```

函数的 docstring 声明如下。

```py
def decrement(number):
    """This is the docstring of this function.

    It describes what the function does, what are its calling conventions and
        what are its side effects"""
    number=number-1
    return number
```

## 在 python 中访问注释

执行程序时不能访问注释，因为它们不与任何对象相关联。只有当某人有权访问源代码文件时，才能访问注释。

## 在 python 中访问文档字符串

我们可以使用任何 python 对象的`__doc__`属性来访问与该对象相关联的 docstring，如下所示。

一个类的 docstring 可以由`className.__doc__` 访问，如下所示。

```py
class MyNumber():
    """This is the docstring of this class.

    It describes what this class does and all its attributes."""
    def __init__(self, value):
        self.value=value
    def increment(self):
        """This is the docstring for this method.

        It describes what the method does, what are its calling conventions and
        what are its side effects"""
        self.value=self.value+1
        return self.value
print (MyNumber.__doc__)
```

输出:

```py
This is the docstring of this class.

    It describes what this class does and all its attributes.
```

方法的 docstring 可以使用`className.methodName.__doc__`来访问，如下所示。

```py
class MyNumber():
    """This is the docstring of this class.

    It describes what this class does and all its attributes."""
    def __init__(self, value):
        self.value=value
    def increment(self):
        """This is the docstring for this method.

        It describes what the method does, what are its calling conventions and
        what are its side effects"""
        self.value=self.value+1
        return self.value
print (MyNumber.increment.__doc__) 
```

输出:

```py
This is the docstring for this method.

        It describes what the method does, what are its calling conventions and
        what are its side effects
```

可以使用`functionName.__doc__`访问函数的 docstring，如下所示。

```py
 def decrement(number):
    """This is the docstring of this function.

    It describes what the function does, what are its calling conventions and
        what are its side effects"""
    number=number-1
    return number
print (decrement.__doc__)
```

输出:

```py
This is the docstring of this function.

    It describes what the function does, what are its calling conventions and
        what are its side effects
```

## 在 python 中使用注释的目的

注释用于增加代码的可理解性，它们通常解释为什么在程序中使用了一个语句。

## 在 python 中使用 docstring 的目的

docstring 以大写字母开始，以句点结束，它描述了与之相关联的对象的元数据，包括参数、调用约定、副作用等。文档字符串用于将文档与 python 中的类、方法和函数等对象相关联，它们描述了对象的功能。

## 结论

在本文中，我们通过查看注释和文档字符串在源代码中是如何声明的以及它们的用途来了解它们之间的区别。请继续关注更多内容丰富的文章。