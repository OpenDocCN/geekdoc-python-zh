# 使用 Python locals()函数

> 原文：<https://www.askpython.com/python/built-in-methods/python-locals-function>

今天，我们将学习使用 Python `locals()`函数。这是另一个实用函数，对于调试程序非常有用。

**locals()函数给出了当前的局部符号表，作为字典。**

现在，如果您不确定局部符号表到底是什么，请继续阅读。让我们一步步来，从符号表的定义开始。

* * *

## 什么是符号表？

符号表是由关于不同符号的信息组成的表。在这里，符号可以表示任何东西——变量名、关键字、函数名等。

它们代表程序中所有变量、类和函数的名称。

通常，符号表不仅包括这些对象的名称，还包括其他有用的信息，如[对象的类型](https://www.askpython.com/python/oops/python-classes-objects)、[范围](https://www.askpython.com/python/python-namespace-variable-scope-resolution-legb)等。

现在你知道了符号表的意思，让我们来看看符号表的类别。

对于 Python 程序，有两种类型的符号表:

*   全局符号表->存储与程序的全局范围相关的信息
*   局部符号表->存储与程序的局部(当前)范围相关的信息

这是基于全局范围和局部(当前)范围定义的两个符号表。

当我们引用一个局部符号表时，当解释器逐行执行我们的代码时，我们引用我们的**当前**范围内的所有信息。

## Python locals()函数到底是做什么的？

现在，`locals()`函数所做的就是简单地将本地符号表信息粘贴到控制台上，粘贴到调用`locals()`的作用域上！

所以这自然意味着 Python `locals()`的输出将是所有变量名称和属性、范围等的字典。

例如，如果您有一个名为`main.py`的文件。让我们把`locals()`作为我们唯一的语句，看看会发生什么。我们应该在`main`范围内获得所有相关信息(在这种情况下，它与全局范围相同)

```py
# main.py
print(locals())

```

**可能的输出**

```py
{'__name__': '__main__', '__doc__': None, '__package__': None, '__loader__': <_frozen_importlib_external.SourceFileLoader object at 0x12ba85542>, '__spec__': None, '__annotations__': {}, '__builtins__': <module 'builtins' (built-in)>, '__file__': '/Users/askpython/home/locals_example.py', '__cached__': None}

```

嗯，我们可以看到`main`模块(全局范围)的一些属性，其中也包括一些包的细节！

正如你们中的一些人可能马上意识到的，这和这里的`globals()`是一样的，因为两者都指向同一个全局范围。

* * *

## 从局部范围调用局部变量()

现在，让我们考虑从一个函数调用局部范围内的`locals()`。

### 在函数内部调用局部变量()

让我们考虑一个简单的函数`fun(a, b)`，它有两个参数`a`和`b`，并返回总和。我们将在函数返回之前调用`locals()`。

```py
# Global variable
global_var = 1234

def fun(a, b):
    global global_var
    temp = 100
    print(f"locals() inside fun(a, b) = {locals()}")
    return a + b

if __name__ == '__main__':
    print(f"locals() outside fun(a, b) = {locals()}")
    print(fun(10, 20))

```

**输出**

```py
locals() outside fun(a, b) = {'__name__': '__main__', '__doc__': None, '__package__': None, '__loader__': <_frozen_importlib_external.SourceFileLoader object at 0x7f7023e1ff60>, '__spec__': None, '__annotations__': {}, '__builtins__': <module 'builtins' (built-in)>, '__file__': 'locals_example.py', '__cached__': None, 'global_var': 1234, 'fun': <function fun at 0x7f7023e5b1e0>}
locals() inside fun(a, b) = {'a': 10, 'b': 20, 'temp': 100}
30

```

这里，从`fun(a, b)`内部有一个明显的变化。这里，局部符号表仅由与该函数相关的名称组成。

因为局部作用域不是包的一部分，所以没有包信息，它只包含与函数相关的变量和参数。

还要注意，全局变量`global_var`是全局符号表的一部分，因此不存在于局部符号表中！

### 在类内部调用局部变量()

这类似于从函数中调用，但是这将包含所有的类方法和相关的属性。

让我们用一个例子快速看一下。

```py
class Student():
    def __init__(self, name):
        self.name = name
    def get_name(self):
        return self.name
    print(f"Calling locals() from inside a class => {locals()}")
    print(f"Calling globals() from inside a class => {globals()}")

if __name__ == '__main__':
    s = Student('Amit')
    print(s.get_name())

```

在这里，我们将在定义了所有的类方法之后在类内部调用`locals()`。所以这些类方法也必须是局部符号表的一部分。

**输出**

```py
Calling locals() from inside a class => {'__module__': '__main__', '__qualname__': 'Student', '__init__': <function Student.__init__ at 0x7fe2672f0c80>, 'get_name': <function Student.get_name at 0x7fe2672f0d08>}

Calling globals() from inside a class => {'__name__': '__main__', '__doc__': None, '__package__': None, '__loader__': <_frozen_importlib_external.SourceFileLoader object at 0x7fe2673cff28>, '__spec__': None, '__annotations__': {}, '__builtins__': <module 'builtins' (built-in)>, '__file__': 'locals_class.py', '__cached__': None}
Amit

```

的确，如你所见，`locals()`确实给出了相关方法！

当我们从类体内调用`locals()`时，我们将获得模块名、类名和类变量。

正如预期的那样，全局符号表中没有这种类型的内容。

* * *

## 结论

在本文中，我们学习了如何使用`locals()`函数从本地范围获取信息。这将从局部符号表中返回所有有用名称和属性的字典，对于调试非常有用。

## 参考

*   [Python locals()函数上的 StackOverflow 问题](https://stackoverflow.com/questions/40796264/what-does-pythons-locals-do)
*   [Python 官方文档](https://docs.python.org/3.8/library/functions.html#locals)关于 Python locals()
*   关于 Python locals()的 JournalDev 文章

* * *