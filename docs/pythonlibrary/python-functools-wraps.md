# Python -如何使用 functools.wraps

> 原文：<https://www.blog.pythonlibrary.org/2016/02/17/python-functools-wraps/>

今天我想谈谈一个鲜为人知的工具。它叫做 **[包裹](https://docs.python.org/3.5/library/functools.html#functools.wraps)** ，是 **functools** 模块的一部分。您可以使用包装作为修饰器来修复文档字符串和被修饰函数的名称。为什么这很重要？起初，这听起来像是一个奇怪的边缘案例，但是如果你正在编写一个 API 或者其他人会使用的任何代码，那么这可能是很重要的。原因是当你使用 Python 的内省来理解别人的代码时，修饰过的函数会返回错误的信息。让我们来看一个简单的例子，我称之为**decrument . py**:

```py

# decorum.py

#----------------------------------------------------------------------
def another_function(func):
    """
    A function that accepts another function
    """

    def wrapper():
        """
        A wrapping function
        """
        val = "The result of %s is %s" % (func(),
                                          eval(func())
                                          )
        return val
    return wrapper

#----------------------------------------------------------------------
@another_function
def a_function():
    """A pretty useless function"""
    return "1+1"

#----------------------------------------------------------------------
if __name__ == "__main__":
    print(a_function.__name__)
    print(a_function.__doc__)

```

在这段代码中，我们用**的另一个 _ 函数**来修饰名为 **a_function** 的函数。您可以使用函数的 **__name__** 和 **__doc__** 属性打印出 **a_function 的**名称和 docstring。如果您运行这个示例，您将得到以下输出:

```py

wrapper

        A wrapping function

```

这是不对的！如果你在 IDLE 或者解释器中运行这个程序，它会变得更加明显，变得非常混乱，非常快。

```py

>>> import decorum
>>> help(decorum)
Help on module decorum:

NAME
    decorum - #----------------------------------------------------------------------

FILE
    /home/mike/decorum.py

FUNCTIONS
    a_function = wrapper()
        A wrapping function

    another_function(func)
        A function that accepts another function

>>> help(decorum.a_function)
Help on function other_func in module decorum:

wrapper()
    A wrapping function

```

基本上，这里发生的是装饰器将被装饰的函数的名称和 docstring 改为它自己的。

* * *

### 快来救援！

我们如何解决这个小问题？Python 开发者在 functools.wraps 中给了我们解决方案！让我们来看看:

```py

from functools import wraps

#----------------------------------------------------------------------
def another_function(func):
    """
    A function that accepts another function
    """

    @wraps(func)
    def wrapper():
        """
        A wrapping function
        """
        val = "The result of %s is %s" % (func(),
                                          eval(func())
                                          )
        return val
    return wrapper

#----------------------------------------------------------------------
@another_function
def a_function():
    """A pretty useless function"""
    return "1+1"

#----------------------------------------------------------------------
if __name__ == "__main__":
    #a_function()
    print(a_function.__name__)
    print(a_function.__doc__)

```

这里我们从 **functools** 模块导入**包装器**，并将其用作 **another_function** 内部嵌套包装器函数的装饰器。如果这次运行它，输出将会发生变化:

```py

a_function
A pretty useless function

```

现在我们又有了正确的名称和 docstring。如果你进入你的 Python 解释器，帮助功能现在也可以正常工作了。我将跳过把它的输出放在这里，留给你去尝试。

* * *

### 包扎

wraps decorator 很像一匹只会一招的小马，但是当你需要它的时候，它非常方便。如果您碰巧注意到您的函数没有给您正确的名称或 docstring，那么您现在知道如何非常容易地修复它了。祝您编码愉快！