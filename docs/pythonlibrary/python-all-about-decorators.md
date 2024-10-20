# Python:关于装饰者的一切

> 原文：<https://www.blog.pythonlibrary.org/2017/07/18/python-all-about-decorators/>

第一次遇到装饰者时，他们可能会有点困惑，调试起来也有点棘手。但是它们是向函数和类添加功能的一个好方法。装饰者也被称为“高阶函数”。这意味着它们可以接受一个或多个函数作为参数，并返回一个函数作为结果。换句话说，装饰者将接受他们正在装饰的函数并扩展它的行为，而实际上并不修改函数本身的功能。

自 2.2 版以来，Python 中有两个装饰器，即 **classmethod()** 和 **staticmethod()** 。然后将 [PEP 318](https://www.python.org/dev/peps/pep-0318/) 放在一起，并添加修饰语法，使 Python 2.4 中的修饰函数和方法成为可能。类装饰器在 [PEP 3129](https://www.python.org/dev/peps/pep-3129/) 中被提议包含在 Python 2.6 中。它们似乎在 Python 2.7 中工作，但是 PEP 指出它们直到 Python 3 才被接受，所以我不确定那里发生了什么。

让我们从讨论一般的函数开始，以便有一个工作的基础。

* * *

### 卑微的功能

Python 和许多其他编程语言中的函数只是可重用代码的集合。一些程序员会采用一种几乎类似 bash 的方法，将他们所有的代码写在一个文件中，根本没有函数。代码只是从上到下运行。这可能会导致大量的复制粘贴式代码。当你看到两段代码在做同样的事情时，它们几乎总是可以放入一个函数中。这将使更新你的代码更容易，因为你只有一个地方来更新它们。

这是一个基本功能:

```py

def doubler(number):
    return number * 2

```

这个函数接受一个参数，数字。然后将它乘以 2 并返回结果。您可以像这样调用该函数:

```py

>>> doubler(5)
10

```

如您所见，结果将是 10。

* * *

### 函数也是对象

在 Python 中，很多作者将函数描述为“一级对象”。当他们这样说的时候，他们的意思是一个函数可以被传递并作为其他函数的参数使用，就像你处理一个普通的数据类型一样，比如一个整数或字符串。让我们看几个例子，这样我们就可以习惯这个想法:

```py

>>> def doubler(number):
       return number * 2
>>> print(doubler)
 >>> print(doubler(10))
20
>>> doubler.__name__
'doubler'
>>> doubler.__doc__
None
>>> def doubler(number):
        """Doubles the number passed to it"""
        return number * 2 
>>> doubler.__doc__
'Doubles the number passed to it'
>>> dir(doubler)
['__call__', '__class__', '__closure__', '__code__', '__defaults__', '__delattr__', '__dict__', '__doc__', '__format__', '__get__', '__getattribute__', '__globals__', '__hash__', '__init__', '__module__', '__name__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', 'func_closure', 'func_code', 'func_defaults', 'func_dict', 'func_doc', 'func_globals', 'func_name'] 
```

正如您所看到的，您可以创建一个函数，然后将它传递给 Python 的 **print()** 函数或任何其他函数。您还会注意到，一旦定义了一个函数，它就会自动拥有我们可以访问的属性。例如，在上面的例子中，我们访问了最初为空的 **func_doc** 。该属性保存函数的 docstring 的内容。因为我们没有 docstring，所以它不返回任何内容。所以我们重新定义了函数来添加一个 docstring，并再次访问 func_doc 来查看 docstring。我们还可以通过 **func_name** 属性来获取函数的名称。请随意查看上面最后一个示例中显示的一些其他属性..

* * *

### 我们的第一个装潢师

创建一个装饰器实际上很容易。如前所述，创建装饰器所需要做的就是创建一个接受另一个函数作为参数的函数。让我们来看看:

```py

>>> def doubler(number):
        """Doubles the number passed to it"""
        return number * 2
>>> def info(func):
        def wrapper(*args):
            print('Function name: ' + func.__name__)
            print('Function docstring: ' + str(func.__doc__))
            return func(*args)
        return wrapper 
>>> my_decorator = info(doubler)
>>> print(my_decorator(2))
Function name: doubler
Function docstring: Doubles the number passed to it
4

```

您会注意到，在装饰函数 **info()** 中，嵌套了一个名为 **wrapper()** 的函数。您可以随意调用嵌套函数。包装器函数接受用装饰器包装的函数的参数(也可以是关键字参数)。在本例中，我们打印出包装函数的名称和 docstring，如果它存在的话。然后我们返回函数，用它的参数调用它。最后，我们返回包装函数。

为了使用装饰器，我们创建一个装饰器对象:

```py

>>> my_decorator = info(doubler)

```

然后为了调用装饰器，我们像调用普通函数一样调用它: **my_decorator(2)** 。

然而，这不是调用装饰器的常用方法。Python 有一个专门的语法！

* * *

### 使用装饰语法

Python 允许使用以下语法调用装饰器: **@info** 。让我们更新前面的例子，使用正确的修饰语法:

```py

def info(func):
    def wrapper(*args):
        print('Function name: ' + func.__name__)
        print('Function docstring: ' + str(func.__doc__))
        return func(*args)
    return wrapper

@info
def doubler(number):
    """Doubles the number passed to it"""
    return number * 2

print(doubler(4))

```

现在可以调用 **doubler()** 本身，而不是调用 decorator 对象。函数定义上面的 **@info** 告诉 Python 自动包装(或修饰)函数，并在调用函数时调用装饰器。

* * *

### 堆叠装饰者

你也可以堆叠或者链接装饰器。这意味着你可以在一个函数上同时使用多个装饰器！让我们来看一个愚蠢的例子:

```py

def bold(func):
    def wrapper():
        return "" + func() + ""
    return wrapper

def italic(func):
    def wrapper():
        return "*" + func() + "*"
    return wrapper

@bold
@italic
def formatted_text():
    return 'Python rocks!'

print(formatted_text())

```

**bold()** decorator 将使用标准的粗体 HTML 标签来包装文本，而 *italic()* decorator 做同样的事情，但是使用斜体 HTML 标签。你应该试着颠倒一下装饰者的顺序，看看会有什么样的效果。在继续之前尝试一下。

现在你已经完成了，你会注意到你的 Python 首先运行离函数最近的装饰器，然后沿着链向上。所以在上面的代码版本中，文本将首先用斜体显示，然后用粗体标记显示。如果你交换它们，就会发生相反的情况。

* * *

### 向装饰者添加参数

向 decorators 添加参数与您想象的有所不同。你不能只做类似于 **@my_decorator(3，' Python')** 的事情，因为 decorator 希望将函数本身作为它的参数...还是可以？

```py

def info(arg1, arg2):
    print('Decorator arg1 = ' + str(arg1))
    print('Decorator arg2 = ' + str(arg2))

    def the_real_decorator(function):

        def wrapper(*args, **kwargs):
            print('Function {} args: {} kwargs: {}'.format(
                function.__name__, str(args), str(kwargs)))
            return function(*args, **kwargs)

        return wrapper

    return the_real_decorator

@info(3, 'Python')
def doubler(number):
    return number * 2

print(doubler(5))

```

如你所见，我们有一个嵌套在函数中的函数！这是如何工作的？函数的参数似乎没有被定义。让我们移除装饰器，按照之前创建装饰器对象时的方式进行操作:

```py

def info(arg1, arg2):
    print('Decorator arg1 = ' + str(arg1))
    print('Decorator arg2 = ' + str(arg2))

    def the_real_decorator(function):

        def wrapper(*args, **kwargs):
            print('Function {} args: {} kwargs: {}'.format(
                function.__name__, str(args), str(kwargs)))
            return function(*args, **kwargs)

        return wrapper

    return the_real_decorator

def doubler(number):
    return number * 2

decorator = info(3, 'Python')(doubler)
print(decorator(5))

```

这段代码相当于前面的代码。当您调用 **info(3，' Python')** 时，它返回实际的装饰函数，然后我们通过向它传递函数 **doubler** 来调用它。这给了我们装饰对象本身，然后我们可以用原始函数的参数调用它。不过，我们可以进一步细分:

```py

def info(arg1, arg2):
    print('Decorator arg1 = ' + str(arg1))
    print('Decorator arg2 = ' + str(arg2))

    def the_real_decorator(function):

        def wrapper(*args, **kwargs):
            print('Function {} args: {} kwargs: {}'.format(
                function.__name__, str(args), str(kwargs)))
            return function(*args, **kwargs)

        return wrapper

    return the_real_decorator

def doubler(number):
    return number * 2

decorator_function = info(3, 'Python')
print(decorator_function)

actual_decorator = decorator_function(doubler)
print(actual_decorator)

# Call the decorated function
print(actual_decorator(5))

```

这里我们展示了我们首先获得装饰函数对象。然后我们得到 decorator 对象，它是 **info()** 中的第一个嵌套函数，即 **the_real_decorator()** 。这是您希望传递正在被修饰的函数的地方。现在我们有了修饰函数，所以最后一行是调用修饰函数。

我还发现了一个[巧妙的技巧](https://stackoverflow.com/a/25827070/393194),你可以用 Python 的 functools 模块来做，这将使创建带参数的装饰器变得更短:

```py

from functools import partial

def info(func, arg1, arg2):
    print('Decorator arg1 = ' + str(arg1))
    print('Decorator arg2 = ' + str(arg2))

    def wrapper(*args, **kwargs):
        print('Function {} args: {} kwargs: {}'.format(
            function.__name__, str(args), str(kwargs)))
        return function(*args, **kwargs)

    return wrapper

decorator_with_arguments = partial(info, arg1=3, arg2='Py')

@decorator_with_arguments
def doubler(number):
    return number * 2

print(doubler(5))

```

在这种情况下，您可以创建一个**分部**函数，它接受您要传递给装饰器的参数。这允许您将要修饰的函数和修饰器的参数传递给同一个函数。这实际上非常类似于如何使用 **functools.partial** 向 wxPython 或 Tkinter 中的事件处理程序传递额外的参数。

* * *

### 班级装饰者

当你查找术语“类装饰者”时，你会发现各种各样的文章。有些人谈论使用类来创建装饰者。其他人谈论用一个函数装饰一个类。让我们从创建一个可以用作装饰器的类开始:

```py

class decorator_with_arguments:

    def __init__(self, arg1, arg2):
        print('in __init__')
        self.arg1 = arg1
        self.arg2 = arg2
        print('Decorator args: {}, {}'.format(arg1, arg2))

    def __call__(self, f):
        print('in __call__')
        def wrapped(*args, **kwargs):
            print('in wrapped()')
            return f(*args, **kwargs)
        return wrapped

@decorator_with_arguments(3, 'Python')
def doubler(number):
    return number * 2

print(doubler(5))

```

这里我们有一个简单的类，它接受两个参数。我们覆盖了 **__call__()** 方法，该方法允许我们将正在装饰的函数传递给类。然后在我们的 __call__()方法中，我们只是打印出我们在代码中的位置并返回函数。这与上一节中的示例的工作方式非常相似。我个人喜欢这种方法，因为我们没有在另一个函数中嵌套两层的函数，尽管有些人可能会认为部分示例也解决了这个问题。

总之，你通常会发现的类装饰器的另一个用例是一种元编程。假设我们有下面的类:

```py

class MyActualClass:
    def __init__(self):
        print('in MyActualClass __init__()')

    def quad(self, value):
        return value * 4

obj = MyActualClass()
print(obj.quad(4))

```

这很简单，对吧？现在，假设我们想在不修改类已有功能的情况下向类中添加特殊功能。例如，这可能是由于向后兼容的原因或一些其他业务需求，我们不能更改的代码。相反，我们可以修饰它来扩展它的功能。下面是我们如何添加一个新方法，例如:

```py

def decorator(cls):
    class Wrapper(cls):
        def doubler(self, value):
            return value * 2
    return Wrapper

@decorator
class MyActualClass:
    def __init__(self):
        print('in MyActualClass __init__()')

    def quad(self, value):
        return value * 4

obj = MyActualClass()
print(obj.quad(4))
print(obj.doubler(5)

```

这里我们创建了一个装饰函数，它内部有一个类。这个类将使用传递给它的类作为它的父类。换句话说，我们正在创建一个子类。这允许我们添加新的方法。在这种情况下，我们添加 doubler()方法。现在，当您创建修饰的 **MyActualClass()** 类的实例时，您将实际上以**包装器()**子类版本结束。如果你打印 **obj** 变量，你实际上可以看到这一点。

* * *

### 包扎

Python 语言本身内置了很多修饰功能。有@property、@classproperty 和@staticmethod 可以直接使用。然后是 **functools** 和 **contextlib** 模块，它们提供了许多方便的装饰器。例如，您可以使用**functools . wrapps**修复装饰器混淆，或者通过**context lib . context manager**将任何函数设为上下文管理器。

许多开发人员使用 decorator 通过创建日志记录 decorator、捕捉异常、增加安全性等等来增强他们的代码。它们值得花时间去学习，因为它们可以让你的代码更具可扩展性，甚至更具可读性。装饰者也提倡代码重用。尽快给他们一个尝试的机会！

* * *

### 相关阅读

*   Python 征服宇宙- Python [装饰者](http://pythonconquerstheuniverse.wordpress.com/2012/04/29/python-decorators/)
*   真正的 Python-[Python 装饰者入门](https://realpython.com/blog/python/primer-on-python-decorators/)
*   StackOverflow - [链接装饰器](http://stackoverflow.com/questions/739654/how-can-i-make-a-chain-of-function-decorators-in-python)
*   Python 201: [装饰者](https://www.blog.pythonlibrary.org/2014/03/13/python-201-decorators/)
*   Python 如何使用 functools . wrapps
*   StackOverflow - [带参数的装饰器](https://stackoverflow.com/questions/5929107/decorators-with-parameters)
*   J.精致的[装饰者篇](http://jfine-python-classes.readthedocs.io/en/latest/decorators.html)
*   Python 3 成语- [装饰者](http://python-3-patterns-idioms-test.readthedocs.io/en/latest/PythonDecorators.html)