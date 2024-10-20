# Python 101 -函数介绍

> 原文：<https://www.blog.pythonlibrary.org/2022/06/28/python-101-an-intro-to-functions/>

函数是可重用的代码片段。任何时候你发现自己写了两次相同的代码，这些代码应该放在一个函数中。

比如 Python 有很多内置函数，比如`dir()`和`sum()`。您还导入了`math`模块，并使用了它的平方根函数`sqrt()`。

在本教程中，您将了解:

*   创建函数
*   调用函数
*   传递参数
*   键入暗示你的论点
*   传递关键字参数
*   必需和默认参数
*   `*args`和`**kwargs`
*   仅位置参数
*   范围

我们开始吧！

## 创建函数

函数以关键字`def`开头，后跟函数名、两个括号，然后是一个冒号。接下来，在函数下缩进一行或多行代码，形成函数“块”。

下面是一个空函数:

```py
def my_function():
    pass
```

当你创建一个函数时，通常建议函数名全部小写，并用下划线隔开。这叫做**蛇案**。

`pass`是 Python 中的一个关键字，Python 知道可以忽略它。您也可以像这样定义一个空函数:

```py
def my_function():
    ...
```

在本例中，函数除了省略号之外没有任何内容。

接下来让我们学习如何使用一个函数！

## 调用函数

现在你有了一个函数，你需要让它做一些事情。让我们先这样做:

```py
def my_function():
    print('Hello from my_function')
```

现在不用省略号或关键字`pass`，而是用一个函数打印出一条消息。

要调用一个函数，你需要写出它的名字，后跟括号:

```py
>>> def my_function():
...     print('Hello from my_function')
... 
>>> my_function()
Hello from my_function
```

很好，很简单！

现在让我们来学习如何向函数传递参数。

## 传递参数

大多数函数允许你传递参数给它们。这样做的原因是，您通常希望向函数传递一个或多个位置参数，以便函数可以对它们做一些事情。

让我们创建一个函数，它接受一个名为`name`的参数，然后打印出一条欢迎消息:

```py
>>> def welcome(name):
...     print(f'Welcome {name}')
... 
>>> welcome('Mike')
Welcome Mike
```

如果您使用过其他编程语言，您可能知道其中一些语言需要函数来返回某些内容。如果不指定返回值，Python 会自动返回`None`。

让我们试着调用这个函数，并将其结果赋给一个名为`return_value`的变量:

```py
>>> def welcome(name):
...     print(f'Welcome {name}')
... 
>>> return_value = welcome('Mike')
Welcome Mike
>>> print(return_value)
None
```

当你打印出`return_value`的时候，你可以看到它是`None`。

## 键入暗示你的论点

有些编程语言使用静态类型，因此当您编译代码时，编译器会警告您与类型相关的错误。Python 是一种动态类型的语言，所以直到运行时才会发生这种情况

然而，在 Python 3.5 中，`typing`模块被添加到 Python 中，以允许开发人员将**类型提示**添加到他们的代码中。这允许你在代码中指定参数和返回值的类型，但是**没有**强制执行它。你可以使用外部工具，比如**mypy**([http://mypy-lang.org/](http://mypy-lang.org/))来检查你的代码库是否遵循了你设置的类型提示。

Python 中不需要类型提示，这种语言也没有强制要求，但是当与开发团队一起工作时，尤其是当团队由不熟悉 Python 的人组成时，类型提示非常有用。

让我们重写最后一个例子，使它使用类型提示:

```py
>>> def welcome(name: str) -> None:
...     print(f'Welcome {name}')
... 
>>> return_value = welcome('Mike')
Welcome Mike
>>> print(return_value)
None
```

这一次，当您放入`name`参数时，您用冒号(:)结尾，后跟您期望的类型。在这种情况下，您希望传入一个字符串类型。之后你会注意到代码的`-> None:`位。`->`是一种特殊的语法，用来表示预期的返回值。对于这个代码，返回值是`None`。

如果您想显式返回值，那么您可以使用`return`关键字，后跟您希望返回的内容。

当您运行代码时，它的执行方式与之前完全相同。

为了证明类型提示不是强制的，您可以使用`return`关键字告诉 Python 返回一个整数:

```py
>>> def welcome(name: str) -> None:
...     print(f'Welcome {name}')
...     return 5
... 
>>> return_value = welcome('Mike')
Welcome Mike
>>> print(return_value)
5
```

当您运行这段代码时，您可以看到类型提示说返回值应该是`None`，但是您对它进行了编码，使它返回整数`5`。Python 不会抛出异常。

您可以对这段代码使用 **mypy** 工具来验证它是否遵循了类型提示。如果你这样做了，你会发现这确实说明了一个问题。你将在本书的第二部分学习如何使用 **mypy** 。

这里的要点是 Python 支持类型提示。但是 Python 并不强制类型。然而，一些 Python 编辑器可以在内部使用类型提示来警告您与类型相关的问题，或者您可以手动使用 **mypy** 来查找问题。

现在让我们学习你还可以传递给函数什么。

## 传递关键字参数

Python 还允许你传入关键字参数。关键字参数是通过传入一个命名参数来指定的，例如您可以传入`age=10`。

让我们创建一个显示常规参数和单个关键字参数的新示例:

```py
>>> def welcome(name: str, age: int=15) -> None:
...     print(f'Welcome {name}. You are {age} years old.')
... 
>>> welcome('Mike')
Welcome Mike. You are 15 years old.
```

这个例子有一个常规参数`name`和一个关键字参数`age`，默认为`15`。当您在没有指定`age`的情况下调用这个代码时，您会看到它默认为 15。

为了让事情更加清楚，你可以用另一种方式来称呼它:

```py
>>> def welcome(name: str, age: int) -> None:
...     print(f'Welcome {name}. You are {age} years old.')
... 
>>> welcome(age=12, name='Mike')
Welcome Mike. You are 12 years old.
```

在这个例子中，您指定了`age`和`name`两个参数。当您这样做时，您可以以任何顺序指定它们。例如，这里您以相反的顺序指定了它们，Python 仍然理解您的意思，因为您指定了这两个值。

让我们看看不使用关键字参数时会发生什么:

```py
>>> def welcome(name: str, age: int) -> None:
...     print(f'Welcome {name}. You are {age} years old.')
... 
>>> welcome(12, 'Mike')
Welcome 12\. You are Mike years old.
```

当您在没有指定它们应该去哪里的情况下传递值时，它们将按顺序传递。于是`name`变成了`12`，`age`变成了`'Mike'`。

## 必需和默认参数

**默认参数**是一种简便的方法，可以用更少的参数调用你的函数，而**必需参数**是那些你必须传递给函数来执行函数的参数。

让我们看一个有一个必需参数和一个默认参数的例子:

```py
>>> def multiply(x: int, y: int=5) -> int:
...     return x * y
... 
>>> multiply(5)
25
```

第一个参数`x`是必需的。如果您不带任何参数调用`multiply()`，您将收到一个错误:

```py
>>> multiply()
Traceback (most recent call last):
  Python Shell, prompt 25, line 1
builtins.TypeError: multiply() missing 1 required positional argument: 'x'
```

第二个参数`y`，不是必需的。换句话说，它是一个默认参数，默认值为`5`。这允许你只用一个参数调用`multiply()`！

## 什么是`*args`和`**kwargs`？

大多数情况下，您会希望您的函数只接受少量的参数、关键字参数或两者都接受。你通常不希望有太多的参数，因为以后改变你的函数会变得更加复杂。

然而 Python 确实支持任意数量的参数或关键字参数的概念。

您可以在函数中使用以下特殊语法:

*   `*args` -任意数量的参数
*   `**kwargs` -任意数量的关键字参数

你需要注意的是`*`和`**`。名字，`arg`或者`kwarg`可以是任何东西，但是习惯上把它们命名为`args`和`kwargs`。换句话说，大多数 Python 开发者称它们为`*args`或`**kwargs`。虽然您没有被强制这样做，但您可能应该这样做，以便代码易于识别和理解。

让我们看一个例子:

```py
>>> def any_args(*args):
...     print(args)
... 
>>> any_args(1, 2, 3)
(1, 2, 3)
>>> any_args(1, 2, 'Mike', 4)
(1, 2, 'Mike', 4)
```

这里您创建了`any_args()`,它接受任意数量的参数(包括零)并打印出来。

您实际上可以创建一个函数，它有一个必需的参数和任意数量的附加参数:

```py
>>> def one_required_arg(required, *args):
...     print(f'{required=}')
...     print(args)
... 
>>> one_required_arg('Mike', 1, 2)
required='Mike'
(1, 2)
```

所以在这个例子中，函数的第一个参数是必需的。如果你在没有任何参数的情况下调用`one_required_arg()`，你将得到一个错误。

现在让我们尝试添加关键字参数:

```py
>>> def any_keyword_args(**kwargs):
...     print(kwargs)
... 
>>> any_keyword_args(1, 2, 3)
Traceback (most recent call last):
  Python Shell, prompt 7, line 1
builtins.TypeError: any_keyword_args() takes 0 positional arguments but 3 were given
```

哎呀！您创建了接受关键字参数的函数，但只传入普通参数。这导致抛出了一个`TypeError`。

让我们尝试传入与关键字参数相同的值:

```py
>>> def any_keyword_args(**kwargs):
...     print(kwargs)
... 
>>> any_keyword_args(one=1, two=2, three=3)
{'one': 1, 'two': 2, 'three': 3}
```

这一次，它以你期望的方式工作。

现在让我们检查一下我们的`*args`和`**kwargs`，看看它们是什么:

```py
>>> def arg_inspector(*args, **kwargs):
...     print(f'args are of type {type(args)}')
...     print(f'kwargs are of type {type(kwargs)}')
... 
>>> arg_inspector(1, 2, 3, x='test', y=5)
args are of type <class 'tuple'>
kwargs are of type <class 'dict'>
```

这意味着`args`是一个`tuple`，而`kwargs`是一个`dict`。

让我们看看是否可以为`*args`和`**kwargs`传递函数 a `tuple`和`dict`:

```py
>>> my_tuple = (1, 2, 3)
>>> my_dict = {'one': 1, 'two': 2}
>>> def output(*args, **kwargs):
...     print(f'{args=}')
...     print(f'{kwargs=}')
... 
>>> output(my_tuple)
args=((1, 2, 3),)
kwargs={}
>>> output(my_tuple, my_dict)
args=((1, 2, 3), {'one': 1, 'two': 2})
kwargs={}
```

好吧，那不太对劲。`tuple`和`dict`都在`*args`结束。不仅如此，`tuple`还停留在了一个元组，而不是变成了三个实参。

如果您使用特殊的语法，您可以做到这一点:

```py
>>> def output(*args, **kwargs):
...     print(f'{args=}')
...     print(f'{kwargs=}')
... 
>>> output(*my_tuple)
args=(1, 2, 3)
kwargs={}
>>> output(**my_dict)
args=()
kwargs={'one': 1, 'two': 2}
>>> output(*my_tuple, **my_dict)
args=(1, 2, 3)
kwargs={'one': 1, 'two': 2}
```

在这个例子中，你用`*my_tuple`调用`output()`。Python 将提取`tuple`中的单个值，并将它们作为参数传入。接下来传入`**my_dict`，它告诉 Python 将每个键/值对作为关键字参数传入。

最后一个例子同时传入了`tuple`和`dict`。

相当整洁！

## 仅位置参数

Python 3.8 为函数增加了一个新特性，称为**仅位置参数**。它们使用一种特殊的语法告诉 Python，一些参数必须是位置参数，一些参数必须是关键字参数。

让我们看一个例子:

```py
>>> def positional(name, age, /, a, b, *, key):
...     print(name, age, a, b, key)
... 
>>> positional(name='Mike')
Traceback (most recent call last):
  Python Shell, prompt 21, line 1
builtins.TypeError: positional() got some positional-only arguments passed as 
keyword arguments: 'name'
```

前两个参数`name`和`age`仅是位置性的。它们不能作为关键字参数传入，这就是你看到上面的`TypeError`的原因。参数，`a`和`b`可以是位置参数，也可以是关键字。最后，`key`，仅关键字。

正斜杠`/`向 Python 表明，正斜杠之前的所有参数都是仅限位置的参数。任何跟在正斜杠后面的都是位置或关键字参数，直到 th `*`。星号表示其后的所有内容都是仅关键字参数。

以下是调用该函数的有效方法:

```py
>>> positional('Mike', 17, 2, b=3, keyword='test')
Mike 17 2 3 test
```

但是，如果试图只传入位置参数，将会出现错误:

```py
>>> positional('Mike', 17, 2, 3, 'test')
Traceback (most recent call last):
  Python Shell, prompt 25, line 1
builtins.TypeError: positional() takes 4 positional arguments but 5 were given
```

`positional()`函数期望最后一个参数是关键字参数。

主要思想是，仅位置参数允许在不破坏客户端代码的情况下更改参数名。

您也可以对仅位置参数和`**kwargs`使用相同的名称:

```py
>>> def positional(name, age, /, **kwargs):
...     print(f'{name=}')
...     print(f'{age=}')
...     print(f'{kwargs=}')
... 
>>> positional('Mike', 17, name='Mack')
name='Mike'
age=17
kwargs={'name': 'Mack'}
```

您可以在此阅读语法背后的完整实现和推理:

*   [https://www.python.org/dev/peps/pep-0570](https://www.python.org/dev/peps/pep-0570)

让我们继续，了解一点关于作用域的话题！

## 范围

所有的编程语言都有范围的概念。作用域告诉编程语言哪些变量或函数对它们可用。

让我们看一个例子:

```py
>>> name = 'Mike'
>>> def welcome(name):
...     print(f'Welcome {name}')
... 
>>> welcome()
Traceback (most recent call last):
  Python Shell, prompt 34, line 1
builtins.TypeError: welcome() missing 1 required positional argument: 'name'
>>> welcome('Nick')
Welcome Nick
>>> name
'Mike'
```

变量`name`在`welcome()`函数之外定义。如果你试图调用`welcome()`而不给它传递一个参数，即使参数与变量`name`匹配，它也会抛出一个错误。如果你给`welcome()`传递一个值，这个变量只在`welcome()`函数内部被改变。您在函数外部定义的`name`保持不变。

让我们看一个在函数内部定义变量的例子:

```py
>>> def add():
...     a = 2
...     b = 4
...     return a + b
... 
>>> def subtract():
...     a = 3
...     return a - b
... 
>>> add()
6
>>> subtract()
Traceback (most recent call last):
  Python Shell, prompt 40, line 1
  Python Shell, prompt 38, line 3
builtins.NameError: name 'b' is not defined
```

在`add()`中，你定义`a`和`b`并将它们加在一起。变量`a`和`b`具有**局部**范围。这意味着它们只能在`add()`功能中使用。

在`subtract()`中，你只定义了`a`却试图使用`b`。Python 直到运行时才检查`subtract()`函数中是否存在`b`。

这意味着 Python 不会警告你这里缺少了什么，直到你真正调用了`subtract()`函数。这就是为什么直到最后你才看到任何错误。

Python 有一个特殊的`global`关键字，您可以使用它来允许跨函数使用变量。

让我们更新代码，看看它是如何工作的:

```py
>>> def add():
...     global b
...     a = 2
...     b = 4
...     return a + b
... 
>>> def subtract():
...     a = 3
...     return a - b
... 
>>> add()
6
>>> subtract()
-1
```

这次您在`add()`函数的开头将`b`定义为`global`。这允许你在`subtract()`中使用`b`，即使你没有在那里定义它。

通常不建议使用全局变量。在大型代码文件中很容易忽略它们，这使得跟踪细微的错误变得困难——例如，如果你在调用`add()`之前调用了`subtract()`,你仍然会得到错误，因为即使`b`是`global`,它也不存在，直到`add()`被运行。

在大多数情况下，如果你想使用`global`，你可以使用`class`来代替。

只要你明白你在做什么，使用全局变量没有错。他们有时会很有帮助。但是你应该小心使用它们。

## 包扎

函数是重用代码的一种非常有用的方式。它们可以被反复调用。函数也允许您传递和接收数据。

在本教程中，您学习了以下主题:

*   创建函数
*   调用函数
*   传递参数
*   键入暗示你的论点
*   传递关键字参数
*   必需和默认参数
*   `*args`和`**kwargs`
*   仅位置参数
*   范围

您可以使用函数来保持代码的整洁和有用。一个好的函数是独立的，可以被其他函数轻松使用。虽然本教程没有涉及到，但是您可以将函数嵌套在一起。

## 相关文章

*   Python 101 - [了解所有函数(视频)](https://www.blog.pythonlibrary.org/2022/05/07/python-101-learn-all-about-functions-video/)

*   Python 的[内置函数](https://www.blog.pythonlibrary.org/2021/02/17/an-intro-to-pythons-built-in-functions/)简介