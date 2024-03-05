# Python 作用域 LEGB 规则:解析代码中的名称

> 原文：<https://realpython.com/python-scope-legb-rule/>

**范围**的概念决定了如何在你的代码中查找[变量](https://realpython.com/python-variables/)和名字。它决定了代码中变量的可见性。名称或变量的范围取决于您在代码中创建该变量的位置。Python 作用域的概念通常使用一个称为 **LEGB 规则**的规则来表示。

首字母缩写词 LEGB 中的字母代表**局部、封闭、全局和内置**作用域。这不仅总结了 Python 的作用域级别，还总结了 Python 在程序中解析名称时遵循的步骤顺序。

在本教程中，您将学习:

*   什么是**范围**以及它们在 Python 中是如何工作的
*   为什么了解 **Python 作用域**很重要
*   什么是 **LEGB 规则**以及 Python 如何使用它来解析名称
*   如何使用`global`和`nonlocal`修改 Python 作用域的**标准行为**
*   Python 提供了哪些**范围相关的工具**以及如何使用它们

有了这些知识，您就可以利用 Python 作用域来编写更可靠、更易维护的程序。使用 Python 作用域将帮助您避免或最小化与名称冲突相关的错误，以及在您的程序中对全局名称的错误使用。

如果你熟悉 Python 的中级概念，比如[类](https://realpython.com/courses/intro-object-oriented-programming-oop-python/)、[函数](https://realpython.com/defining-your-own-python-function/)、[内部函数](https://realpython.com/inner-functions-what-are-they-good-for/#closures-and-factory-functions)、[变量](https://realpython.com/courses/variables-python/)、[异常](https://realpython.com/courses/introduction-python-exceptions/)、[综合](https://realpython.com/courses/using-list-comprehensions-effectively/)、[内置函数](https://realpython.com/lessons/operators-and-built-functions/)和标准的[数据结构](https://realpython.com/python-data-structures/)，你将从本教程中获益匪浅。

**免费奖励:** [掌握 Python 的 5 个想法](https://realpython.com/bonus/python-mastery-course/)，这是一个面向 Python 开发者的免费课程，向您展示将 Python 技能提升到下一个水平所需的路线图和心态。

## 理解范围

在编程中，一个名字的[作用域](https://realpython.com/courses/python-basics-scopes/)定义了一个程序的区域，在这个区域中你可以明确地访问这个名字，比如变量、函数、对象等等。名称仅对其范围内的代码可见和可访问。一些编程语言利用作用域来避免名称冲突和不可预知的行为。最常见的情况是，您将区分两个通用范围:

1.  **全局作用域:**你在这个作用域中定义的名字对你所有的代码都是可用的。

2.  **局部作用域:**您在这个作用域中定义的名字只对这个作用域中的代码可用或可见。

范围的出现是因为早期的编程语言(比如 BASIC)只有**全局名**。有了这样的名字，程序的任何部分都可以在任何时候修改任何变量，因此维护和调试大型程序可能会成为一场真正的噩梦。要使用全局名称，您需要同时记住所有代码，以便随时了解给定名称的值。这是没有示波器的一个重要副作用。

像 Python 这样的一些语言使用**范围**来避免这种问题。当你使用一种实现作用域的语言时，你没有办法在程序的所有位置访问程序中的所有变量。在这种情况下，您访问给定名称的能力将取决于您在哪里定义了该名称。

**注意:**您将使用术语 **name** 来指代变量、常量、函数、类或任何其他可以被赋予名称的对象的标识符。

程序中的名字将具有定义它们的代码块的范围。当您可以从代码中的某个地方访问给定名称的值时，您会说该名称是作用域中的**。如果你不能访问这个名字，那么你会说这个名字是**超出范围**。**

[*Remove ads*](/account/join/)

### Python 中的名称和作用域

因为 Python 是一种[动态类型的](https://wiki.python.org/moin/Why%20is%20Python%20a%20dynamic%20language%20and%20also%20a%20strongly%20typed%20language)语言，所以当你第一次给变量赋值时，Python 中的变量就存在了。另一方面，在您分别使用 [`def`](https://docs.python.org/3/reference/compound_stmts.html#def) 或 [`class`](https://docs.python.org/3/reference/compound_stmts.html#class-definitions) 定义函数和类之后，它们就可用了。最后，[模块](https://realpython.com/python-modules-packages/)在你导入后存在。总之，您可以通过以下操作之一创建 Python 名称:

| 操作 | 声明 |
| --- | --- |
| [分配任务](https://docs.python.org/3/reference/simple_stmts.html#assignment-statements) | `x = value` |
| [导入操作](https://realpython.com/courses/python-imports-101/) | `import module`或`from module import name` |
| 函数定义 | `def my_func(): ...` |
| 函数上下文中的参数定义 | `def my_func(arg1, arg2,... argN): ...` |
| 类别定义 | `class MyClass: ...` |

所有这些操作都会创建或在赋值的情况下更新新的 Python 名称，因为所有这些操作都会为变量、常量、函数、类、实例、模块或其他 Python 对象赋值。

**注:****赋值操作**和**引用或访问操作**有一个重要的区别。当您引用一个名称时，您只是检索它的内容或值。当您分配名称时，您要么创建该名称，要么修改它。

Python 使用名称分配或定义的位置将其与特定范围相关联。换句话说，您在代码中分配或定义名称的位置决定了该名称的范围或可见性。

例如，如果在函数内部给一个名字赋值，那么这个名字将有一个**本地 Python 作用域**。相比之下，如果你在所有函数之外给一个名字赋值——比如说，在一个模块的顶层——那么这个名字将会有一个**全局 Python 作用域**。

### Python 作用域与名称空间

在 Python 中，作用域的概念与[名称空间](https://realpython.com/python-namespaces-scope/)的概念密切相关。到目前为止，您已经了解到，Python 作用域决定了名字在程序中的可见位置。Python 作用域被实现为将名称映射到对象的[字典](https://realpython.com/python-dicts/)。这些[字典](https://realpython.com/courses/dictionaries-python/)俗称**名称空间**。这些是 Python 用来存储名称的具体机制。它们存储在一个名为 [`.__dict__`](https://docs.python.org/3/library/stdtypes.html#object.__dict__) 的特殊属性中。

模块顶层的名称存储在模块的命名空间中。换句话说，它们存储在模块的`.__dict__`属性中。看一下下面的代码:

>>>

```py
>>> import sys
>>> sys.__dict__.keys() dict_keys(['__name__', '__doc__', '__package__',..., 'argv', 'ps1', 'ps2'])
```

导入 [`sys`](https://docs.python.org/3/library/sys.html#module-sys) 后，可以使用`.keys()`来检查`sys.__dict__`的按键。这将返回一个[列表](https://realpython.com/courses/lists-tuples-python/)，其中包含模块顶层定义的所有名称。在这种情况下，你可以说`.__dict__`持有`sys`的名称空间，并且是模块范围的具体表示。

**注意:**为了节省空间，本教程中一些示例的输出被缩写为(`...`)。根据您的平台、Python 版本，甚至根据您使用当前 Python 交互式会话的时间长短，输出可能会有所不同。

作为进一步的例子，假设您需要使用名称`ps1`，它在`sys`中定义。如果您知道 Python 中的`.__dict__`和名称空间是如何工作的，那么您至少可以用两种不同的方式引用`ps1`:

1.  在`module.name`形式的模块名称上使用[点符号](https://docs.python.org/3/reference/expressions.html#attribute-references)
2.  在`module.__dict__['name']`表单中对`.__dict__`使用[订阅操作](https://docs.python.org/3/reference/expressions.html#subscriptions)

看一下下面的代码:

>>>

```py
>>> sys.ps1
'>>> '
>>> sys.__dict__['ps1']
'>>> '
```

一旦你导入了`sys`，你就可以使用`sys`上的点符号来访问`ps1`。您也可以使用关键字`'ps1'`通过字典关键字查找来访问`ps1`。这两个操作返回相同的结果`'>>> '`。

**注意:** [`ps1`](https://docs.python.org/3/library/sys.html#sys.ps1) 是一个[字符串](https://realpython.com/python-strings/)指定 Python 解释器的主要提示。`ps1`仅在解释器处于交互模式时定义，其初始值为`'>>> '`。

无论何时使用一个名称，比如变量或函数名，Python 都会搜索不同的作用域级别(或名称空间)来确定该名称是否存在。如果这个名字存在，那么你将总是得到它的第一个出现。否则，你会得到一个错误。您将在下一节中介绍这种搜索机制。

[*Remove ads*](/account/join/)

## 将 LEGB 规则用于 Python 范围

Python 使用所谓的 **LEGB 规则**来解析名称，该规则以 Python 的名称范围命名。LEGB 中的字母代表本地、封闭、全局和内置。以下是这些术语含义的简要概述:

*   **局部(或函数)作用域**是任意 Python 函数或 [`lambda`](https://realpython.com/python-lambda/) 表达式的代码块或代码体。这个 Python 范围包含您在函数中定义的名称。这些名称只能从函数的代码中看到。它是在函数调用时创建的，*而不是*在函数定义时创建的，所以你会有和函数调用一样多的不同局部作用域。即使多次调用同一个函数，或者递归调用，也是如此。每个调用都将导致创建一个新的本地范围。

*   **封闭(或非局部)作用域**是一个特殊的作用域，只存在于嵌套函数中。如果局部作用域是一个[内部或嵌套函数](https://realpython.com/inner-functions-what-are-they-good-for/)，那么封闭作用域就是外部或封闭函数的作用域。该作用域包含您在封闭函数中定义的名称。从内部函数和封闭函数的代码中可以看到封闭范围内的名称。

*   **全局(或模块)作用域**是 Python 程序、脚本或模块中最顶层的作用域。这个 Python 作用域包含您在程序或模块的顶层定义的所有名称。这个 Python 范围内的名称在代码中随处可见。

*   **内置作用域**是一个特殊的 Python 作用域，每当你[运行一个脚本](https://realpython.com/run-python-scripts/)或打开一个交互式会话时，它就会被创建或加载。这个范围包含诸如[关键字](https://realpython.com/python-keywords/)、函数、[异常](https://realpython.com/python-exceptions/)和其他内置于 Python 的属性。这个 Python 范围内的名称也可以在代码中的任何地方找到。当你运行一个程序或脚本时，Python 会自动加载它。

LEGB 规则是一种名称查找过程，它决定了 Python 查找名称的顺序。例如，如果您引用一个给定的名称，那么 Python 将在本地、封闭、全局和内置范围内依次查找该名称。如果该名称存在，那么您将获得它的第一次出现。否则，你会得到一个错误。

**注意:**注意，只有在函数(局部作用域)或嵌套或内部函数(局部和封闭作用域)中使用名称时，才会搜索局部和封闭 Python 作用域。

总之，当您使用嵌套函数时，通过首先检查局部范围或最内部函数的局部范围来解析名称。然后，Python 从最内部的范围到最外部的范围查看外部函数的所有封闭范围。如果没有找到匹配，那么 Python 会查看全局和内置范围。如果它找不到名字，那么你会得到一个错误。

在执行过程中的任何时候，根据您在代码中的位置，您最多有四个活动的 Python 作用域——本地、封闭、全局和内置。另一方面，您将始终拥有至少两个活动范围，它们是全局范围和内置范围。这两个范围将永远为您提供。

### 功能:局部范围

**局部作用域**或函数作用域是在函数调用时创建的 Python 作用域。每次你调用一个函数，你也在创建一个新的局部作用域。另一方面，您可以将每个`def`语句和`lambda`表达式视为新局部范围的蓝图。每当您调用手边的函数时，这些局部作用域就会出现。

默认情况下，在函数内部分配的参数和名称仅存在于与函数调用相关联的函数或局部范围内。当函数返回时，局部作用域被破坏，名字被遗忘。这是如何工作的:

>>>

```py
>>> def square(base):
...     result = base ** 2
...     print(f'The square of {base} is: {result}')
...
>>> square(10)
The square of 10 is: 100
>>> result  # Isn't accessible from outside square() Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
    result
NameError: name 'result' is not defined
>>> base  # Isn't accessible from outside square() Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
    base
NameError: name 'base' is not defined
>>> square(20)
The square of 20 is: 400
```

`square()`是计算给定数字`base`的平方的函数。当您调用该函数时，Python 会创建一个包含名称`base`(一个参数)和`result`(一个局部变量)的局部范围。在第一次调用`square()`之后，`base`保存值`10`，而`result`保存值`100`。第二次，本地名称将不会记得第一次调用函数时存储在其中的值。注意`base`现在保存值`20`，而`result`保存值`400`。

**注意:**如果你在函数调用后试图访问`result`或`base`，那么你会得到一个 [`NameError`](https://docs.python.org/3/library/exceptions.html#NameError) ，因为这些只存在于调用`square()`所创建的局部作用域中。每当你试图访问一个没有在任何 Python 作用域中定义的名字时，你都会得到一个`NameError`。错误消息将包含找不到的名称。

由于不能从函数外部的语句中访问局部名称，不同的函数可以用相同的名称定义对象。看看这个例子:

>>>

```py
>>> def cube(base):
...     result = base ** 3
...     print(f'The cube of {base} is: {result}')
...
>>> cube(30)
The cube of 30 is: 27000
```

注意，您使用在`square()`中使用的相同变量和参数来定义`cube()`。然而，由于`cube()`看不到`square()`的本地范围内的名字，反之亦然，两个函数都像预期的那样工作，没有任何名字冲突。

通过正确使用本地 Python 作用域，可以避免程序中的名称冲突。这也使得函数更加独立，并创建可维护的程序单元。此外，由于您不能从代码中的远程位置更改本地名称，因此您的程序将更容易调试、阅读和修改。

您可以使用`.__code__`检查函数的名称和参数，这是一个保存函数内部代码信息的属性。看看下面的代码:

>>>

```py
>>> square.__code__.co_varnames ('base', 'result')
>>> square.__code__.co_argcount
1
>>> square.__code__.co_consts
(None, 2, 'The square of ', ' is: ')
>>> square.__code__.co_name
'square'
```

在这个代码示例中，您在`square()`上检查`.__code__`。这是一个特殊的属性，用于保存 Python 函数代码的相关信息。在这种情况下，您会看到`.co_varnames`持有一个元组，其中包含您在`square()`中定义的名称。

[*Remove ads*](/account/join/)

### 嵌套函数:封闭范围

**当[将函数嵌套在其他函数](https://realpython.com/inner-functions-what-are-they-good-for/)中时，会观察到封闭或非局部作用域**。封闭范围是在 Python 2.2 中添加的[。它采用任何封闭函数的局部范围的形式。您在封闭 Python 作用域中定义的名称通常被称为**非本地名称**。考虑以下代码:](https://docs.python.org/3/whatsnew/2.2.html#pep-227-nested-scopes)

>>>

```py
>>> def outer_func():
...     # This block is the Local scope of outer_func() ...     var = 100  # A nonlocal var
...     # It's also the enclosing scope of inner_func() ...     def inner_func():
...         # This block is the Local scope of inner_func() ...         print(f"Printing var from inner_func(): {var}")
...
...     inner_func()
...     print(f"Printing var from outer_func(): {var}")
...
>>> outer_func()
Printing var from inner_func(): 100
Printing var from outer_func(): 100
>>> inner_func()
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'inner_func' is not defined
```

当你调用`outer_func()`时，你也在创建一个局部作用域。`outer_func()`的局部范围同时也是`inner_func()`的包围范围。从`inner_func()`内部看，这个作用域既不是全局作用域，也不是局部作用域。这是一个介于这两个作用域之间的特殊作用域，被称为**包围作用域**。

**注:**从某种意义上来说，`inner_func()`是一个临时函数，只有在它的封闭函数`outer_func()`执行的时候才会有生命。注意`inner_func()`仅对`outer_func()`中的代码可见。

您在封闭范围内创建的所有名字在`inner_func()`内部都是可见的，除了那些在您调用`inner_func()`之后创建的名字。这里有一个新版本的`outer_fun()`表明了这一点:

>>>

```py
>>> def outer_func():
...     var = 100
...     def inner_func():
...         print(f"Printing var from inner_func(): {var}")
...         print(f"Printing another_var from inner_func(): {another_var}")
...
...     inner_func()
...     another_var = 200  # This is defined after calling inner_func() ...     print(f"Printing var from outer_func(): {var}")
...
>>> outer_func()
Printing var from inner_func(): 100
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
    outer_func()
  File "<stdin>", line 7, in outer_func
    inner_func()
  File "<stdin>", line 5, in inner_func
    print(f"Printing another_var from inner_func(): {another_var}")
NameError: free variable 'another_var' referenced before assignment in enclosing
 scope
```

当你调用`outer_func()`时，代码运行到你调用`inner_func()`的地方。`inner_func()`的最后一条语句试图访问`another_var`。此时，`another_var`还没有定义，所以 Python 抛出了一个`NameError`，因为它找不到您试图使用的名称。

最后但同样重要的是，您不能从嵌套函数内部修改封闭作用域中的名称，除非您在嵌套函数中将它们声明为`nonlocal`。在本教程中，你将在后面的中讲述如何使用`nonlocal` [。](https://realpython.com/python-scope-legb-rule/#the-nonlocal-statement)

### 模块:全局范围

从你开始一个 Python 程序的那一刻起，你就在全局 Python 的范围内。在内部，Python 将程序的主脚本转换成一个名为 [`__main__`](https://docs.python.org/3/library/__main__.html#module-__main__) 的模块来保存主程序的执行。这个模块的名称空间是你的程序的主**全局范围**。

**注意:**在 Python 中，全局范围和全局名称的概念与模块文件紧密相关。例如，如果您在任何 Python 模块的顶层定义了一个名称，那么该名称就被认为是该模块的全局名称。这就是为什么这种作用域也被称为**模块作用域**的原因。

如果您正在 Python 交互式会话中工作，那么您会注意到`'__main__'`也是其主模块的名称。要检查这一点，请打开一个交互式会话并键入以下内容:

>>>

```py
>>> __name__
'__main__'
```

每当你运行一个 Python 程序或者一个交互式会话，比如上面的代码，解释器就执行模块或者脚本中的代码，作为你程序的入口点。这个模块或脚本以特殊名称`__main__`加载。从这一点开始，你可以说你的主要全局作用域是`__main__`的作用域。

要查看主全局范围内的名称，可以使用 [`dir()`](https://realpython.com/python-scope-legb-rule/#dir) 。如果您不带参数调用`dir()`，那么您将获得当前全局范围内的名字列表。看一下这段代码:

>>>

```py
>>> dir()
['__annotations__', '__builtins__',..., '__package__', '__spec__']
>>> var = 100  # Assign var at the top level of __main__ >>> dir()
['__annotations__', '__builtins__',..., '__package__', '__spec__', 'var']
```

当您不带参数调用`dir()`时，您将获得在您的主全局 Python 作用域中可用的名称列表。注意，如果你在模块(这里是`__main__`)的顶层指定一个新名字(比如这里的`var`，那么这个名字将被添加到`dir()`返回的列表中。

**注意:**你将在本教程的部分更详细地讲述`dir()`[。](https://realpython.com/python-scope-legb-rule/#dir)

每个程序执行只有一个全局 Python 作用域。这个作用域一直存在，直到程序终止，所有的名字都被遗忘。否则，下一次运行程序时，这些名称会记住上次运行时的值。

您可以从代码中的任何位置访问或引用任何全局名称的值。这包括函数和类。这里有一个例子来阐明这些观点:

>>>

```py
>>> var = 100
>>> def func():
...     return var  # You can access var from inside func() ...
>>> func()
100
>>> var  # Remains unchanged 100
```

在`func()`内部，可以自由访问或引用`var`的值。这对你的全局名`var`没有影响，但是它告诉你`var`可以在`func()`内自由访问。另一方面，除非使用 [`global`语句](https://realpython.com/python-scope-legb-rule/#the-global-statement)将函数显式声明为全局名称，否则不能在函数内部分配全局名称，稍后您将会看到这一点。

每当在 Python 中为名称赋值时，会发生以下两种情况之一:

1.  你**给**起了一个新名字
2.  您**更新了**一个现有的名称

具体行为将取决于您在其中分配名称的 Python 范围。如果你试图在一个函数中给一个全局名字赋值，那么你将会在函数的局部作用域中创建这个名字，隐藏或者覆盖这个全局名字。这意味着你不能从函数内部改变大多数在函数外部定义的变量。

如果您遵循这一逻辑，那么您将意识到下面的代码不会像您预期的那样工作:

>>>

```py
>>> var = 100  # A global variable >>> def increment():
...     var = var + 1  # Try to update a global variable ...
>>> increment()
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
    increment()
  File "<stdin>", line 2, in increment
    var = var + 1
UnboundLocalError: local variable 'var' referenced before assignment
```

在`increment()`中，您尝试增加全局变量`var`。由于`var`没有在`increment()`中声明`global`，Python 在函数中创建了一个同名的新局部变量`var`。在这个过程中，Python 意识到您试图在第一次赋值(`var + 1`)之前使用局部`var`，所以它引发了一个`UnboundLocalError`。

这是另一个例子:

>>>

```py
>>> var = 100  # A global variable >>> def func():
...     print(var)  # Reference the global variable, var ...     var = 200   # Define a new local variable using the same name, var ...
>>> func()
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
    func()
  File "<stdin>", line 2, in func
    print(var)
UnboundLocalError: local variable 'var' referenced before assignment
```

您可能希望能够打印全局`var`并能够稍后更新`var`，但是您再次得到了一个`UnboundLocalError`。这里发生的事情是，当你运行`func()`的主体时，Python 决定`var`是一个局部变量，因为它是在函数范围内赋值的。这不是一个 bug，而是一个设计选择。Python 假设在函数体中分配的名称对于该函数是局部的。

**注意:**全局名称可以在您的全局 Python 范围内的任何地方更新或修改。除此之外，`global`语句可以用来从代码中的几乎任何地方修改全局名称，正如你将在[的`global`语句](https://realpython.com/python-scope-legb-rule/#the-global-statement)中看到的。

修改全局名称通常被认为是糟糕的编程实践，因为它会导致代码:

*   **难调试:**程序中几乎任何语句都可以改变一个全局名的值。
*   **难以理解:**你需要注意所有访问和修改全局名的语句。
*   无法重用:代码依赖于特定于具体程序的全局名称。

良好的编程实践建议使用本地名称而不是全局名称。以下是一些建议:

*   **写**依赖局部名字而不是全局名字的自包含函数。
*   **尝试**使用唯一的对象名，不管你在什么范围。
*   在你的程序中避免修改全局名字。
*   **避免**跨模块修改名称。
*   **使用**全局名作为[常量](https://realpython.com/python-constants/)，它们在程序执行过程中不会改变。

到目前为止，您已经讨论了三个 Python 范围。下面的例子总结了它们在代码中的位置，以及 Python 如何通过它们查找名称:

>>>

```py
>>> # This area is the global or module scope >>> number = 100
>>> def outer_func():
...     # This block is the local scope of outer_func() ...     # It's also the enclosing scope of inner_func() ...     def inner_func():
...         # This block is the local scope of inner_func() ...         print(number)
...
...     inner_func()
...
>>> outer_func()
100
```

当您调用`outer_func()`时，您会在屏幕上看到`100`。但是在这种情况下 Python 如何查找名字`number`？遵循 LEGB 规则，您将在以下位置查找`number`:

1.  **在`inner_func()`里面:**这是本地范围，但是`number`在那里不存在。
2.  `outer_func()`里面的**:**这是封闭的范围，但是`number`也没有在这里定义。
3.  **在模块作用域:**这是全局作用域，你在那里找到`number`，就可以把`number`打印到屏幕上。

如果`number`没有在全局范围内定义，那么 Python 通过查看内置范围继续搜索。这是 LEGB 规则的最后一个组成部分，您将在下一节看到。

[*Remove ads*](/account/join/)

### `builtins`:内置范围

**内置作用域**是一个特殊的 Python 作用域，在 Python 3.x 中被实现为一个名为 [`builtins`](https://docs.python.org/3/library/builtins.html#module-builtins) 的标准库模块，Python 的所有内置对象都在这个模块中。当您运行 Python 解释器时，它们会自动加载到内置范围。Python 在其 LEGB 查找中最后搜索`builtins`,因此您可以免费获得它定义的所有名称。这意味着您可以在不导入任何模块的情况下使用它们。

请注意，`builtins`中的名称总是以特殊名称`__builtins__`加载到您的全局 Python 范围中，如您在以下代码中所见:

>>>

```py
>>> dir()
['__annotations__', '__builtins__',..., '__package__', '__spec__']
>>> dir(__builtins__) ['ArithmeticError', 'AssertionError',..., 'tuple', 'type', 'vars', 'zip']
```

在第一次调用`dir()`的输出中，您可以看到`__builtins__`总是出现在全局 Python 范围内。如果您使用`dir()`检查`__builtins__`本身，那么您将获得 Python 内置名称的完整列表。

内置范围为您当前的全局 Python 范围带来了 150 多个名称。例如，在 [Python 3.8](https://realpython.com/courses/cool-new-features-python-38/) 中，您可以知道名称的确切数量，如下所示:

>>>

```py
>>> len(dir(__builtins__))
152
```

通过调用 [`len()`](https://realpython.com/len-python-function/) ，可以得到`dir()`返回的`list`中的物品数量。这将返回 152 个名称，包括异常、函数、类型、特殊属性和其他 Python 内置对象。

即使您可以免费访问所有这些 Python 内置对象(无需导入任何内容)，您也可以显式导入`builtins`并使用点符号访问名称。这是如何工作的:

>>>

```py
>>> import builtins  # Import builtins as a regular module >>> dir(builtins)
['ArithmeticError', 'AssertionError',..., 'tuple', 'type', 'vars', 'zip']
>>> builtins.sum([1, 2, 3, 4, 5])
15
>>> builtins.max([1, 5, 8, 7, 3])
8
>>> builtins.sorted([1, 5, 8, 7, 3])
[1, 3, 5, 7, 8]
>>> builtins.pow(10, 2)
100
```

您可以像导入任何其他 Python 模块一样导入`builtins`。从这一点开始，您可以通过使用带点的属性查找或[全限定名称](https://docs.python.org/3/glossary.html#term-qualified-name)来访问`builtins`中的所有名称。如果您希望确保在任何全局名称覆盖任何内置名称时不会发生名称冲突，这将非常有用。

您可以在全局范围内覆盖或重新定义任何内置名称。如果您这样做，那么请记住，这将影响您的所有代码。看一下下面的例子:

>>>

```py
>>> abs(-15)  # Standard use of a built-in function 15
>>> abs = 20  # Redefine a built-in name in the global scope >>> abs(-15)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: 'int' object is not callable
```

如果你覆盖或者重新赋值`abs`，那么原来内置的 [`abs()`](https://realpython.com/python-absolute-value/#using-the-built-in-abs-function-with-numbers) 会影响你所有的代码。现在，假设你需要调用原来的`abs()`，而你忘记了你已经重新分配了名字。在这种情况下，当您再次调用`abs()`时，您会得到一个`TypeError`，因为`abs`现在保存了一个对整数的引用，这是不可调用的。

**注意:**在你的全局作用域中偶然或不经意地覆盖或重定义内置名称可能是危险且难以发现的 bug 的来源。最好尽量避免这种做法。

如果您正在试验一些代码，并且意外地在交互提示符下重新分配了一个内置名称，那么您可以重新启动您的会话或者运行`del name`来从您的全局 Python 作用域中移除重新定义。这样，您就在内置范围内恢复了原来的名称。如果你重温一下`abs()`的例子，那么你可以这样做:

>>>

```py
>>> del abs  # Remove the redefined abs from your global scope >>> abs(-15)  # Restore the original abs() 15
```

当您删除自定义`abs`名称时，您将从全局范围中删除该名称。这允许您再次访问内置作用域中的原始`abs()`。

要解决这种情况，您可以显式导入`builtins`，然后使用完全限定的名称，如以下代码片段所示:

>>>

```py
>>> import builtins
>>> builtins.abs(-15)
15
```

一旦显式导入了`builtins`，就可以在全局 Python 范围内使用模块名称。从这一点开始，您可以使用完全限定的名称从`builtins`中明确地获得您需要的名称，就像您在上面的例子中对`builtins.abs()`所做的那样。

作为快速总结，下表显示了 Python 范围的一些含义:

| 行动 | 全球代码 | 本地代码 | 嵌套功能代码 |
| --- | --- | --- | --- |
| 访问或引用全局范围内的名称 | 是 | 是 | 是 |
| 修改或更新全局范围内的名称 | 是 | 否(除非声明为`global`) | 否(除非声明为`global`) |
| 访问或引用本地范围内的名称 | 不 | 是(它自己的本地范围)，否(其他本地范围) | 是(它自己的本地范围)，否(其他本地范围) |
| 覆盖内置范围中的名称 | 是 | 是(在功能执行期间) | 是(在功能执行期间) |
| 访问或引用位于其封闭范围内的名称 | 不适用的 | 不适用的 | 是 |
| 修改或更新位于其封闭范围内的名称 | 不适用的 | 不适用的 | 否(除非声明为`nonlocal`) |

此外，不同范围内的代码可以对不同的对象使用相同的名称。这样，您可以使用一个名为`spam`的局部变量和一个同名的全局变量`spam`。然而，这被认为是糟糕的编程实践。

[*Remove ads*](/account/join/)

## 修改 Python 作用域的行为

到目前为止，您已经了解了 Python 作用域是如何工作的，以及它们如何将变量、函数、类和其他 Python 对象的可见性限制在代码的特定部分。您现在知道可以从代码中的任何地方访问或引用**全局名称**，但是可以在全局 Python 范围内修改或更新它们。

您还知道，您只能从创建它们的本地 Python 作用域内部或从嵌套函数内部访问**本地名称**，但是您不能从全局 Python 作用域或其他本地作用域访问它们。此外，您已经了解到**非本地名称**可以从嵌套函数内部访问，但是不能从那里修改或更新。

尽管 Python 作用域在默认情况下遵循这些通用规则，但是有一些方法可以修改这种标准行为。Python 提供了两个关键字，允许您修改全局和非本地名称的内容。这两个关键字是:

1.  [**`global`**T4】](https://docs.python.org/3/reference/simple_stmts.html#global)
2.  [**`nonlocal`**T4】](https://docs.python.org/3/reference/simple_stmts.html#nonlocal)

在接下来的两节中，您将介绍如何使用这些 Python 关键字来修改 Python 范围的标准行为。

### `global`语句

您已经知道，当您试图在函数内部给全局名称赋值时，您会在函数范围内创建一个新的局部名称。要修改这个行为，你可以使用一个 **`global`语句**。使用这个语句，您可以定义一个将被视为全局名称的名称列表。

该语句由关键字`global`组成，后跟一个或多个用逗号分隔的名称。您还可以在一个名称(或一个名称列表)中使用多个`global`语句。您在`global`语句中列出的所有名字都将被映射到您定义它们的全局或模块范围。

这里有一个例子，你试图从一个函数中更新一个全局变量:

>>>

```py
>>> counter = 0  # A global name
>>> def update_counter():
...     counter = counter + 1  # Fail trying to update counter ...
>>> update_counter()
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "<stdin>", line 2, in update_counter
UnboundLocalError: local variable 'counter' referenced before assignment
```

当你试图在`update_counter()`中赋值`counter`时，Python 假设`counter`是`update_counter()`的本地变量，并引发一个`UnboundLocalError`，因为你试图访问一个还没有定义的名字。

如果您希望这段代码以您期望的方式工作，那么您可以使用如下的`global`语句:

>>>

```py
>>> counter = 0  # A global name
>>> def update_counter():
...     global counter  # Declare counter as global ...     counter = counter + 1  # Successfully update the counter ...
>>> update_counter()
>>> counter
1
>>> update_counter()
>>> counter
2
>>> update_counter()
>>> counter
3
```

在这个新版本的`update_counter()`中，在试图改变`counter`之前，您将语句`global counter`添加到函数体中。通过这个微小的改变，您将函数作用域中的名称`counter`映射到全局或模块作用域中的相同名称。从这一点开始，你可以在`update_counter()`里面自由修改`counter`。所有的改变都会反映在全局变量中。

使用语句`global counter`，您告诉 Python 在全局范围内查找名称`counter`。这样，表达式`counter = counter + 1`不会在函数范围内创建一个新名字，而是在全局范围内更新它。

**注意:**使用`global`通常被认为是不好的做法。如果你发现自己在使用`global`来解决上述问题，那么停下来想想是否有更好的方法来编写你的代码。

例如，您可以尝试编写一个依赖于本地名称而不是全局名称的自包含函数，如下所示:

>>>

```py
>>> global_counter = 0  # A global name >>> def update_counter(counter):
...     return counter + 1  # Rely on a local name ...
>>> global_counter = update_counter(global_counter)
>>> global_counter
1
>>> global_counter = update_counter(global_counter)
>>> global_counter
2
>>> global_counter = update_counter(global_counter)
>>> global_counter
3
```

这个`update_counter()`的实现将`counter`定义为一个参数，并在每次调用函数时返回增加了`1`单位的值。这样，`update_counter()`的结果取决于您用作输入的`counter`，而不是其他函数(或代码段)可以对全局变量`global_counter`执行的更改。

还可以使用`global`语句通过在函数中声明来创建惰性全局名。看一下下面的代码:

>>>

```py
>>> def create_lazy_name():
...     global lazy  # Create a global name, lazy ...     lazy = 100
...     return lazy
...
>>> create_lazy_name()
100
>>> lazy  # The name is now available in the global scope 100
>>> dir()
['__annotations__', '__builtins__',..., 'create_lazy_name', 'lazy']
```

当您调用`create_lazy_name()`时，您也创建了一个名为`lazy`的全局变量。注意，在调用函数之后，名字`lazy`在全局 Python 范围内是可用的。如果您使用`dir()`检查全局名称空间，那么您会看到`lazy`出现在列表的最后。

**注意:**尽管您可以使用`global`语句来创建懒惰的全局名称，但这可能是一种危险的做法，会导致错误的代码。所以，最好在你的代码中避免这样的事情。

例如，假设您试图访问其中一个惰性名称，由于某种原因，您的代码还没有调用创建该名称的函数。在这种情况下，你会得到一个`NameError`并且你的程序会崩溃。

最后，值得注意的是，您可以在任何函数或嵌套函数内部使用`global`,所列出的名称将总是映射到全局 Python 范围内的名称。

还要注意，尽管在模块的顶层使用`global`语句是合法的，但这并没有多大意义，因为在全局作用域中分配的任何名称根据定义都已经是全局名称了。看一下下面的代码:

>>>

```py
>>> name = 100
>>> dir()
['__annotations__', '__builtins__',..., '__spec__', 'name']
>>> global name
>>> dir()
['__annotations__', '__builtins__',..., '__spec__', 'name']
```

像`global name`这样的`global`语句的使用不会改变你当前的全局范围，正如你在`dir()`的输出中看到的。无论你是否使用`global`，变量`name`都是一个全局变量。

[*Remove ads*](/account/join/)

### `nonlocal`语句

与全局名称类似，非局部名称可以从内部函数访问，但不能赋值或更新。如果你想修改它们，那么你需要使用一个 **`nonlocal`语句**。使用`nonlocal`语句，您可以定义一个将被视为非本地的名称列表。

`nonlocal`语句由`nonlocal`关键字组成，后跟一个或多个用逗号分隔的名称。这些名称将引用封闭 Python 范围中的相同名称。下面的例子展示了如何使用`nonlocal`来修改在封闭或非本地作用域中定义的变量:

>>>

```py
>>> def func():
...     var = 100  # A nonlocal variable ...     def nested():
...         nonlocal var  # Declare var as nonlocal ...         var += 100
...
...     nested()
...     print(var)
...
>>> func()
200
```

通过语句`nonlocal var`，你告诉 Python 你将在`nested()`中修改`var`。然后，使用一个[增加的赋值操作](https://docs.python.org/3/reference/simple_stmts.html#augmented-assignment-statements)来增加`var`。这种变化反映在非本地名称`var`中，该名称现在的值为`200`。

与`global`不同，您不能在嵌套或封闭函数之外使用`nonlocal`。更准确地说，你不能在全局范围或局部范围内使用`nonlocal`语句。这里有一个例子:

>>>

```py
>>> nonlocal my_var  # Try to use nonlocal in the global scope
  File "<stdin>", line 1
SyntaxError: nonlocal declaration not allowed at module level
>>> def func():
...     nonlocal var  # Try to use nonlocal in a local scope ...     print(var)
...
  File "<stdin>", line 2
SyntaxError: no binding for nonlocal 'var' found
```

这里，您首先尝试在全局 Python 范围内使用一个`nonlocal`语句。由于`nonlocal`只在内部或嵌套函数中起作用，你得到一个 [`SyntaxError`](https://realpython.com/invalid-syntax-python/) 告诉你不能在模块范围内使用`nonlocal`。注意`nonlocal`也不能在局部范围内工作。

**注意:**要了解关于`nonlocal`声明的更多详细信息，请查看[PEP 3104——访问外部作用域中的名称](http://www.python.org/dev/peps/pep-3104/)。

与`global`相反，你不能使用`nonlocal`来创建懒惰的非本地名字。如果要将名称用作非本地名称，则名称必须已经存在于封闭 Python 范围中。这意味着你不能通过在嵌套函数的`nonlocal`语句中声明来创建非本地名字。看一下下面的代码示例:

>>>

```py
>>> def func():
...     def nested():
...         nonlocal lazy_var  # Try to create a nonlocal lazy name ...
  File "<stdin>", line 3
SyntaxError: no binding for nonlocal 'lazy_var' found
```

在这个例子中，当您试图使用`nonlocal lazy_var`定义一个非本地名称时，Python 会立即抛出一个`SyntaxError`，因为`lazy_var`不存在于`nested()`的封闭范围内。

## 使用封闭作用域作为闭包

**闭包**是封闭 Python 范围的特殊用例。当您将嵌套函数作为数据处理时，组成该函数的语句与它们执行的环境打包在一起。由此产生的对象称为闭包。换句话说，[闭包](https://realpython.com/inner-functions-what-are-they-good-for/#closures-and-factory-functions)是一个内部或嵌套的函数，它携带关于其封闭范围的信息，即使这个范围已经完成了它的执行。

**注意:**你也可以将这种函数称为**工厂**、**工厂函数**，或者更准确地说，称为**闭包工厂**，以指定该函数构建并返回闭包(内部函数)，而不是类或实例。

闭包提供了一种在函数调用之间保留状态信息的方法。当您想要基于[懒惰或延迟评估](https://en.wikipedia.org/wiki/Lazy_evaluation)的概念编写代码时，这可能是有用的。请看下面的代码，这是一个闭包如何工作以及如何在 Python 中利用它们的例子:

>>>

```py
>>> def power_factory(exp):
...     def power(base):
...         return base ** exp
...     return power
...
>>> square = power_factory(2)
>>> square(10)
100
>>> cube = power_factory(3)
>>> cube(10)
1000
>>> cube(5)
125
>>> square(15)
225
```

您的闭包工厂函数`power_factory()`接受一个名为`exp`的参数。您可以使用这个函数来构建运行不同 power 操作的闭包。这是可行的，因为每个对`power_factory()`的调用都获得自己的一组状态信息。换句话说，它为`exp`获取它的值。

**注:**类似`exp`的变量称为**自由变量**。它们是在代码块中使用但没有定义的变量。自由变量是闭包用来在调用之间保留状态信息的机制。

在上面的例子中，内部函数`power()`首先被赋值给`square`。在这种情况下，该函数会记住`exp`等于`2`。在第二个例子中，您使用`3`作为参数来调用`power_factory()`。这样，`cube`持有一个函数对象，它记住了`exp`就是`3`。注意，您可以自由地重用`square`和`cube`，因为它们不会忘记各自的状态信息。

关于如何使用闭包的最后一个例子，假设您需要计算一些样本数据的平均值。您通过对正在分析的参数的一系列连续测量来收集数据。在这种情况下，您可以使用闭包工厂来生成一个闭包，该闭包会记住样本中以前的度量。看一下下面的代码:

>>>

```py
>>> def mean():
...     sample = []
...     def _mean(number):
...         sample.append(number)
...         return sum(sample) / len(sample)
...     return _mean
...
>>> current_mean = mean()
>>> current_mean(10)
10.0
>>> current_mean(15)
12.5
>>> current_mean(12)
12.333333333333334
>>> current_mean(11)
12.0
>>> current_mean(13)
12.2
```

您在上面的代码中创建的闭包会在调用`current_mean`之间记住`sample`的状态信息。这样，你就可以用优雅而[蟒](https://realpython.com/learning-paths/writing-pythonic-code/)的方式解决问题。

**注意:**如果你想学习更多关于作用域和闭包的知识，那么可以看看[探索 Python 中的作用域和闭包](https://realpython.com/courses/exploring-scopes-and-closures-in-python/)视频课程。

请注意，如果您的数据流变得太大，那么这个函数在内存使用方面会成为一个问题。这是因为每次调用`current_mean`，`sample`都会保存越来越大的值列表。使用`nonlocal`查看下面的替代实现代码:

>>>

```py
>>> def mean():
...     total = 0
...     length = 0
...     def _mean(number):
...         nonlocal total, length
...         total += number
...         length += 1
...         return total / length
...     return _mean
...
>>> current_mean = mean()
>>> current_mean(10)
10.0
>>> current_mean(15)
12.5
>>> current_mean(12)
12.333333333333334
>>> current_mean(11)
12.0
>>> current_mean(13)
12.2
```

尽管这个解决方案更加冗长，但是您不再需要一个不断增长的列表。现在`total`和`length`只有一个值。与之前的解决方案相比，这种实现在内存消耗方面要高效得多。

最后，您可以在 Python 标准库中找到一些使用闭包的例子。例如， [`functools`](https://docs.python.org/3/library/functools.html) 提供了一个名为 [`partial()`](https://docs.python.org/3/library/functools.html#functools.partial) 的函数，它利用闭包技术来创建新的函数对象，可以使用预定义的参数来调用这些对象。这里有一个例子:

>>>

```py
>>> from functools import partial
>>> def power(exp, base):
...     return base ** exp
...
>>> square = partial(power, 2)
>>> square(10)
100
```

您使用`partial`构建一个记住状态信息的函数对象，其中`exp=2`。然后，调用这个对象执行乘幂运算，得到最终结果。

[*Remove ads*](/account/join/)

## 用`import` 将名称带入范围

当您编写 Python 程序时，通常会将代码组织成几个模块。为了让你的程序工作，你需要把那些独立模块中的名字带到你的`__main__`模块中。为此，您需要显式地`import`模块或名称。这是在主全局 Python 范围内使用这些名称的唯一方式。

请看下面的代码，这是一个当您导入一些标准模块和名称时会发生什么的例子:

>>>

```py
>>> dir()
['__annotations__', '__builtins__',..., '__spec__']
>>> import sys >>> dir()
['__annotations__', '__builtins__',..., '__spec__', 'sys']
>>> import os >>> dir()
['__annotations__', '__builtins__',..., '__spec__', 'os', 'sys']
>>> from functools import partial >>> dir()
['__annotations__', '__builtins__',..., '__spec__', 'os', 'partial', 'sys']
```

你先从 Python 标准库中导入 [`sys`](https://docs.python.org/3/library/sys.html) 和 [`os`](https://docs.python.org/3/library/os.html) 。通过不带参数地调用`dir()`,您可以看到这些模块现在可以作为名称在您当前的全局作用域中使用了。这样，您可以使用点符号来访问在`sys`和`os`中定义的名称。

**注意:**如果你想更深入地了解 Python 中导入是如何工作的，那么看看 Python 中的[绝对与相对导入](https://realpython.com/absolute-vs-relative-python-imports/)。

在最近的`import`操作中，您使用表单`from <module> import <name>`。这样，您可以在代码中直接使用导入的名称。换句话说，你不需要明确地使用点符号。

## 发现不寻常的 Python 作用域

您会发现一些 Python 结构的名称解析似乎不符合 Python 作用域的 LEGB 规则。这些结构包括:

*   [理解](https://realpython.com/courses/using-list-comprehensions-effectively/)
*   [异常块](https://realpython.com/courses/introduction-python-exceptions/)
*   [类和实例](https://realpython.com/courses/intro-object-oriented-programming-oop-python/)

在接下来的几节中，您将讨论 Python 作用域如何作用于这三种结构。有了这些知识，您将能够避免与在这些类型的 Python 结构中使用名称相关的微妙错误。

### 理解变量范围

你要覆盖的第一个结构是 [**理解**](https://realpython.com/list-comprehension-python/) 。理解是处理集合或序列中所有或部分元素的一种简洁方式。你可以使用理解来创建[列表](https://realpython.com/python-lists-tuples/)、[字典](https://realpython.com/courses/dictionaries-python/)和[集合](https://realpython.com/python-sets/)。

理解由一对包含表达式的括号(`[]`)或花括号(`{}`)组成，后跟一个或多个`for`子句，然后每个`for`子句有零个或一个`if`子句。

理解中的`for`子句类似于传统的 [`for`循环](https://realpython.com/python-for-loop/)。理解中的循环变量是结构的局部变量。查看以下代码:

>>>

```py
>>> [item for item in range(5)]
[0, 1, 2, 3, 4]
>>> item  # Try to access the comprehension variable Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
    item
NameError: name 'item' is not defined
```

一旦你运行了[列表理解](https://realpython.com/list-comprehension-python/)，变量`item`被遗忘，你不能再访问它的值。不太可能需要在理解之外使用这个变量，但是不管怎样，Python 会确保一旦理解完成，它的值就不再可用。

请注意，这只适用于理解。对于常规的`for`循环，循环变量保存循环处理的最后一个值:

>>>

```py
>>> for item in range(5):
...     print(item)
...
0
1
2
3
4
>>> item  # Access the loop variable 4
```

一旦循环结束，您可以自由访问循环变量`item`。这里，循环变量保存循环处理的最后一个值，在本例中是`4`。

[*Remove ads*](/account/join/)

### 异常变量范围

您将遇到的另一个 Python 作用域的非典型情况是**异常变量**的情况。异常变量是保存对由 [`try`语句](https://docs.python.org/3/reference/compound_stmts.html#the-try-statement)引发的[异常](https://realpython.com/python-exceptions/)的引用的变量。在 Python 3.x 中，这样的变量对于`except`块来说是局部的，当块结束时就会被遗忘。查看以下代码:

>>>

```py
>>> lst = [1, 2, 3]
>>> try:
...     lst[4]
... except IndexError as err:
...     # The variable err is local to this block ...     # Here you can do anything with err ...     print(err)
...
list index out of range
>>> err # Is out of scope Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
    err
NameError: name 'err' is not defined
```

`err`保存对由`try`子句引发的异常的引用。您只能在`except`子句的代码块中使用`err`。这样，您可以说异常变量的 Python 范围对于`except`代码块来说是局部的。还要注意，如果您试图从`except`块外部访问`err`，那么您将得到一个`NameError`。那是因为一旦`except`块结束，名字就不存在了。

要解决这个问题，您可以在`try`语句中定义一个辅助变量，然后将异常分配给`except`块中的那个变量。看看下面的例子:

>>>

```py
>>> lst = [1, 2, 3]
>>> ex = None >>> try:
...     lst[4]
... except IndexError as err:
...     ex = err ...     print(err)
...
list index out of range
>>> err  # Is out of scope Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'err' is not defined
>>> ex  # Holds a reference to the exception list index out of range
```

您使用`ex`作为辅助变量来保存对由`try`子句引发的异常的引用。当您需要在代码块完成后对异常对象做一些事情时，这可能会很有用。注意，如果没有出现异常，那么`ex`保持`None`。

### 类别和实例属性范围

当您定义一个类时，您正在创建一个新的本地 Python 范围。在类的顶层分配的名字存在于这个局部作用域中。您在`class`语句中指定的名字不会与其他地方的名字冲突。你可以说这些名字遵循了 LEGB 规则，其中类块代表了 **L** 级别。

与函数不同，类局部范围不是在调用时创建的，而是在执行时创建的。每个类对象都有自己的`.__dict__`属性，保存所有**类属性**所在的类范围或名称空间。查看以下代码:

>>>

```py
>>> class A:
...     attr = 100
...
>>> A.__dict__.keys() dict_keys(['__module__', 'attr', '__dict__', '__weakref__', '__doc__'])
```

当您检查`.__dict__`的键时，您会看到`attr`和其他特殊名称一起出现在列表中。这个字典表示类的局部范围。此范围内的名称对该类的所有实例和该类本身都是可见的。

要从类外部访问类属性，您需要使用点符号，如下所示:

>>>

```py
>>> class A:
...     attr = 100
...     print(attr)  # Access class attributes directly ...
100
>>> A.attr  # Access a class attribute from outside the class 100
>>> attr  # Isn't defined outside A Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
    attr
NameError: name 'attr' is not defined
```

在`A`的局部范围内，可以直接访问类属性，就像在语句`print(attr)`中一样。一旦类的代码块被执行，要访问任何类属性，你需要使用点符号或[属性引用](https://docs.python.org/3/reference/expressions.html#attribute-references)，就像你使用`A.attr`一样。否则，您将得到一个`NameError`，因为属性`attr`对于类块来说是局部的。

另一方面，如果你试图访问一个没有在类中定义的属性，那么你会得到一个 [`AttributeError`](https://docs.python.org/3/library/exceptions.html#AttributeError) 。看看下面的例子:

>>>

```py
>>> A.undefined  # Try to access an undefined class attribute Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
    A.undefined
AttributeError: type object 'A' has no attribute 'undefined'
```

在这个例子中，您试图访问属性`undefined`。由于这个属性在`A`中不存在，您会得到一个`AttributeError`，告诉您`A`没有名为`undefined`的属性。

还可以使用类的实例访问任何类属性，如下所示:

>>>

```py
>>> obj = A()
>>> obj.attr
100
```

一旦有了实例，就可以使用点符号访问类属性，就像这里使用`obj.attr`一样。类属性是特定于类对象的，但是您可以从该类的任何实例中访问它们。值得注意的是，类属性对于一个类的所有实例都是通用的。如果您修改了一个类属性，那么这些更改将在该类的所有实例中可见。

**注意:**把点符号想象成你在告诉 Python，“在`obj`中寻找名为`attr`的属性。如果你找到了，就把它还给我。”

无论何时调用一个类，你都在创建该类的一个新实例。实例有自己的`.__dict__`属性，该属性保存实例本地范围或名称空间中的名称。这些名称通常被称为**实例属性**，并且是本地的，特定于每个实例。这意味着，如果您修改实例属性，则更改将仅对该特定实例可见。

要在类内部创建、更新或访问任何实例属性，需要使用 [`self`](https://docs.python.org/3/faq/programming.html#what-is-self) 和点符号。这里，`self`是表示当前实例的特殊属性。另一方面，要从类外部更新或访问任何实例属性，您需要创建一个实例，然后使用点符号。这是如何工作的:

>>>

```py
>>> class A:
...     def __init__(self, var):
...         self.var = var  # Create a new instance attribute ...         self.var *= 2  # Update the instance attribute ...
>>> obj = A(100)
>>> obj.__dict__ {'var': 200}
>>> obj.var
200
```

类`A`接受一个名为`var`的参数，该参数使用赋值操作`self.var *= 2`在 [`.__init__()`](https://docs.python.org/3/reference/datamodel.html#object.__init__) 中自动加倍。注意，当您在`obj`上检查`.__dict__`时，您会得到一个包含所有实例属性的字典。在这种情况下，字典只包含名字`var`，它的值现在是`200`。

**注意:**关于 Python 中类如何工作的更多信息，请查看[Python 中面向对象编程的介绍](https://realpython.com/courses/intro-object-oriented-programming-oop-python/)。

尽管您可以在一个类的任何方法中创建实例属性，但是在`.__init__()`中创建和初始化它们是一个很好的实践。看看这个新版本的`A`:

>>>

```py
>>> class A:
...     def __init__(self, var):
...         self.var = var
...
...     def duplicate_var(self):
...         return self.var * 2 ...
>>> obj = A(100)
>>> obj.var
100
>>> obj.duplicate_var()
200
>>> A.var Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
    A.var
AttributeError: type object 'A' has no attribute 'var'
```

这里，您修改`A`来添加一个名为`duplicate_var()`的新方法。然后，通过将`100`传递给类初始化器，创建一个`A`的实例。之后，您现在可以在`obj`上调用`duplicate_var()`来复制存储在`self.var`中的值。最后，如果您试图使用类对象而不是实例来访问`var`，那么您将得到一个`AttributeError`，因为实例属性不能使用类对象来访问。

一般来说，当你用 Python 写[面向对象的](https://realpython.com/python3-object-oriented-programming/)代码并试图访问一个属性时，你的程序会采取以下步骤:

1.  首先检查**实例**的本地范围或名称空间。
2.  如果在那里没有找到该属性，那么检查**类**的局部范围或名称空间。
3.  如果这个名字也不存在于类名称空间中，那么您将得到一个 **`AttributeError`** 。

这是 Python 解析类和实例中的名称的底层机制。

尽管类定义了类的局部作用域或命名空间，但它们并没有为方法创建封闭的作用域。因此，当您实现一个类时，对属性和方法的引用必须使用点符号:

>>>

```py
>>> class A:
...     var = 100
...     def print_var(self):
...         print(var)  # Try to access a class attribute directly ...
>>> A().print_var()
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
    A().print_var()
  File "<stdin>", line 4, in print_var
    print(var)
NameError: name 'var' is not defined
```

因为类不会为方法创建封闭范围，所以您不能像这里尝试的那样从`print_var()`中直接访问`var`。要从任何方法内部访问类属性，您需要使用点符号。要解决这个例子中的问题，将`print_var()`中的语句`print(var)`改为`print(A.var)`，看看会发生什么。

可以用实例属性重写类属性，这将修改类的一般行为。但是，您可以使用点符号明确地访问这两个属性，如下例所示:

>>>

```py
>>> class A:
...     var = 100
...     def __init__(self):
...         self.var = 200
...
...     def access_attr(self):
...         # Use dot notation to access class and instance attributes ...         print(f'The instance attribute is: {self.var}')
...         print(f'The class attribute is: {A.var}')
...
>>> obj = A()
>>> obj.access_attr()
The instance attribute is: 200
The class attribute is: 100
>>> A.var  # Access class attributes 100
>>> A().var # Access instance attributes 200
>>> A.__dict__.keys()
dict_keys(['__module__', 'var', '__init__',..., '__getattribute__'])
>>> A().__dict__.keys()
dict_keys(['var'])
```

上面的类有一个实例属性和一个同名的类属性`var`。您可以使用以下代码来访问它们:

1.  **实例:**使用`self.var`来访问这个属性。
2.  **类:**使用`A.var`访问该属性。

因为这两种情况都使用点符号，所以不存在名称冲突问题。

**注意:**一般来说，良好的 [OOP](https://realpython.com/courses/intro-object-oriented-programming-oop-python/) 实践建议不要用具有不同职责或执行不同操作的实例属性来隐藏类属性。这样做可能会导致细微且难以发现的错误。

最后，请注意，类`.__dict__`和实例`.__dict__`是完全不同且独立的字典。这就是为什么在运行或导入定义类的模块后，类属性立即可用。相比之下，实例属性只有在对象或实例创建之后才具有生命力。

[*Remove ads*](/account/join/)

## 使用与范围相关的内置函数

有许多内置函数与 Python 范围和名称空间的概念密切相关。在前面的小节中，您已经使用了`dir()`来获取给定范围内存在的名称的信息。除了`dir()`之外，当您试图获取关于 Python 作用域或名称空间的信息时，还有其他一些内置函数可以帮助您。在本节中，您将了解如何使用:

*   [T2`globals()`](https://docs.python.org/3/library/functions.html#globals)
*   [T2`locals()`](https://docs.python.org/3/library/functions.html#locals)
*   [T2`dir()`](https://docs.python.org/3/library/functions.html#dir)
*   [T2`vars()`](https://docs.python.org/3/library/functions.html#vars)

因为所有这些都是内置函数，所以它们在内置范围内是免费的。这意味着您可以随时使用它们，而无需导入任何内容。这些函数中的大部分旨在用于交互式会话中，以获取关于不同 Python 对象的信息。然而，您也可以在您的代码中找到一些有趣的用例。

### `globals()`

在 Python 中， **`globals()`** 是一个内置函数，返回对当前全局作用域或命名空间字典的引用。这个字典总是存储当前模块的名称。这意味着如果你在一个给定的模块中调用`globals()`，那么在调用`globals()`之前，你会得到一个字典，包含你在那个模块中定义的所有名字。这里有一个例子:

>>>

```py
>>> globals()
{'__name__': '__main__',..., '__builtins__': <module 'builtins' (built-in)>}
>>> my_var = 100
>>> globals()
{'__name__': '__main__',..., 'my_var': 100}
```

对`globals()`的第一次调用返回一个字典，其中包含了`__main__`模块或程序中的名字。注意，当你在模块的顶层指定一个新名字时，比如在`my_var = 100`中，这个名字被添加到由`globals()`返回的字典中。

如何在代码中使用`globals()`的一个有趣的例子是动态地分派位于全局范围内的函数。假设您想要动态调度平台相关的功能。为此，您可以如下使用`globals()`:

```py
 1# Filename: dispatch.py
 2
 3from sys import platform
 4
 5def linux_print():
 6    print('Printing from Linux...')
 7
 8def win32_print():
 9    print('Printing from Windows...')
10
11def darwin_print():
12    print('Printing from macOS...')
13
14printer = globals()[platform + '_print'] 15
16printer()
```

如果您在命令行中运行这个脚本，那么您将得到一个依赖于您当前平台的输出。

如何使用`globals()`的另一个例子是在全局范围内检查**特殊名称**的列表。看看下面的列表理解:

>>>

```py
>>> [name for name in globals() if name.startswith('__')]
['__name__', '__doc__', '__package__',..., '__annotations__', '__builtins__']
```

这个列表理解将返回一个列表，其中包含当前全局 Python 作用域中定义的所有特殊名称。请注意，您可以像使用任何常规词典一样使用`globals()`词典。例如，您可以使用这些传统方法通过它对进行迭代:

*   `.keys()`
*   `.values()`
*   `.items()`

您还可以通过使用类似于`globals()['name']`中的方括号在`globals()`上执行常规订阅操作。例如，您可以修改`globals()`的内容，尽管我们不建议这样做。看一下这个例子:

>>>

```py
>>> globals()['__doc__'] = """Docstring for __main__.""" >>> __doc__
'Docstring for __main__.'
```

在这里，您更改键`__doc__`，为`__main__`包含一个[文档字符串](https://realpython.com/documenting-python-code/#documenting-your-python-code-base-using-docstrings)，这样从现在开始，主模块的文档字符串将具有值`'Docstring for __main__.'`。

### `locals()`

另一个与 Python 作用域和名称空间相关的函数是 **`locals()`** 。该函数更新并返回一个字典，该字典保存了本地 Python 范围或名称空间的当前状态的副本。当你在一个函数块中调用`locals()`时，你会得到直到你调用`locals()`时在局部或函数作用域中分配的所有名字。这里有一个例子:

>>>

```py
>>> def func(arg):
...     var = 100
...     print(locals()) ...     another = 200
...
>>> func(300)
{'var': 100, 'arg': 300}
```

每当您在`func()`中调用`locals()`时，结果字典包含映射到值`100`的名称`var`和映射到`300`的`arg`。因为`locals()`只在你调用它之前获取指定的名字，所以`another`不在字典中。

如果您在全局 Python 范围内调用`locals()`，那么您将获得与调用`globals()`时相同的字典:

>>>

```py
>>> locals()
{'__name__': '__main__',..., '__builtins__': <module 'builtins' (built-in)>}
>>> locals() is globals() True
```

当您在全局 Python 范围内调用`locals()`时，您会得到一个与调用`globals()`返回的字典相同的字典。

请注意，您不应该修改`locals()`的内容，因为更改可能对本地和自由名称的值没有影响。看看下面的例子:

>>>

```py
>>> def func():
...     var = 100
...     locals()['var'] = 200 ...     print(var)
...
>>> func()
100
```

当您试图使用`locals()`修改`var`的内容时，这种变化不会反映在`var`的值中。所以，你可以说`locals()`只对读操作有用，因为 Python 忽略了对`locals`字典的更新。

[*Remove ads*](/account/join/)

### `vars()`

**`vars()`** 是一个 Python 内置函数，返回模块、类、实例或任何其他具有字典属性的对象的`.__dict__`属性。记住`.__dict__`是 Python 用来实现名称空间的特殊字典。看看下面的例子:

>>>

```py
>>> import sys
>>> vars(sys) # With a module object {'__name__': 'sys',..., 'ps1': '>>> ', 'ps2': '... '}
>>> vars(sys) is sys.__dict__ True
>>> class MyClass:
...     def __init__(self, var):
...         self.var = var
...
>>> obj = MyClass(100)
>>> vars(obj)  # With a user-defined object {'var': 100}
>>> vars(MyClass)  # With a class mappingproxy({'__module__': '__main__',..., '__doc__': None})
```

当你使用`sys`作为参数调用`vars()`时，你得到了`sys`的`.__dict__`。你也可以使用不同类型的 Python 对象来调用`vars()`，只要它们具有这个字典属性。

没有任何参数，`vars()`的行为类似于`locals()`,返回一个包含本地 Python 范围内所有名称的字典:

>>>

```py
>>> vars()
{'__name__': '__main__',..., '__builtins__': <module 'builtins' (built-in)>}
>>> vars() is locals() True
```

在这里，您在交互式会话的顶层调用`vars()`。如果没有参数，该调用将返回一个包含全局 Python 范围内所有名称的字典。注意，在这个级别，`vars()`和`locals()`返回相同的字典。

如果你用一个没有`.__dict__`的对象调用`vars()`，那么你将得到一个`TypeError`，如下例所示:

>>>

```py
>>> vars(10)  # Call vars() with objects that don't have a .__dict__ Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: vars() argument must have __dict__ attribute
```

如果你用一个整数对象调用`vars()`，那么你会得到一个 [`TypeError`](https://docs.python.org/3/library/exceptions.html#TypeError) ，因为这种类型的 Python 对象没有`.__dict__`。

### `dir()`

可以使用不带参数的 **`dir()`** 来获取当前 Python 范围内的名称列表。如果您用一个参数调用`dir()`，那么该函数将尝试返回该对象有效属性的`list`:

>>>

```py
>>> dir()  # With no arguments ['__annotations__', '__builtins__',..., '__package__', '__spec__']
>>> dir(zip)  # With a function object ['__class__', '__delattr__',..., '__str__', '__subclasshook__']
>>> import sys
>>> dir(sys)  # With a module object ['__displayhook__', '__doc__',..., 'version_info', 'warnoptions']
>>> var = 100
>>> dir(var)  # With an integer variable ['__abs__', '__add__',..., 'imag', 'numerator', 'real', 'to_bytes']
```

如果您不带参数调用`dir()`，那么您将得到一个包含全局范围内的名字的列表。您还可以使用`dir()`来检查不同对象的名称或属性列表。这包括函数、模块、变量等等。

尽管[官方文档](https://docs.python.org/3/library/functions.html#dir)说`dir()`是用于交互使用的，但是你可以使用该函数来提供一个给定对象的属性的完整列表。注意，您也可以从函数内部调用`dir()`。在这种情况下，您将获得在函数作用域中定义的名称列表:

>>>

```py
>>> def func():
...     var = 100
...     print(dir())
...     another = 200  # Is defined after calling dir() ...
>>> func()
['var']
```

在这个例子中，您在`func()`中使用了`dir()`。当您调用该函数时，您会得到一个包含您在局部范围内定义的名称的列表。值得注意的是，在这种情况下，`dir()`只显示你在函数调用前声明的名字。

## 结论

变量或名称的**范围**定义了它在整个代码中的可见性。在 Python 中，作用域实现为局部、封闭、全局或内置作用域。当您使用变量或名称时，Python 会按顺序搜索这些范围来解析它。如果找不到这个名字，你会得到一个错误。这是 Python 用于名称解析的一般机制，被称为 **LEGB 规则**。

**您现在能够:**

*   **利用**Python 作用域的优势来避免或最小化与名称冲突相关的错误
*   在你的程序中充分利用全局和局部名字来提高代码的可维护性
*   使用一致的策略来访问、修改或更新所有 Python 代码的名称

此外，您还了解了 Python 提供的一些与作用域相关的工具和技术，以及如何使用它们来收集关于存在于给定作用域中的名称的信息，或者修改 Python 作用域的标准行为。当然，这个主题的更多内容已经超出了本教程的范围，所以请出去继续学习 Python 中的名称解析吧！**********