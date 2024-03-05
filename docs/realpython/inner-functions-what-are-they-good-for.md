# Python 内部函数:它们有什么用？

> 原文：<https://realpython.com/inner-functions-what-are-they-good-for/>

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和文字教程一起看，加深理解: [**Python 内部函数**](/courses/python-inner-functions/)

**内部函数**，也称为**嵌套函数**，是你在其他函数内部定义的[函数](https://realpython.com/defining-your-own-python-function/)。在 Python 中，这种函数可以直接访问封闭函数中定义的变量和名称。内部函数有很多用途，最著名的是作为闭包工厂和装饰函数。

**在本教程中，您将学习如何:**

*   提供**封装**并隐藏您的功能，防止外部访问
*   编写**助手函数**以促进代码重用
*   创建在调用之间保持状态的**闭包工厂函数**
*   编写代码**装饰函数**以向现有函数添加行为

**免费奖励:** 并学习 Python 3 的基础知识，如使用数据类型、字典、列表和 Python 函数。

## 创建 Python 内部函数

定义在另一个函数内部的函数被称为**内部函数**或**嵌套函数**。在 Python 中，这种函数可以访问封闭函数中的[名](https://realpython.com/python-scope-legb-rule/#names-and-scopes-in-python)。以下是如何在 Python 中创建内部函数的示例:

>>>

```py
>>> def outer_func():
...     def inner_func():
...         print("Hello, World!")
...     inner_func()
...

>>> outer_func()
Hello, World!
```

在这段代码中，您在`outer_func()`到[中定义了`inner_func()`将`Hello, World!`消息打印到屏幕上](https://realpython.com/python-print/)。为此，您在`outer_func()`的最后一行调用`inner_func()`。这是用 Python 编写内部函数最快的方法。然而，内部函数提供了许多有趣的可能性，超出了你在这个例子中看到的。

内部函数的核心特性是，即使在函数返回后，它们也能从封闭函数中访问变量和对象。封闭函数提供了一个内部函数可访问的名称空间:

>>>

```py
>>> def outer_func(who):
...     def inner_func():
...         print(f"Hello, {who}")
...     inner_func()
...

>>> outer_func("World!")
Hello, World!
```

现在您可以将一个[字符串](https://realpython.com/python-strings/)作为参数传递给`outer_func()`，`inner_func()`将通过名称`who`访问该参数。然而这个名字是在`outer_func()`的[局部作用域](https://realpython.com/python-scope-legb-rule/#functions-the-local-scope)中定义的。您在外部函数的局部作用域中定义的名称被称为**非局部名称**。从`inner_func()`的角度看，他们是外地的。

下面是一个如何创建和使用更复杂的内部函数的例子:

>>>

```py
>>> def factorial(number):
...     # Validate input
...     if not isinstance(number, int):
...         raise TypeError("Sorry. 'number' must be an integer.")
...     if number < 0:
...         raise ValueError("Sorry. 'number' must be zero or positive.")
...     # Calculate the factorial of number
...     def inner_factorial(number):
...         if number <= 1:
...             return 1
...         return number * inner_factorial(number - 1)
...     return inner_factorial(number)
...

>>> factorial(4)
24
```

在`factorial()`中，首先验证输入数据，确保用户提供的是一个等于或大于零的整数。然后定义一个名为`inner_factorial()`的递归内部函数来执行阶乘计算，[将结果返回给](https://realpython.com/python-return-statement/)。最后一步是给`inner_factorial()`打电话。

**注:**关于递归和递归函数更详细的讨论，请查看[用 Python 递归思维](https://realpython.com/python-thinking-recursively/)和[用 Python 递归:简介](https://realpython.com/python-recursion/)。

使用这种模式的主要优点是，通过在外部函数中执行所有的参数检查，您可以安全地跳过内部函数中的错误检查，并专注于手头的计算。

[*Remove ads*](/account/join/)

## 使用内部函数:基础知识

Python 内部函数的用例多种多样。你可以使用它们来提供[封装](https://en.wikipedia.org/wiki/Encapsulation_(computer_programming))和[隐藏](http://en.wikipedia.org/wiki/Information_hiding)你的外部访问的函数，你可以编写助手内部函数，你也可以创建[闭包](https://en.wikipedia.org/wiki/Closure_(computer_programming))和[装饰器](https://en.wikipedia.org/wiki/Decorator_pattern)。在本节中，您将了解内部函数的前两个用例，在后面的章节中，您将了解如何创建[闭包工厂函数](#retaining-state-with-inner-functions-closures)和[装饰器](#adding-behavior-with-inner-functions-decorators)。

### 提供封装

内部函数的一个常见用例是，当你需要保护或隐藏一个给定的函数，使其不被外部发生的任何事情影响时，这个函数就完全隐藏在全局[作用域](https://realpython.com/python-namespaces-scope/)之外了。这种行为俗称**封装**。

这里有一个例子突出了这个概念:

>>>

```py
>>> def increment(number):
...     def inner_increment():
...         return number + 1
...     return inner_increment()
...

>>> increment(10)
11

>>> # Call inner_increment()
>>> inner_increment()
Traceback (most recent call last):
  File "<input>", line 1, in <module>
    inner_increment()
NameError: name 'inner_increment' is not defined
```

在这个例子中，你不能直接访问`inner_increment()`。如果你尝试去做，那么你会得到一个`NameError`。那是因为`increment()`完全隐藏了`inner_increment()`，防止你从全局范围内访问它。

### 构建助手内部函数

有时你有一个函数，它在体的几个地方执行相同的代码块。例如，假设您想编写一个函数来处理包含纽约市 [Wi-Fi 热点](https://data.cityofnewyork.us/City-Government/NYC-Wi-Fi-Hotspot-Locations/yjub-udmw)信息的 [CSV 文件](https://realpython.com/python-csv/)。要查找纽约热点的总数以及提供大多数热点的公司，您可以创建以下脚本:

```py
# hotspots.py

import csv
from collections import Counter

def process_hotspots(file):
    def most_common_provider(file_obj):
        hotspots = []
        with file_obj as csv_file:
            content = csv.DictReader(csv_file)

            for row in content:
                hotspots.append(row["Provider"])

        counter = Counter(hotspots)
        print(
            f"There are {len(hotspots)} Wi-Fi hotspots in NYC.\n"
            f"{counter.most_common(1)[0][0]} has the most with "
            f"{counter.most_common(1)[0][1]}."
        )

    if isinstance(file, str):
        # Got a string-based filepath
        file_obj = open(file, "r")
        most_common_provider(file_obj)
    else:
        # Got a file object
        most_common_provider(file)
```

这里，`process_hotspots()`以`file`为自变量。该函数检查`file`是一个物理文件的基于字符串的路径还是一个[文件对象](https://docs.python.org/3/glossary.html#term-file-object)。然后它调用助手内部函数`most_common_provider()`，该函数接受一个文件对象并执行以下操作:

1.  将文件内容读入一个生成器，该生成器使用 [`csv.DictReader`](https://docs.python.org/3/library/csv.html#csv.DictReader) 生成[字典](https://realpython.com/python-dicts/)。
2.  创建 Wi-Fi 提供商列表。
3.  使用 [`collections.Counter`](https://docs.python.org/3/library/collections.html#collections.Counter) 对象统计每个提供商的 Wi-Fi 热点数量。
4.  打印包含检索信息的消息。

如果运行该函数，您将获得以下输出:

>>>

```py
>>> from hotspots import process_hotspots

>>> file_obj = open("./NYC_Wi-Fi_Hotspot_Locations.csv", "r")
>>> process_hotspots(file_obj)
There are 3319 Wi-Fi hotspots in NYC.
LinkNYC - Citybridge has the most with 1868.

>>> process_hotspots("./NYC_Wi-Fi_Hotspot_Locations.csv")
There are 3319 Wi-Fi hotspots in NYC.
LinkNYC - Citybridge has the most with 1868.
```

无论是用基于字符串的文件路径还是用文件对象调用`process_hotspots()`，都会得到相同的结果。

### 使用内部与私有帮助函数

通常，当您想要提供封装时，可以创建类似于`most_common_provider()`的帮助器内部函数。如果您认为除了包含函数之外不会在其他地方调用内部函数，您也可以创建内部函数。

虽然将助手函数编写为内部函数可以达到预期的效果，但是将它们提取为顶级函数可能会更好。在这种情况下，您可以在函数名中使用一个前导下划线(`_`)来表示它是当前[模块](https://realpython.com/python-modules-packages/)或类的私有函数。这将允许您从当前模块或类中的任何地方访问您的助手函数，并在需要时重用它们。

将内部函数提取到顶级私有函数中可以让你的代码更整洁，可读性更好。这种实践可以产生应用[单一责任原则](https://en.wikipedia.org/wiki/Single-responsibility_principle)的函数。

## 内部函数保持状态:闭包

在 Python 中，函数是[一等公民](https://realpython.com/primer-on-python-decorators/#first-things-first)。这意味着它们与任何其他对象一样，比如[数字](https://realpython.com/python-numbers/)、[字符串](https://realpython.com/python-strings/)、[列表、元组](https://realpython.com/python-lists-tuples/)、模块等等。您可以动态地创建或销毁它们，将它们存储在[数据结构](https://realpython.com/python-data-structures/)中，将它们作为参数传递给其他函数，将它们作为[返回值](https://realpython.com/python-return-statement/)等等。

也可以在 Python 中创建[高阶函数](http://en.wikipedia.org/wiki/Higher-order_function)。**高阶函数**是对其他函数进行操作的函数，将它们作为参数，返回它们，或者两者都做。

到目前为止，您看到的所有内部函数的例子都是普通函数，只是碰巧嵌套在其他函数中。除非你需要对外界隐藏你的函数，否则没有必要嵌套它们。您可以将这些函数定义为私有的顶级函数，这样就可以了。

在本节中，您将了解到**闭包工厂函数**。闭包是由其他函数返回的动态创建的函数。它们的主要特性是，即使封闭函数已经返回并完成执行，它们也可以完全访问在创建闭包的本地名称空间中定义的变量和名称。

在 Python 中，当您返回一个内部函数对象时，解释器会将函数与其包含的环境或闭包一起打包。function 对象保存了在其包含范围内定义的所有变量和名称的快照。要定义一个闭包，您需要采取三个步骤:

1.  创建一个内部函数。
2.  来自封闭函数的引用变量。
3.  返回内部函数。

有了这些基础知识，您就可以马上开始创建您的闭包，并利用它们的主要特性:**在函数调用之间保持状态**。

[*Remove ads*](/account/join/)

### 关闭时的保持状态

闭包使得内部函数在被调用时保持其环境的状态。闭包不是内部函数本身，而是内部函数及其封闭环境。闭包捕获包含函数中的局部变量和名称，并保存它们。

考虑下面的例子:

```py
 1# powers.py
 2
 3def generate_power(exponent):
 4    def power(base):
 5        return base ** exponent
 6    return power
```

下面是该函数中发生的情况:

*   **第 3 行**创建`generate_power()`，这是一个闭包工厂函数。这意味着它每次被调用时都会创建一个新的闭包，然后将其返回给调用者。
*   **第 4 行**定义了`power()`，这是一个内部函数，它接受一个参数`base`，并返回表达式`base ** exponent`的结果。
*   **第 6 行**将`power`作为函数对象返回，不调用它。

`power()`从哪里得到`exponent`的值？这就是闭包发挥作用的地方。在这个例子中，`power()`从外部函数`generate_power()`中获取`exponent`的值。当您调用`generate_power()`时，Python 会这样做:

1.  定义一个新的`power()`实例，它接受一个参数`base`。
2.  拍摄`power()`周围状态的快照，其中包括`exponent`及其当前值。
3.  返回`power()`连同它的整个周围状态。

这样，当你调用由`generate_power()`返回的`power()`的实例时，你会看到函数记住了`exponent`的值:

>>>

```py
>>> from powers import generate_power

>>> raise_two = generate_power(2)
>>> raise_three = generate_power(3)

>>> raise_two(4)
16
>>> raise_two(5)
25

>>> raise_three(4)
64
>>> raise_three(5)
125
```

在这些例子中，`raise_two()`记得那个`exponent=2`，而`raise_three()`记得那个`exponent=3`。注意，两个闭包都会在调用之间记住它们各自的`exponent`。

现在考虑另一个例子:

>>>

```py
>>> def has_permission(page):
...     def permission(username):
...         if username.lower() == "admin":
...             return f"'{username}' has access to {page}."
...         else:
...             return f"'{username}' doesn't have access to {page}."
...     return permission
...

>>> check_admin_page_permision = has_permission("Admin Page")

>>> check_admin_page_permision("admin")
"'admin' has access to Admin Page."

>>> check_admin_page_permision("john")
"'john' doesn't have access to Admin Page."
```

内部函数检查给定用户是否有访问给定页面的正确权限。您可以快速修改它，以获取会话中的用户，检查他们是否拥有访问某个路由的正确凭证。

不用检查用户是否等于`"admin"`，您可以查询一个 [SQL 数据库](https://realpython.com/python-sql-libraries/)来检查权限，然后根据凭证是否正确返回正确的视图。

您通常会创建不修改其封闭状态的闭包，或者创建具有静态封闭状态的闭包，正如您在上面的例子中所看到的。但是，您也可以创建闭包，通过使用可变对象，比如字典、集合或列表，来修改它们的封闭状态。

假设你需要计算一个数据集的平均值。数据来自于正在分析的参数的连续测量流，您需要您的函数在调用之间保留先前的测量。在这种情况下，您可以像这样编写一个闭包工厂函数:

>>>

```py
>>> def mean():
...     sample = []
...     def inner_mean(number):
...         sample.append(number)
...         return sum(sample) / len(sample)
...     return inner_mean
...

>>> sample_mean = mean()
>>> sample_mean(100)
100.0
>>> sample_mean(105)
102.5
>>> sample_mean(101)
102.0
>>> sample_mean(98)
101.0
```

分配给`sample_mean`的闭包在连续调用之间保持`sample`的状态。即使您在`mean()`中定义了`sample`，它仍然在闭包中可用，因此您可以修改它。在这种情况下，`sample`作为一种动态的封闭状态。

### 修改关闭状态

通常，闭包变量对外界是完全隐藏的。但是，您可以为它们提供 **getter** 和 **setter** 内部函数:

>>>

```py
>>> def make_point(x, y):
...     def point():
...         print(f"Point({x}, {y})")
...     def get_x():
...         return x
...     def get_y():
...         return y
...     def set_x(value):
...         nonlocal x
...         x = value
...     def set_y(value):
...         nonlocal y
...         y = value
...     # Attach getters and setters
...     point.get_x = get_x
...     point.set_x = set_x
...     point.get_y = get_y
...     point.set_y = set_y
...     return point
...

>>> point = make_point(1, 2)
>>> point.get_x()
1
>>> point.get_y()
2
>>> point()
Point(1, 2)

>>> point.set_x(42)
>>> point.set_y(7)
>>> point()
Point(42, 7)
```

这里，`make_point()`返回一个表示一个`point`对象的闭包。这个对象附带了 getter 和 setter 函数。您可以使用这些函数来获得对变量`x`和`y`的读写访问，这些变量在封闭范围中定义，并随闭包一起提供。

尽管这个函数创建的闭包可能比等价的类运行得更快，但是您需要注意，这种技术并没有提供主要的特性，包括[继承](https://realpython.com/inheritance-composition-python/)，属性，[描述符](https://realpython.com/python-descriptors/)，以及[类和静态方法](https://realpython.com/instance-class-and-static-methods-demystified/)。如果您想更深入地研究这项技术，那么请查看[使用闭包和嵌套作用域模拟类的简单工具(Python Recipe)](https://code.activestate.com/recipes/578091-simple-tool-for-simulating-classes-using-closures-/) 。

[*Remove ads*](/account/join/)

## 用内部函数添加行为:装饰者

python[decorator](https://realpython.com/primer-on-python-decorators/)是内部函数的另一个流行且方便的用例，尤其是对于闭包。**decorator**是高阶函数，它将一个可调用函数(函数、方法、类)作为参数，并返回另一个可调用函数。

您可以使用 decorator 函数向现有的可调用程序动态添加职责，并透明地扩展其行为，而不会影响或修改原始的可调用程序。

**注意:**关于 Python 可调用对象的更多细节，请查看 Python 文档中的标准类型层次的[，并向下滚动到“可调用类型”](https://docs.python.org/3/reference/datamodel.html#the-standard-type-hierarchy)

要创建一个装饰器，您只需要定义一个 callable(一个函数、方法或类),它接受一个 function 对象作为参数，处理它，并返回另一个带有附加行为的 function 对象。

一旦有了 decorator 函数，就可以将它应用于任何可调用的。为此，您需要在装饰器名称前面使用 at 符号(`@`)，然后将它放在自己的行上，紧接在被装饰的可调用函数之前:

```py
@decorator def decorated_func():
    # Function body...
    pass
```

这个语法让`decorator()`自动将`decorated_func()`作为参数，并在其主体中处理。该操作是以下赋值的简写:

```py
decorated_func = decorator(decorated_func)
```

下面是一个如何构建装饰函数来向现有函数添加新功能的示例:

>>>

```py
>>> def add_messages(func):
...     def _add_messages():
...         print("This is my first decorator")
...         func()
...         print("Bye!")
...     return _add_messages
...

>>> @add_messages
... def greet():
...     print("Hello, World!")
...

>>> greet()
This is my first decorator
Hello, World!
Bye!
```

在这种情况下，你用`@add_messages`来修饰`greet()`。这给修饰函数增加了新的功能。现在，当您调用`greet()`时，您的函数打印两条新消息，而不仅仅是打印`Hello, World!`。

Python decorators 的用例多种多样。以下是其中的一些:

*   [调试](https://realpython.com/python-debugging-pdb/)
*   [缓存](https://realpython.com/lru-cache-python/)
*   [测井](https://realpython.com/python-logging-source-code/)
*   [定时](https://realpython.com/python-timer/)

调试 Python 代码的一个常见做法是插入对`print()`的调用，以检查变量值，确认代码块被执行，等等。添加和删除对`print()`的呼叫可能很烦人，而且你可能会忘记其中的一些。为了防止这种情况，您可以像这样编写一个装饰器:

>>>

```py
>>> def debug(func):
...     def _debug(*args, **kwargs):
...         result = func(*args, **kwargs)
...         print(
...             f"{func.__name__}(args: {args}, kwargs: {kwargs}) -> {result}"
...         )
...         return result
...     return _debug
...

>>> @debug
... def add(a, b):
...     return a + b
...

>>> add(5, 6)
add(args: (5, 6), kwargs: {}) -> 11
11
```

这个例子提供了`debug()`，这是一个 decorator，它将一个函数作为参数，并用每个参数的当前值及其对应的返回值打印其签名。您可以使用这个装饰器来调试您的函数。一旦您获得了想要的结果，您就可以移除装饰器调用`@debug`，并且您的函数将为下一步做好准备。

**注意:**如果你有兴趣深入探究`*args`和`**kwargs`在 Python 中是如何工作的，那么就去看看 [Python args 和 kwargs:去神秘化](https://realpython.com/python-kwargs-and-args/)。

这是最后一个如何创建装饰器的例子。这一次，您将重新实现`generate_power()`作为装饰函数:

>>>

```py
>>> def generate_power(exponent):
...     def power(func):
...         def inner_power(*args):
...             base = func(*args)
...             return base ** exponent
...         return inner_power
...     return power
...

>>> @generate_power(2)
... def raise_two(n):
...     return n
...
>>> raise_two(7)
49

>>> @generate_power(3)
... def raise_three(n):
...     return n
...
>>> raise_three(5)
125
```

这个版本的`generate_power()`产生的结果与您在最初的实现中获得的结果相同。在这种情况下，您使用一个闭包来记住`exponent`和一个 decorator 来返回输入函数的修改版本`func()`。

在这里，装饰器需要带一个参数(`exponent`)，所以你需要有两层嵌套的内部函数。第一级用`power()`表示，以修饰函数为自变量。第二层以`inner_power()`为代表，将`args`中的自变量`exponent`打包，进行幂的最终计算，并返回结果。

[*Remove ads*](/account/join/)

## 结论

如果你在另一个函数中定义了一个函数，那么你就创建了一个**内部函数**，也称为嵌套函数。在 Python 中，内部函数可以直接访问您在封闭函数中定义的变量和名称。这为您创建助手函数、闭包和装饰器提供了一种机制。

**在本教程中，您学习了如何:**

*   通过在其他函数中嵌套函数来提供**封装**
*   编写**助手函数**来重用代码片段
*   实现**闭包工厂函数**，在调用之间保留状态
*   构建**装饰函数**来提供新的功能

现在，您已经准备好在自己的代码中利用内部函数的许多用途了。如果您有任何问题或意见，请务必在下面的评论区分享。

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和文字教程一起看，加深理解: [**Python 内部函数**](/courses/python-inner-functions/)******