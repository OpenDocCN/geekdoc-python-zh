# pass 语句:如何在 Python 中什么都不做

> 原文：<https://realpython.com/python-pass/>

在 Python 中，`pass` [关键字](https://realpython.com/python-keywords/)本身就是一个完整的**语句**。这条语句[不做任何事情](https://docs.python.org/3/tutorial/controlflow.html#pass-statements):它在**字节编译**阶段被丢弃。但是对于一个什么都不做的语句，Python `pass`语句出奇的有用。

有时`pass`在生产中运行的最终代码中很有用。更多情况下，`pass`在开发代码时作为脚手架很有用。在特定情况下，有比什么都不做更好的选择。

在本教程中，您将学习:

*   Python **`pass`语句**是什么，为什么有用
*   如何在**生产代码**中使用 Python `pass`语句
*   开发代码时如何使用 Python `pass`语句作为**辅助**
*   `pass` 的**替代品是什么，以及何时应该使用它们**

**免费奖励:** 并学习 Python 3 的基础知识，如使用数据类型、字典、列表和 Python 函数。

## Python `pass`语句:语法和语义

在 Python 语法中，新的缩进块后面跟一个冒号(`:`)。有几个地方会出现新的缩进块。当你开始写 Python 代码时，最常见的地方是在 [`if`关键字](https://realpython.com/python-conditional-statements/)之后和 [`for`关键字](https://realpython.com/python-for-loop/)之后:

>>>

```py
>>> for x in [1, 2, 3]:
...     y = x + 1
...     print(x, y)
...
1 2
2 3
3 4
```

在`for`语句之后是 [`for`循环](https://realpython.com/python-for-loop/)的**体**，由紧跟在冒号之后的两行缩进组成。

在这种情况下，主体中有两个语句对每个值重复:

1.  `y = x + 1`
2.  `print(x, y)`

在 Python 语法中，这种类型的块中的语句在技术上被称为[套件](https://docs.python.org/3/reference/compound_stmts.html)。一个套件必须包含一个或多个语句。不能是空的。

要在套件中不做任何事情，可以使用 Python 的特殊`pass`语句。这条语句只包含一个关键字`pass`。虽然您可以在 Python 的许多地方使用`pass`，但它并不总是有用的:

>>>

```py
>>> if 1 + 1 == 2:
...     print("math is ok")
...     pass
...     print("but this is to be expected")
...
math is ok
but this is to be expected
```

在这个`if`语句中，删除`pass`语句将保持功能不变，并使您的代码更短。您可能想知道为什么 Python 语法包含一个告诉解释器什么也不做的语句。你不写声明难道不能达到同样的效果吗？

在某些情况下，显式地告诉 Python 什么也不做有一个重要的目的。例如，因为`pass`语句不做任何事情，所以您可以使用它来满足一个套件至少包含一个语句的要求:

>>>

```py
>>> if 1 + 1 == 3:
...
  File "<stdin>", line 2

 ^
IndentationError: expected an indented block
```

即使您不想在`if`块中添加任何代码，没有语句的`if`块也会创建一个空套件，这是无效的 Python 语法。

要解决这个问题，您可以使用`pass`:

>>>

```py
>>> if 1 + 1 == 3:
...     pass
...
```

现在，多亏了`pass`，您的`if`语句是有效的 Python 语法。

[*Remove ads*](/account/join/)

## `pass`的临时用途

在开发过程中，有很多情况下`pass`会对你有用，即使它不会出现在你代码的最终版本中。就像脚手架一样，`pass`可以在你填充细节之前方便地支撑你程序的主要结构。

编写稍后会被删除的代码听起来可能很奇怪，但是这样做可以加速您的初始开发。

### 未来代码

在许多情况下，代码的结构需要或者可以使用块。虽然您最终可能不得不在那里编写代码，但有时很难摆脱特定工作的流程，并开始处理依赖关系。在这些情况下，`pass`语句是为依赖关系做最少量工作的有用方法，这样您就可以回到您正在做的事情上。

作为一个具体的例子，想象一下[编写一个函数](https://realpython.com/defining-your-own-python-function/)来处理一个字符串，然后将结果写入一个文件，[返回](https://realpython.com/python-return-statement/)它:

```py
def get_and_save_middle(data, fname):
    middle = data[len(data)//3:2*len(data)//3]
    save_to_file(middle, fname)
    return middle
```

这个函数保存并返回一个字符串的中间三分之一。在测试输出的[一位误差](https://en.wikipedia.org/wiki/Off-by-one_error)之前，你不需要完成`save_to_file()`的实现。然而，如果`save_to_file()`以某种形式不存在，那么你会得到一个错误。

可以注释掉对`save_to_file()`的调用，但是你必须记住在确认`get_and_save_middle()`运行良好后取消对调用的注释。相反，您可以用一个`pass`语句快速实现`save_to_file()`:

```py
def save_to_file(data, fname):
    pass # TODO: fill this later
```

这个函数不做任何事情，但是它允许你测试`get_and_save_middle()`没有错误。

`pass`的另一个用例是当你正在编写一个复杂的流控制结构，并且你想为将来的代码留一个占位符。例如，当用[模操作符](https://realpython.com/python-modulo-operator/)实现 [fizz-buzz 挑战](https://en.wikipedia.org/wiki/Fizz_buzz)时，首先理解代码的结构是有用的:

```py
if idx % 15 == 0:
    pass # Fizz-Buzz
elif idx % 3 == 0:
    pass # Fizz
elif idx % 5 == 0:
    pass # Buzz
else:
    pass # Idx
```

这个结构确定了每种情况下应该打印的内容，这为您提供了解决方案的框架。当试图找出需要哪些`if`语句的**分支逻辑**以及需要的顺序时，这样的结构框架是有用的。

例如，在这种情况下，一个关键的见解是第一个`if`语句需要检查可被`15`整除，因为任何可被`15`整除的数也会被`5`和`3`整除。不管具体输出的细节如何，这种结构上的洞察力都是有用的。

在你弄清楚了问题的核心逻辑之后，你就可以决定你是否会在代码中直接 [`print()`](https://realpython.com/python-print/) :

```py
def fizz_buzz(idx):
    if idx % 15 == 0:
        print("fizz-buzz")
    elif idx % 3 == 0:
        print("fizz")
    elif idx % 5 == 0:
        print("buzz")
    else:
        print(idx)
```

这个函数使用起来很简单，因为它直接打印字符串。然而，这不是一个令人愉快的测试功能。这可能是一个有用的权衡。但是，在编码面试中，面试官有时候会让你写测试。首先编写结构允许您在检查其他需求之前确保理解逻辑流程。

另一种方法是编写一个返回字符串的函数，然后在别处进行循环:

```py
def fizz_buzz(idx):
    if idx % 15 == 0:
        return "fizz-buzz"
    elif idx % 3 == 0:
        return "fizz"
    elif idx % 5 == 0:
        return "buzz"
    else:
        return str(idx)
```

这个函数将打印功能推上堆栈，更容易测试。

使用`pass`找出问题的核心条件和结构，可以更容易地决定实现应该如何工作。

这种方法在[编写类](https://realpython.com/python3-object-oriented-programming/#define-a-class-in-python)时也很有用。如果你需要写一个类来实现某个东西，但是你并没有完全理解问题域，那么你可以使用`pass`来首先理解对你的代码架构来说最好的布局。

例如，假设您正在实现一个`Candy`类，但是您需要的属性并不明显。最终，您将需要进行一些仔细的需求分析，但是在实现基本算法时，您可以清楚地看到该类还没有准备好:

```py
class Candy:
    pass
```

这允许您实例化该类的成员并传递它们，而不必决定哪些属性与该类相关。

[*Remove ads*](/account/join/)

### 注释掉代码

当您注释掉代码时，可以通过移除块中的所有代码来使语法无效。如果您有一个`if` … `else`条件，那么注释掉其中一个分支可能是有用的:

```py
def process(context, input_value):
    if input_value is not None:
        expensive_computation(context, input_value)
    else:
        logging.info("skipping expensive: %s", input_value)
```

在这个例子中，`expensive_computation()`运行需要很长时间的代码，比如将大数组的数字相乘。当您在[调试](https://realpython.com/python-debugging-pdb/)时，您可能需要暂时注释掉`expensive_computation()`调用。

例如，您可能想对一些有问题的数据运行这段代码，并通过检查日志中的描述来了解为什么有这么多不是 [`None`](https://realpython.com/null-in-python/) 的值。跳过有效值的昂贵计算将大大加快测试速度。

但是，这不是有效的代码:

```py
def process(context, input_value):
    if input_value is not None:
        # Temporarily commented out the expensive computation
        # expensive_computation(context, input_value)
    else:
        logging.info("skipping expensive: %s", input_value)
```

在这个例子中，`if`分支中没有任何语句。在解析过程的早期，在检查缩进以查看块的开始和结束位置之前，注释被剥离。

在这种情况下，添加一个`pass`语句会使代码有效:

```py
def process(context, input_value):
    if input_value is not None:
        # Temporarily commented out the expensive computation
        # expensive_computation(context, input_value)
        # Added pass to make code valid
        pass
    else:
        logging.info("skipping expensive: %s", input_value)
```

现在可以运行代码，跳过昂贵的计算，生成包含有用信息的日志。

在对行为进行故障排除时，部分注释掉代码在许多情况下是有用的。在类似上面例子的情况下，您可能会注释掉需要很长时间来处理并且不是问题根源的代码。

另一种情况是，在进行故障诊断时，您可能希望注释掉代码，这是因为被注释掉的代码有不良的副作用，比如[发送电子邮件](https://realpython.com/python-send-email/)或更新计数器。

类似地，有时在保留调用的同时注释掉整个函数是很有用的。如果您使用的库需要回调，那么您可以编写如下代码:

```py
def write_to_file(fname, data):
    with open(fname, "w") as fpout:
        fpout.write(data)

get_data(source).add_callback(write_to_file, "results.dat")
```

这段代码调用`get_data()`并给结果附加一个回调。

让测试运行丢弃数据以确保正确给出源代码可能是有用的。但是，这不是有效的 Python 代码:

```py
def write_to_file(fname, data):
    # Discard data for now
    # with open(fname, "w") as fpout:
    #     fpout.write(data)

get_data(source).add_callback(write_to_file, "results.dat")
```

由于函数块中没有语句，Python 无法解析这段代码。

再一次，`pass`可以帮助你:

```py
def write_to_file(fname, data):
    # Discard data for now
    # with open(fname, "w") as fpout:
    #     fpout.write(data)
    pass

get_data(source).add_callback(write_to_file, "results.dat")
```

这是有效的 Python 代码，它将丢弃数据并帮助您确认参数是否正确。

[*Remove ads*](/account/join/)

### 调试器的标记

当您在[调试器](https://realpython.com/python-debugging-pdb/)中运行代码时，可以在代码中设置一个**断点**，调试器将在此处停止，并允许您在继续之前检查程序状态。

当测试运行经常触发断点时，比如在循环中，可能会有很多程序状态不令人感兴趣的情况。为了解决这个问题，许多调试器还允许一个**条件断点**，一个只有当条件为真时才会触发的断点。例如，您可以在一个只有当变量为`None`时才触发的`for`循环中设置一个断点，以查看为什么这种情况没有被正确处理。

然而，许多调试器只允许在断点上设置一些基本条件，比如相等或者大小比较。你可能需要一个更复杂的条件，比如在断开之前检查一个字符串是否是一个回文。

虽然调试器可能无法检查回文，但 Python 可以轻而易举地做到。您可以通过一个什么都不做的`if`语句并在`pass`行上设置一个断点来利用该功能:

```py
for line in filep:
    if line == line[::-1]:
        pass # Set breakpoint here
    process(line)
```

通过用`line == line[::-1]`检查回文，现在你有了一行只有在条件为真时才执行的代码。

虽然`pass`行不做任何事情，但是它让你有可能在那里设置一个断点。现在，您可以在调试器中运行这段代码，并且只中断回文字符串。

### 空功能

在某些情况下，在代码的已部署版本中包含一个空函数可能会很有用。例如，库中的函数可能期望传入一个回调函数。

一个更常见的情况是当你的代码定义了一个类，而这个类[继承了一个期望方法被覆盖的类](https://realpython.com/inheritance-composition-python/)。然而，在你的具体情况下，你不需要做任何事情。或者您重写代码的原因可能是为了防止可重写的方法做任何事情。

在所有这些情况下，您都需要编写一个空的函数或方法。同样，问题是在`def`行之后没有行不是有效的 Python 语法:

>>>

```py
>>> def ignore_arguments(record, status):
...
  File "<stdin>", line 2

 ^
IndentationError: expected an indented block
```

这将失败，因为函数和其他块一样，必须至少包含一条语句。要解决这个问题，您可以使用`pass`:

>>>

```py
>>> def ignore_arguments(record, status):
...     pass
...
```

现在函数有了一个语句，即使它什么也不做，它也是有效的 Python 语法。

作为另一个例子，假设您有一个函数，它期望向一个类似于[文件的对象](https://realpython.com/working-with-files-in-python/)写入数据。但是，您出于另一个原因想要调用该函数，并且想要放弃输出。您可以使用`pass`编写一个丢弃所有数据的类:

```py
class DiscardingIO:
    def write(self, data):
        pass
```

这个类的实例支持`.write()`方法，但是会立即丢弃所有数据。

在这两个例子中，方法或函数的存在很重要，但是它不需要做任何事情。因为 Python 块必须有语句，所以可以通过使用`pass`使空函数或方法有效。

[*Remove ads*](/account/join/)

### 空类

在 Python 中，[异常继承](https://realpython.com/python-exceptions/)很重要，因为它标记了哪些异常被捕获。例如，内置异常`LookupError`是 [`KeyError`](https://realpython.com/python-keyerror/) 的父级。当在[字典](https://realpython.com/python-dicts/)中查找一个不存在的键时，会引发一个`KeyError`异常。这意味着你可以用`LookupError`来捕捉`KeyError`:

>>>

```py
>>> empty={}
>>> try:
...     empty["some key"]
... except LookupError as exc:
...     print("got exception", repr(exc))
...
got exception KeyError('some key')
>>> issubclass(KeyError, LookupError)
True
```

异常`KeyError`被捕获，即使`except`语句指定了`LookupError`。这是因为`KeyError`是`LookupError`的子类。

有时您希望在代码中引发特定的异常，因为它们有特定的恢复路径。但是，您希望确保这些异常继承自一般异常，以防有人捕获一般异常。这些异常类没有行为或数据。它们只是标记。

为了看到丰富的异常层次结构的有用性，您可以考虑密码规则检查。尝试在网站上更改密码之前，您需要在本地测试它所执行的规则:

*   至少八个字符
*   至少一个数字
*   至少一个特殊字符，如问号(`?`)、感叹号(`!`)或句号(`.`)。

**注意:**这个例子纯粹是为了说明 Python 的语义和技术。研究表明，密码复杂性规则不会增加安全性。

欲了解更多信息，请参见国家标准与技术研究所(NIST) [指南](https://pages.nist.gov/800-63-3/sp800-63b.html#appA)和它们所基于的[研究](http://www.cs.umd.edu/~jkatz/security/downloads/passwords_revealed-weir.pdf)。

这些错误中的每一个都应该有自己的异常。以下代码实现了这些规则:

```py
# password_checker.py
class InvalidPasswordError(ValueError):
    pass

class ShortPasswordError(InvalidPasswordError):
    pass

class NoNumbersInPasswordError(InvalidPasswordError):
    pass

class NoSpecialInPasswordError(InvalidPasswordError):
    pass

def check_password(password):
    if len(password) < 8:
        raise ShortPasswordError(password)
    for n in "0123456789":
        if n in password:
            break
    else:
        raise NoNumbersInPasswordError(password)
    for s in "?!.":
        if s in password:
            break
    else:
        raise NoSpecialInPasswordError(password)
```

如果密码不符合指定的规则，该函数将引发异常。一个更现实的例子是记录所有没有被遵守的规则，但这超出了本教程的范围。

您可以在包装器中使用这个函数以一种很好的方式打印异常:

>>>

```py
>>> from password_checker import check_password
>>> def friendly_check(password):
...     try:
...         check_password(password)
...     except InvalidPasswordError as exc:
...         print("Invalid password", repr(exc))
...
>>> friendly_check("hello")
Invalid password ShortPasswordError('hello')
>>> friendly_check("helloworld")
Invalid password NoNumbersInPasswordError('helloworld')
>>> friendly_check("helloworld1")
Invalid password NoSpecialInPasswordError('helloworld1')
```

在这种情况下，`friendly_check()`只捕获`InvalidPasswordError`，因为其他的`ValueError`异常可能是检查器本身的错误。它打印出异常的名称和值，显示出没有被遵循的规则。

在某些情况下，用户可能并不关心输入中存在哪些问题。在这种情况下，您可能只想抓住`ValueError`:

```py
def get_username_and_password(credentials):
    try:
        name, password = credentials.split(":", 1)
        check_password(password)
    except ValueError:
        return get_default_credentials()
    else:
        return name, value
```

在这段代码中，所有无效输入都被同等对待，因为您不关心凭证有什么问题。

由于这些不同的用例，`check_password()`需要所有四个例外:

1.  `InvalidPasswordError`
2.  `ShortPasswordError`
3.  `NoNumbersPasswordError`
4.  `NoSpecialPasswordError`

这些异常中的每一个都描述了被违反的不同规则。在根据更复杂的规则匹配字符串的代码中，可能有更多这样的规则，以复杂的结构排列。

尽管需要四个不同的类，但是没有一个类有任何行为。`pass`语句允许您快速定义所有四个类。

[*Remove ads*](/account/join/)

### 标记方法

类中的一些方法不是为了被调用而存在，而是为了将类标记为以某种方式与该方法相关联。

Python 标准库有 [`abc`模块](https://realpython.com/python-interface/)。模块的名字代表**抽象基类**。这个模块帮助定义了一些类，这些类不是用来实例化的，而是作为一些其他类的公共基础。

如果您正在编写代码来分析 web 服务器的使用模式，那么您可能希望区分来自登录用户的请求和来自未经身份验证的连接的请求。你可以通过一个有两个子类的`Origin`超类来建模:`LoggedIn`和`NotLoggedIn`。

不应该直接实例化`Origin`类。每个请求应该来自`LoggedIn`源或`NotLoggedIn`源。下面是一个极简实现:

```py
import abc

class Origin(abc.ABC):
    @abc.abstractmethod
    def description(self):
        # This method will never be called
        pass

class NotLoggedIn(Origin):
    def description(self):
        return "unauthenticated connection"

class LoggedIn(Origin):
    def description(self):
        return "authenticated connection"
```

虽然一个真正的`Origin`类会更复杂，但是这个例子展示了一些基本的东西。`Origin.description()`永远不会被调用，因为所有子类都必须覆盖它。

因为`Origin`有一个`abstractmethod`，所以不能实例化:

>>>

```py
>>> Origin()
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: Can't instantiate abstract class Origin with abstract...
>>> logged_in.description()
'authenticated connection'
>>> not_logged_in.description()
'unauthenticated connection'
```

不能实例化具有`abstractmethod`方法的类。这意味着任何将`Origin`作为超类的对象都将是覆盖`description()`的类的实例。正因为如此，`Origin.description()`中的 body 并不重要，但是方法需要存在才能表示所有子类都必须实例化它。

因为方法体不能是空的，你必须把*的东西*放在`Origin.description()`里。同样，什么都不做的语句`pass`是一个很好的选择，可以让人们清楚地看到，您只是出于语法原因而包含了这一行。

一种更现代的方法是使用 [`Protocol`](https://mypy.readthedocs.io/en/stable/protocols.html) ，这在 Python 3.8 及更高版本的标准库中可用。在旧的 Python 版本中，它可以通过`typing_extensions` backports 获得。

一个`Protocol`不同于一个抽象基类，因为它不与一个具体类显式关联。相反，它依靠类型匹配在[类型检查](https://realpython.com/python-type-checking/#static-type-checking)时间将其与 [`mypy`](https://mypy.readthedocs.io/en/stable/) 相关联。

永远不会调用`Protocol`中的方法。它们仅用于标记所需方法的类型:

>>>

```py
>>> from typing_extensions import Protocol
>>> class StringReader(Protocol):
...     def read(self, int) -> str:
...         pass
```

演示如何在`mypy`中像这样使用`Protocol`与`pass`语句无关。但是*看到方法的主体只有`pass`语句是很重要的。*

在 Python 语言和标准库之外，还有更多使用这种标记的例子。例如，它们在 [`zope.interface`](https://zopeinterface.readthedocs.io/en/latest/) 包中用来表示接口方法，在 [`automat`](https://automat.readthedocs.io/en/latest/) 中用来表示有限状态自动机的输入。

在所有这些情况下，类需要有方法，但永远不要调用它们。正因为如此，身体就不重要了。但是由于 body 不能为空，所以可以使用`pass`语句添加一个 body。

## `pass`的替代品

`pass`语句并不是在代码中不做任何事情的唯一方法。它甚至不是最短的，稍后你会看到。它甚至不总是最好的或最大的[python 式的](https://realpython.com/learning-paths/writing-pythonic-code/)方法。

Python 中任何一个[表达式](https://docs.python.org/3/reference/expressions.html)都是有效的[语句](https://docs.python.org/3/reference/simple_stmts.html)，每个常量都是有效的表达式。所以下面的表达式什么都不做:

*   `None`
*   `True`
*   `0`
*   `"hello I do nothing"`

您可以使用这些表达式中的任何一个作为一个套件中的唯一语句，它将完成与`pass`相同的任务。避免使用它们作为无所事事的陈述的主要原因是它们不符合外交辞令。当你使用它们时，阅读你的代码的人并不清楚它们为什么在那里。

一般来说，`pass`语句虽然比`0`需要更多的字符，但它是向未来的维护者传达代码块被故意留白的最好方式。

[*Remove ads*](/account/join/)

### 文档字符串

使用`pass`作为无为语句的习语有一个重要的例外。在类、函数和方法中，使用常量字符串表达式会导致表达式被用作对象的 [`.__doc__`属性](https://realpython.com/defining-your-own-python-function/#docstrings)。

交互式解释器中的`help()`和各种[文档生成器](https://realpython.com/documenting-python-code/#documentation-tools-and-resources)、许多[ide](https://realpython.com/python-ides-code-editors-guide/)以及其他读取代码的开发人员使用`.__doc__`属性。一些代码风格坚持在每个类、函数或方法中都有它。

即使 docstring 不是强制的，它通常也是空块中`pass`语句的一个很好的替代品。您可以修改本教程前面的一些例子，使用 docstring 代替`pass`:

```py
class StringReader(Protocol):
      def read(self, length: int) -> str:
          """
 Read a string
 """

class Origin(abc.ABC):
    @abc.abstractmethod
    def description(self):
        """
 Human-readable description of origin
 """

class TooManyOpenParens(ParseError):
    """
 Not all open parentheses were closed
 """

class DiscardingIO:
    def write(self, data):
        """
 Ignore data
 """
```

在所有这些情况下，docstring 使代码更加清晰。当您在交互式解释器和 ide 中使用这段代码时，docstring 也将是可见的，这使得它更有价值。

**注意**:上面的 docstrings 很简短，因为有几个类和函数。用于生产的 docstring 通常会更全面。

docstrings 的一个技术优势，尤其是对于那些从不执行的函数或方法，是它们不会被测试覆盖检查器标记为“未覆盖”。

### 省略号

在 [mypy 存根文件](https://mypy.readthedocs.io/en/stable/stubs.html#stub-file-syntax)中，填充块的推荐方式是使用省略号(`...`)作为常量表达式。这是一个计算结果为 [`Ellipsis`](https://realpython.com/python-ellipsis/) 的模糊常数:

>>>

```py
>>> ...
Ellipsis
>>> x = ...
>>> type(x), x
(<class 'ellipsis'>, Ellipsis)
```

内置`ellipsis`类的`Ellipsis` singleton 对象是由`...`表达式产生的真实对象。

最初使用`Ellipsis`是为了创建[多维切片](https://realpython.com/numpy-array-programming/)。但是，现在它也是在存根文件中填充套件的推荐语法:

```py
# In a `.pyi` file:
def add(a: int, b: int)-> int:
    ...
```

这个函数不仅什么都不做，而且它还在一个 Python 解释器从来不评估的文件中。

### 引发错误

在函数或方法因为从不执行而为空的情况下，有时对它们来说最好的主体是`raise NotImplementedError("this should never happen")`。虽然这在技术上确实有所作为，但它仍然是一个有效的替代`pass`语句的方法。

## `pass`的永久用途

有时,`pass`语句的使用不是临时的——它将保留在运行代码的最终版本中。在这些情况下，除了使用`pass`之外，没有更好的替代方法或更常见的习语来填充空块。

### 在异常捕捉中使用`pass`

当使用`try ... except`到[捕捉异常](https://realpython.com/python-exceptions/)时，你有时不需要对异常做任何事情。在这种情况下，您可以使用`pass`语句来消除错误。

如果你想确定一个文件不存在，那么你可以使用`os.remove()`。如果文件不存在，这个函数将产生一个错误。然而，在这种情况下，文件不在那里正是您想要的，所以错误是不必要的。

下面是一个删除文件的函数，如果文件不存在也不会失败:

```py
import os

def ensure_nonexistence(fname):
    try:
        os.remove(fname)
    except FileNotFoundError:
        pass
```

因为如果引发了一个`FileNotFoundError`就不需要做任何事情，所以可以使用`pass`来创建一个没有其他语句的块。

**注意:**在忽略异常时，一定要小心。异常通常意味着发生了意想不到的事情，需要进行一些恢复。在忽略异常之前，请仔细考虑导致异常的原因。

请注意，`pass`语句通常会被[日志语句](https://realpython.com/python-logging/)所取代。然而，如果错误是预料之中的并且很容易理解，就没有必要这样做。

在这种情况下，您也可以使用上下文管理器`contextlib.suppress()`来抑制错误。然而，如果您需要处理一些错误而忽略其他错误，那么更直接的方法是创建一个空的`except`类，除了`pass`语句之外什么都没有。

例如，如果您想让`ensure_nonexistence()`处理目录和文件，那么您可以使用这种方法:

```py
import os
import shutil

def ensure_nonexistence(fname):
    try:
       os.remove(fname)
    except FileNotFoundError:
       pass
    except IsADirectoryError:
       shutil.rmtree(fname)
```

这里，您在重试`IsADirectoryError`时忽略`FileNotFoundError`。

在这个例子中，`except`语句的顺序无关紧要，因为`FileNotFoundError`和`IsADirectoryError`是兄弟，并且都继承自`OSError`。如果有一个处理一般`OSError`的案例，也许通过记录和忽略它，那么顺序就很重要。在这种情况下，`FileNotFoundError`和它的`pass`语句必须在`OSError`之前。

[*Remove ads*](/account/join/)

### 在`if` … `elif`链条中使用`pass`

当你使用长`if` … `elif`链时，有时你不需要在一种情况下做任何事情。然而，你不能跳过这个`elif`,因为执行会继续到另一个条件。

想象一下，一位招聘人员厌倦了将“嘶嘶作响的挑战”作为面试问题，决定用一种扭曲的方式提问。这一次，规则有点不同:

*   如果数字能被 20 整除，那么打印`"twist"`。
*   否则，如果数字能被 15 整除，则不打印任何内容。
*   否则，如果数字能被 5 整除，则打印`"fizz"`。
*   否则，如果数字能被 3 整除，则打印`"buzz"`。
*   否则，打印号码。

面试官相信这种新的变化会让答案变得更有趣。

和所有的[编码面试问题](https://realpython.com/python-practice-problems/)一样，有很多方法可以解决这个挑战。但是有一种方法是使用一个带有链的`for`循环，模拟上面的描述:

```py
for x in range(100):
    if x % 20 == 0:
       print("twist")
    elif x % 15 == 0:
       pass
    elif x % 5 == 0:
       print("fizz")
    elif x % 3 == 0:
       print("buzz")
    else:
       print(x)
```

`if` … `elif`链反映了只有在前一个选项不起作用时才转向下一个选项的逻辑。

在这个例子中，如果您完全删除了`if x % 15`子句，那么您将改变行为。对于能被 15 整除的数字，不打印任何内容，而是打印`"fizz"`。即使在那种情况下无事可做，这一条款也是必不可少的。

`pass`语句的这个用例允许您避免重构逻辑，并保持代码以匹配行为描述的方式排列。

## 结论

您现在理解 Python `pass`语句的作用了。您已经准备好使用它来提高您的开发和调试速度，并在您的生产代码中巧妙地部署它。

**在本教程中，您已经学习了:**

*   Python **`pass`语句**是什么，为什么有用
*   如何在**生产代码**中使用 Python `pass`语句
*   开发代码时如何使用 Python `pass`语句作为**辅助**
*   `pass` 的**替代品是什么，以及何时应该使用它们**

现在，通过了解如何告诉 Python 什么也不做，您将能够编写更好、更高效的代码。*******