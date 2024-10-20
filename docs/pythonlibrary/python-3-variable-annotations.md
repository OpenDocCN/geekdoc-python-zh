# Python 3:变量注释

> 原文：<https://www.blog.pythonlibrary.org/2017/10/31/python-3-variable-annotations/>

Python 在 3.6 版本中增加了称为**变量注释**的语法。变量注释基本上是类型提示的增强，是在 Python 3.5 中引入的。变量注释背后的完整解释在 [PEP 526](https://www.python.org/dev/peps/pep-0526) 中解释。在本文中，我们将快速回顾一下类型提示，然后介绍新的变量注释语法。

### 什么是类型暗示？

Python 中的类型提示基本上是声明函数和方法中的参数具有某种类型。Python 并不强制类型“提示”，但是您可以使用像 [mypy](http://mypy-lang.org/) 这样的工具来强制类型提示，就像 C++在运行时强制类型声明一样。让我们看一个没有添加类型提示的普通函数:

```py

def add(a, b):
    return a + b

if __name__ == '__main__':
    add(2, 4)

```

这里我们创建了一个接受两个参数的 **add()** 函数。它将两个参数相加并返回结果。我们不知道的是我们需要传递给函数什么。我们可以向它传递整数、浮点数、链表或字符串，它很可能会工作。但是它会像我们预期的那样工作吗？让我们在代码中添加一些类型提示:

```py

def add(a: int, b: int) -> int:
    return a + b

if __name__ == '__main__':
    print(add(2, 4))

```

这里我们改变了函数的定义，以利用类型提示。您会注意到，现在参数被标记了它们应该是什么类型:

*   答:int
*   乙:int

我们还暗示了返回值，也就是“-> int”是什么。这意味着我们期望一个整数作为返回值。如果您尝试用几个字符串或一个浮点数和一个整数调用 add()函数，您不会看到错误。正如我所说的，Python 只允许您提示参数的类型，但并不强制执行。

让我们将代码更新如下:

```py

def add(a: int, b: int) -> int:
    return a + b

if __name__ == '__main__':
    print(add(5.0, 4))

```

如果你运行这个，你会看到它执行得很好。现在让我们使用 pip 安装 mypy:

```py

pip install mypy

```

现在我们有了 mypy，我们可以用它来确定我们是否正确地使用了我们的函数。打开一个终端，导航到保存上述脚本的文件夹。然后执行以下命令:

```py

mypy hints.py

```

当我运行这个命令时，我收到了以下输出:

```py

hints.py:5: error: Argument 1 to "add" has incompatible type "float"; expected "int"

```

如你所见，mypy 在我们的代码中发现了一个问题。我们为第一个参数传入了一个 float，而不是 int。您可以在持续集成服务器上使用 mypy，该服务器可以在提交提交到您的分支之前检查您的代码是否存在这些问题，或者在提交代码之前在本地运行它。

* * *

### 变量注释

假设您不仅想注释函数参数，还想注释正则变量。在 Python 3.5 中，您不能使用与函数参数相同的语法来实现这一点，因为这会引发一个 **SyntaxError** 。相反，您可能需要使用注释，但是现在 3.6 已经发布了，我们可以使用新的语法了！让我们看一个例子:

```py

from typing import List

def odd_numbers(numbers: List) -> List:
    odd: List[int] = []
    for number in numbers:
        if number % 2:
            odd.append(number)

    return odd

if __name__ == '__main__':
    numbers = list(range(10))
    print(odd_numbers(numbers))

```

在这里，他指定变量 **odd** 应该是一个整数列表。如果您对这个脚本运行 mypy，您将不会收到任何输出，因为我们做的一切都是正确的。让我们试着改变代码，添加一些整数以外的东西！

```py

from typing import List

def odd_numbers(numbers: List) -> List:
    odd: List[int] = []
    for number in numbers:
        if number % 2:
            odd.append(number)

    odd.append('foo')

    return odd

if __name__ == '__main__':
    numbers = list(range(10))
    print(odd_numbers(numbers))

```

这里我们添加了一个新的行，将一个字符串追加到整数列表中。现在，如果我们对这个版本的代码运行 mypy，我们应该会看到以下内容:

```py

hints2.py:9: error: Argument 1 to "append" of "list" has incompatible type "str"; expected "int"

```

再次重申，在 Python 3.5 中，您可以进行变量注释，但是您必须将注释放在注释中:

```py

# Python 3.6
odd: List[int] = []

# Python 3.5
odd = [] # type: List[int]

```

请注意，如果您将代码更改为使用变量注释语法的 Python 3.5 版本，mypy 仍然会正确标记错误。不过，您必须在井号后面指定“类型:”。如果你去掉它，它就不再是变量注释了。基本上，PEP 526 添加的所有内容都是为了使语法在整个语言中更加统一。

* * *

### 包扎

此时，无论您使用的是 Python 3.5 还是 3.6，您都应该有足够的信息开始在自己的代码中进行变量注释。我认为这是一个很好的概念，对于那些与更熟悉静态类型语言的人一起工作的程序员来说尤其有用。

* * *

### 相关阅读

*   Python 3: [对类型提示的介绍](https://www.blog.pythonlibrary.org/2016/01/19/python-3-an-intro-to-type-hinting/)
*   PEP 526 - [变量注释的语法](https://www.python.org/dev/peps/pep-0526/)