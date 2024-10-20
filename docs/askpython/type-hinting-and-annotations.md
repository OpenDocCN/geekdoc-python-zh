# Python 中的类型提示和注释

> 原文：<https://www.askpython.com/python/examples/type-hinting-and-annotations>

Python 是一种动态类型语言。我们不必明确提到声明的变量或函数的数据类型。Python 解释器在运行时根据变量值为变量分配类型。我们也有静态类型的语言，如 Java、C 或 C++，我们需要在声明时声明变量类型，而变量类型在编译时是已知的。

从 **Python 3.5** 开始，在 **PEP 484** 和 **PEP 483** 中引入了一种叫做**类型提示**的东西。Python 语言的这一新增功能有助于构建我们的代码，并使它看起来更像一种静态类型语言。这有助于避免错误，但同时也使代码更加冗长。

然而，Python 运行时并不强制函数和变量类型注释。它们可以被第三方工具使用，比如类型检查器、ide、linters 等等。

***亦读:[巨蟒中的魔蛊](https://www.askpython.com/python/oops/magic-methods)***

## 类型检查、类型提示和代码编译

起初，我们有一些外部第三方库，例如开始进行类型提示的**静态类型检查器**如 `**mypy**` ，mypy 的许多想法实际上被引入了规范 Python 本身并直接集成到 Python 中。

现在，关于类型提示的事情是，它不会修改 Python 本身的运行方式。类型提示确实与代码的其余部分一起被编译，但是它们不影响 Python 执行代码的方式。

让我们看一个例子，并通过给一个函数分配类型提示得到一个概述。

**代码:**

```py
def multiply(num1: int, num2: int):
    return num1 * num2

print(multiply(5, 7))  # 35
print(multiply(5, "Hi"))  # HiHiHiHiHi

```

**说明:**

在上面声明的函数中，我们将[内置数据类型](https://www.askpython.com/python/built-in-methods/python-built-in-functions-brief-overview)赋给参数。这是一个很好的普通函数，但是这里的语法有点不同。我们可以注意到，参数有一个分号，数据类型分配给它们 **`(num1: int, num2: int)`**

这个函数有两个参数 **`num1`** 和 `**num2**`，这是 Python 在运行代码时看到的。它需要两个变量。即使我们不放置任何类型提示，比如说 **`num1`** 和 **`num2`** 应该是整数**，Python 也不会有问题。**

**因此，根据它，我们应该传递两个整数值给我们的代码，这将工作得很好。然而，如果我们试图传递一个**整数**和一个**字符串**呢？**

**类型提示告诉我们传入`**int**` 值，然而我们传入的是`**str**`。当我们尝试运行代码时，它运行良好，没有任何问题。如果在像 `**int, str, dict,**`这样的类型提示中有一个**有效数据类型**，Python 解释器编译代码没有问题。**

## **为什么要使用类型提示呢？**

**在上面的例子中，我们看到，即使我们传递一个字符串值给它，代码也运行良好。Python 将`**int**` 乘以 **`str`** 没有问题。*然而，即使 Python 忽略了类型提示，还是有一些很好的理由使用它们。***

*   **其中之一是它帮助 ide 显示上下文相关的帮助信息，例如不仅是函数参数，还有预期的类型。**
*   **类型提示通常用于代码文档。有多个自动代码文档生成器在生成文档时会使用类型提示，例如，如果我们正在编写自己的代码库，其中包含许多函数和类以及注释。**
*   **即使 Python 根本不使用类型提示，它也有助于我们在编写代码时利用它来使用更具声明性的方法，并使用外部库来提供运行时验证。**

## **使用类型检查器**

**Python 有几个类型检查器。其中之一是我的小蜜蜂。**

**让我们使用在使用`**int**` 和 **`str`** 之前运行的相同代码。我们将使用静态类型检查器 **`mypy`** ，看看 Python 对我们的代码有什么看法。**

*   **安装`mypy`**

```py
`pip install mypy`
```

*   **带有类型提示的代码，在运行代码时使用类型检查器**

**代码:**

```py
def multiply(num1: int, num2: int):
    return num1 * num2

print(multiply(5, "Hi")) 
```

**在终端中运行带有类型检查器前缀的文件:**

```py
mypy type-hints.py 
```

**输出:**

```py
type-hints.py:9: error: Argument 2 to "multiply" has incompatible type "str"; expected "int"
Found 1 error in 1 file (checked 1 source file) 
```

**解释:**

**当我们用类型检查器运行我们的文件时，现在 *Python 解释器对我们的代码*有了问题。预期的参数是两个参数的 **`int`** 数据类型，我们在其中一个参数中传递了一个**字符串**。类型检查器跟踪了这个错误，并在我们的输出中显示出来。类型检查器帮助我们解决了代码中的问题。**

## **Python 中类型提示的更多示例**

**在上面的例子中，我们在提示时使用了`int` 和`str` 类型。*类似地，其他数据类型也可以用于类型提示*。我们也可以为函数中的 **`return` 类型**声明一个类型提示。**

**让我们浏览一下代码，看看一些例子。**

**示例:1**

```py
def some_function(a: int, b: float, c: dict, d: bool = False):
    pass 
```

**解释:**

**这里我们为我们的参数使用不同的数据类型。请注意，如果没有提供参数，我们也可以为参数分配默认值。**

**示例:2**

```py
def person(name: str, age: int) -> str:
    return f"{name} is a {age} year old programmer" 
```

**解释:**

**在上面的代码中，还声明了返回类型。当我们尝试使用 mypy 之类的类型检查器运行代码时，Python 运行代码不会有问题，因为我们有一个字符串作为返回类型，它与提供的类型提示相匹配。**

**示例:3**

```py
def other_function(name: str, age: int) -> None:
    return f"{name} is a {age} year old programmer" 
```

**解释:**

**该代码的返回类型为 **`None`** 。当我们试图使用`**mypy**` 类型检查器运行这段代码时，Python 会抛出一个异常，因为它期望的是 **`None`** 的`**return**` 类型，而*代码返回的是一个字符串。***

**示例:4**

```py
my_list: list = ["apple", "orange", "mango"]
my_tuple: tuple = ("Hello", "Friend")
my_dictionary: dict = {"name": "Peter", "age": 30}
my_set: set = {1, 4, 6, 7} 
```

**解释:**

**上面的代码显示了通常被称为**变量注释的类型提示。就像我们在上面的例子中为函数提供类型提示一样，即使是变量也可以保存类似的信息，帮助代码更具声明性和文档化。****

## ****`typing`** 模块**

**很多时候，我们有更高级或更复杂的类型，它们必须作为参数传递给函数。Python 有一个内置的`**typing**` **模块**，使我们能够编写这种类型的带注释的变量，使代码更加文档化。我们必须将输入模块导入到我们的文件中，然后使用这些函数。其中包括**链表、字典、集合、**和**元组**等数据结构。**

**让我们看看代码 a，获得一个概述，以及作为解释的注释。**

```py
from typing import List, Dict, Set, Tuple, Union

# Declaring a nested list with annotations
my_new_list: List[List[int]] = [[1, 2, 3], [4, 5]]

# Declaring a dictionary with keys as strings and values as integers
my_new_dict: Dict[str, int] = {"age": 34, "languages": 2}

# Declaring a set with only string values
my_new_set: Set[str] = {"apple", "mango"}	

# Declaring a tuple with exactly 3 parameters with string values only
my_new_tuple: Tuple[str, str, str] = ("Hello", "There", "Friend")

# Declaring a list that may hold integers, floats, and strings all at once
my_union_list: List[Union[int, float, str]] = [12, "notebook", 34.56, "pencil", 78]

'''
Since Python 3.10.x we can use the pipe | character 
instead of importing typing Module for using Union
'''
def func(a: str | int | float, b: int) -> str | int | float:
    return a * b 
```

## **摘要**

**在 Python 中还有许多其他利用类型提示的方法。使用类型不会影响代码的性能，我们也不会得到任何额外的功能。然而，使用**类型提示**为我们的代码提供了健壮性，并为以后阅读代码的人提供了文档。**

**这当然有助于避免引入难以发现的错误。在当前的技术场景中，编写代码时使用类型变得越来越流行，Python 也遵循这种模式，为我们提供了同样易于使用的函数。有关更多信息，请参考官方文档。**

## **参考**

**[Python 类型提示文档](https://docs.python.org/3/library/typing.html)**