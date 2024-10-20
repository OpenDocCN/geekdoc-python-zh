# Python 3.10 -简化了类型注释中的联合

> 原文：<https://www.blog.pythonlibrary.org/2021/09/11/python-3-10-simplifies-unions-in-type-annotations/>

Python 3.10 有几个新的类型特性。这里给出了它们的详细信息:

*   [**人教版 604**](https://www.python.org/dev/peps/pep-0604) ，允许将工会类型写成 X | Y
*   [**人教版 613**](https://www.python.org/dev/peps/pep-0613) ，显式别名
*   [**PEP 612**](https://www.python.org/dev/peps/pep-0612) ，参数说明变量

本教程的重点是谈论 [**PEP 604**](https://www.python.org/dev/peps/pep-0604) ，这使得在向您的代码库添加类型注释(又名:类型提示)时编写联合类型更加容易。

## 工会老叶道

在 Python 3.10 之前，如果你想说一个变量或参数可以有多种不同的类型，你需要使用 **Union** :

```py
from typing import Union

rate: Union[int, str] = 1
```

这是 Python 文档中的另一个例子:

```py
from typing import Union

def square(number: Union[int, float]) -> Union[int, float]:
    return number ** 2
```

让我们看看 3.10 将如何解决这个问题！

## 新联盟

在 Python 3.10 中，你根本不再需要导入 **Union** 。所有细节都在 [PEP 604](https://www.python.org/dev/peps/pep-0604/) 里。PEP 允许您用|操作符完全替换它。完成后，上面的代码如下所示:

```py
def square(number: int | float) -> int | float:
    return number ** 2
```

您也可以使用新的语法来替换可选的，这也是您在类型提示中使用的，参数可以是**None**:

```py
# Old syntax

from typing import Optional, Union

def f(param: Optional[int]) -> Union[float, str]:
   ...

# New syntax in 3.10

def f(param: int | None) -> float | str:
   ...
```

您甚至可以将此语法与 **isinstance()** 和 **issubclass()** 一起使用:

```py
>>> isinstance(1, int | str)
True
```

## 包扎

虽然这对于 Python 语言来说并不是一个令人惊奇的新特性，但是使用带有类型注释的|操作符使您的代码看起来更加整洁。你不需要进口那么多，这样看起来更好。

## 相关阅读

*   Python 3: [变量注释](https://www.blog.pythonlibrary.org/2017/10/31/python-3-variable-annotations/)
*   [Python 中的类型检查](https://www.blog.pythonlibrary.org/2020/04/15/type-checking-in-python/)