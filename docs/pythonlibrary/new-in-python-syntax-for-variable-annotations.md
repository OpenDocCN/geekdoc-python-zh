# Python 中的新特性:变量注释的语法

> 原文：<https://www.blog.pythonlibrary.org/2017/01/12/new-in-python-syntax-for-variable-annotations/>

Python 3.6 增加了另一个有趣的新特性，被称为变量注释的*语法。这个新特性在 [PEP 526](https://www.python.org/dev/peps/pep-0526) 中有所概述。这个 PEP 的基本前提是将类型提示( [PEP 484](https://www.python.org/dev/peps/pep-0484) )的思想带到它的下一个逻辑步骤，基本上是将选项类型定义添加到 Python 变量中，包括类变量和实例变量。请注意，添加这些注释或定义不会突然使 Python 成为静态类型语言。解释器仍然不关心变量是什么类型。但是，Python IDE 或其他实用程序(如 pylint)可以添加一个注释检查器，当您使用一个已经注释为一种类型的变量，然后通过在函数中间更改其类型而被错误使用时，该检查器会突出显示。*

让我们看一个简单的例子，这样我们就可以知道这是如何工作的:

```py

# annotate.py
name: str = 'Mike'

```

这里我们有一个 Python 文件，我们将其命名为 **annotate.py** 。在其中，我们创建了一个变量， **name** ，并对其进行了注释，表明它是一个字符串。这是通过在变量名后添加一个冒号，然后指定它应该是什么类型来实现的。如果你不想的话，你不必给变量赋值。以下内容同样有效:

```py

# annotate.py
name: str 

```

当您注释一个变量时，它将被添加到模块或类的 **__annotations__** 属性中。让我们尝试导入注释模块的第一个版本，并访问该属性:

```py

>>> import annotate
>>> annotate.__annotations__
{'name': }
>>> annotate.name
'Mike' 
```

如您所见， **__annotations__** 属性返回一个 Python dict，其中变量名作为键，类型作为值。让我们给我们的模块添加一些其他的注释，看看 **__annotations__** 属性是如何更新的。

```py

# annotate2.py
name: str = 'Mike'

ages: list = [12, 20, 32]

class Car:
    variable: dict

```

在这段代码中，我们添加了一个带注释的列表变量和一个带注释的类变量的类。现在让我们导入新版本的 annotate 模块，并检查它的 **__annotations__** 属性:

```py

>>> import annotate
>>> annotate.__annotations__
{'name': , 'ages': <class>}
>>> annotate.Car.__annotations__
{'variable': <class>}
>>> car = annotate.Car()
>>> car.__annotations__
{'variable': <class>}
```

这一次，我们看到注释字典包含两个条目。您会注意到模块级别的 **__annotations__** 属性不包含带注释的类变量。要访问它，我们需要直接访问 Car 类，或者创建一个 Car 实例，并以这种方式访问属性。

正如我的一个读者指出的，你可以通过使用类型模块使这个例子更符合 PEP 484。看一下下面的例子:

```py

# annotate3.py
from typing import Dict, List

name: str = 'Mike'

ages: List[int] = [12, 20, 32]

class Car:

    variable: Dict

```

让我们在解释器中运行这段代码，看看输出是如何变化的:

```py

import annotate

In [2]: annotate.__annotations__
Out[2]: {'ages': typing.List[int], 'name': str}

In [3]: annotate.Car.__annotations__
Out[3]: {'variable': typing.Dict}

```

您会注意到，现在大多数类型都来自于类型模块。

* * *

### 包扎

我发现这个新功能非常有趣。虽然我喜欢 Python 的动态特性，但在过去几年使用 C++后，我也看到了知道变量应该是什么类型的价值。当然，由于 Python 出色的内省支持，弄清楚一个对象的类型是微不足道的。但是这个新特性可以让静态检查器更好，也可以让你的代码更明显，特别是当你不得不回去更新一个几个月或几年都没用过的软件的时候。

* * *

### 附加阅读

*   PEP 526 - [变量注释的语法](https://www.python.org/dev/peps/pep-0526)
*   PEP 484 - [类型提示](https://www.python.org/dev/peps/pep-0484)
*   Python 3.6: [新功能](https://docs.python.org/3.6/whatsnew/3.6.html)