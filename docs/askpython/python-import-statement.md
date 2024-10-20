# Python 导入语句

> 原文：<https://www.askpython.com/python/python-import-statement>

Python `**import**`语句使用户能够在相应的程序中导入特定的模块。

它类似于 C/C++中的#include header_file。

一旦解释器遇到特定代码中的导入语句，它就在局部范围内搜索相同的语句并导入模块(如果在搜索路径中存在的话)。

它首先在其内置模块部分搜索一个特定的模块。如果没有找到，它会在当前目录中搜索这些模块。

**模块在特定程序中只加载一次，不受模块导入次数的影响。**

**语法:**

```py
import module_name
```

**举例:**

```py
import collections

```

* * *

### 1.从模块导入类/函数

我们可以使用以下语法从模块中导入类/函数:

```py
from {module} import {class/function}
```

**举例:**

```py
from collections import OrderedDict
from os import path
from math import pi
print(pi)

```

**输出:**

```py
3.141592653589793
```

* * *

### 2.*导入** 语句

特定模块的所有方法和常量都可以使用 import *操作符导入。

```py
from math import *
print(pi)
print(floor(3.15))

```

**输出:**

```py
3.141592653589793
3
```

* * *

### 3.Python 的 *import as* 语句

`**import as**`语句帮助用户提供原始模块名的别名。

```py
# python import as
import math as M

print(M.pi)
print(M.floor(3.18))

```

**输出:**

```py
3.141592653589793
3
```

* * *

### 4.导入用户定义的模块

我们可以用一个程序的名字将它的功能导入到另一个程序中。

最初，我们需要创建一个 python 代码。

***test.py***

```py
def sub(a, b):
    return int(a) - int(b)

def lower_case(str1):
    return str(str1).lower()

```

然后创建另一个 python 脚本，其中我们需要导入上面的 create test.py 脚本。

***test2.py***

```py
import test

print(test.sub(5,4))
print(test.lower_case('SafA'))

```

**输出:**

```py
1
safa
```

* * *

### 5.从另一个目录导入

`**importlib**`库用于从另一个目录导入脚本。

最初，我们需要创建一个 python 脚本并在其中定义函数。

***test1.py***

```py
def sub(a, b):
    return int(a) - int(b)

def lower_case(str1):
    return str(str1).lower()

```

然后，我们将创建另一个 python 脚本，并将其保存到另一个目录中，然后从 test1.py(驻留在另一个目录中)导入功能。

***design . py***

```py
import importlib, importlib.util

def module_directory(name_module, path):
    P = importlib.util.spec_from_file_location(name_module, path)
    import_module = importlib.util.module_from_spec(P)
    P.loader.exec_module(import_module)
    return import_module

result = module_directory("result", "../inspect_module/test1.py")

print(result.sub(3,2))
print(result.lower_case('SaFa'))

```

**输出:**

```py
1
safa
```

另一种方法是将模块目录添加到 **sys.path** 列表中。

* * *

### 6.从另一个文件导入类

***tests . py***

```py
class Employee:
    designation = ""

    def __init__(self, result):
        self.designation = result

    def show_designation(self):
        print(self.designation)

class Details(Employee):
    id = 0

    def __init__(self, ID, name):
        Employee.__init__(self, name)
        self.id = name

    def get_Id(self):
        return self.id

```

***design . py***

```py
import importlib, importlib.util

def module_directory(name_module, path):
    P = importlib.util.spec_from_file_location(name_module, path)
    import_module = importlib.util.module_from_spec(P)
    P.loader.exec_module(import_module)
    return import_module

result = module_directory("result", "../Hello/tests.py")

a = result.Employee('Project Manager')
a.show_designation()

x = result.Details(4001,'Safa')
x.show_designation()
print(x.get_Id())

```

**输出:**

```py
Project Manager
Safa
Safa
```

* * *

## 结论

因此，在本文中，我们已经理解了 import 语句提供的功能。

* * *

## 参考

*   Python 导入语句
*   [进口声明文件](https://docs.python.org/3/reference/import.html)