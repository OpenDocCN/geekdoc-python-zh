# Python 常量:提高代码的可维护性

> 原文：<https://realpython.com/python-constants/>

在编程中，术语**常量**指的是代表在程序执行期间不变的值的名称。常量是编程中的一个基本概念，Python 开发人员在很多情况下都会用到它们。然而，Python 没有定义常量的专用语法。实际上，Python 常量只是*从不改变*的**变量**。

为了防止程序员重新分配一个应该包含常量的名称，Python 社区采用了一种命名约定:*使用大写字母*。对于每一个 Pythonista 来说，知道什么是常量，以及为什么和什么时候使用它们是很重要的。

**在本教程中，您将学习如何:**

*   在 Python 中正确地定义常数
*   识别一些**内置常数**
*   使用常量来提高代码的可读性、可重用性、可维护性和可维护性
*   应用不同的方法**组织**和**管理项目中的**常量
*   在 Python 中使用几种技术使常量**成为严格常量**

通过学习定义和使用常量，您将极大地提高代码的可读性、可维护性和可重用性。

为了最大限度地从本教程中学习，您将需要 Python [变量](https://realpython.com/python-variables/)、[函数](https://realpython.com/defining-your-own-python-function/)、[模块、包](https://realpython.com/python-modules-packages/)和[名称空间](https://realpython.com/python-namespaces-scope/)的基础知识。你还需要知道 Python 中面向对象编程的基础知识。

**示例代码:** [点击此处下载示例代码](https://realpython.com/bonus/python-constants-code/)，向您展示如何在 Python 中使用常量。

## 理解常数和变量

**变量**和**常数**是计算机编程中两个历史性的基本概念。大多数编程语言都使用这些概念来操作数据，并以一种有效且符合逻辑的方式工作。

变量和常量可能会出现在每个项目、应用程序、库或您编写的其他代码中。问题是:实际中变量和常数是什么？

[*Remove ads*](/account/join/)

### 是什么变量

在数学中，变量被定义为一个符号，指的是*可以*随时间变化的值或量。在编程中，变量也是通常与包含值、对象或数据的内存地址相关联的符号或名称。与数学一样，编程变量的内容可以在定义它的代码执行期间改变。

变量通常有一个描述性的名字，这个名字以某种方式与目标值或对象相关联。这个目标值可以是任何数据类型。因此，您可以使用变量来表示[数字](https://realpython.com/python-numbers/)、[字符串](https://realpython.com/python-strings/)、序列、自定义对象等等。

您可以对变量执行两个主要操作:

1.  **访问**它的值
2.  给分配一个新值

在大多数编程语言中，您可以通过在代码中引用变量名来访问与变量关联的值。为了给一个给定的变量赋值，您将使用一个[赋值](https://realpython.com/python-variables/#variable-assignment)语句，它通常由变量名、赋值操作符和期望值组成。

在实践中，您会发现许多可以表示为变量的数量、数据和对象的例子。一些例子包括温度、速度、时间和长度。其他可以作为变量处理的数据例子包括一个[网络应用](https://realpython.com/python-web-applications/)的注册用户数量，一个[视频游戏](https://realpython.com/platformer-python-arcade/)的活跃角色数量，以及一个跑步者跑了多少英里。

### 是什么常数

数学也有**常数**的概念。这个术语指的是*永远不会*改变的值或量。在编程中，常量是指与在程序执行过程中从不改变的值相关联的名称。

就像变量一样，编程常量由两部分组成:一个名称和一个关联值。该名称将清楚地描述常数是什么。值是常数本身的具体表达。

与变量一样，与给定常数关联的值可以是任何数据类型。因此，您可以定义整数常量、浮点常量、字符常量、字符串常量等等。

在你定义了一个常量之后，它只允许你对它执行一个操作。您只能*访问*常量的值，但不能随时间改变它。这不同于变量，变量允许你访问它的值，也可以重新赋值。

您将使用常量来表示不会改变的值。在你的日常编程中，你会发现很多这样的价值观。一些例子包括光速、一小时的分钟数和项目根文件夹的名称。

### 为什么使用常数

在大多数编程语言中，当您在凌晨两点编码时，常量可以防止您在代码的某个地方意外更改它们的值，从而导致无法预料和难以调试的错误。常量还可以帮助您使代码更具可读性和可维护性。

在代码中使用常量而不是*直接使用它们的值*的一些优点包括:

| 优势 | 描述 |
| --- | --- |
| 提高可读性 | 在整个程序中代表给定值的描述性名称总是比基本值本身更易读、更明确。例如，一个名为`MAX_SPEED`的常数比具体的速度值本身更容易阅读和理解。 |
| 明确传达意图 | 大多数人会假设`3.14`可能指的是[π](https://en.wikipedia.org/wiki/Pi)常数。然而，使用`Pi`、`pi`或`PI`名称会比直接使用值更清楚地传达您的意图。这种做法将允许其他开发人员快速准确地理解您的代码。 |
| 更好的可维护性 | 常数使您能够在整个代码中使用相同的名称来标识相同的值。如果您需要更新常量的值，那么您不必更改该值的每个实例。你只需要在一个地方改变这个值:常量定义。这提高了代码的可维护性。 |
| 降低出错风险 | 在整个程序中表示给定值的常数比该值的几个显式实例更不容易出错。假设您根据目标计算对 Pi 使用不同的精度级别。您已经明确使用了每个计算所需精度的值。如果您需要更改一组计算的精度，那么替换这些值很容易出错，因为您最终可能会更改错误的值。为不同的精度级别创建不同的常量并在一个地方更改代码更安全。 |
| 减少调试需求 | 常量在程序的生命周期内保持不变。因为它们总是有相同的值，所以它们不会导致错误和缺陷。这个特性在小型项目中可能不是必需的，但是在有多个开发人员的大型项目中可能是至关重要的。开发人员不必花时间调试任何常量的当前值。 |
| 线程安全的数据存储 | 常量只能访问，不能写入。这个特性使它们成为线程安全的对象，这意味着几个线程可以同时使用一个常量，而没有破坏或丢失底层数据的风险。 |

正如您在本表中所了解到的，常量是编程中的一个重要概念，这是有道理的。它们可以让您的生活更加愉快，让您的代码更加可靠、可维护和可读。那么，什么时候应该使用常量呢？

### 当使用常量时

生活，尤其是科学，充满了不变的价值观的例子。一些例子包括:

*   **3.141592653589793** :用 *π* 表示的常数，英文拼写为 *Pi* ，表示圆的[周长](https://en.wikipedia.org/wiki/Circumference)与其直径的比值
*   **2.718281828459045** :用 *e* 表示的常数，称为[欧拉数](https://en.wikipedia.org/wiki/E_(mathematical_constant))，与[自然对数](https://en.wikipedia.org/wiki/Natural_logarithm)和[复利](https://en.wikipedia.org/wiki/Compound_interest)密切相关
*   3600 秒:一小时中的秒数，在大多数应用中被认为是恒定的，尽管有时会添加[闰秒](https://en.wikipedia.org/wiki/Leap_second)来解释地球自转速度的变化
*   **-273.15** :以摄氏度表示[绝对零度](https://en.wikipedia.org/wiki/Absolute_zero)的常数，相当于[开尔文](https://en.wikipedia.org/wiki/Kelvin)温标上的 0 开尔文

以上例子都是人们在生活和科学中常用的常量值。在编程中，您会经常发现自己在处理这些和许多其他类似的值，您可以将它们视为常量。

总之，用一个常量来表示一个量、数量、对象、参数或任何其他在生命周期中保持不变的数据。

[*Remove ads*](/account/join/)

## 在 Python 中定义自己的常量

到目前为止，您已经了解了常量在生活、科学和编程中的一般概念。现在是时候学习 Python 如何处理常量了。首先，你应该知道 Python 没有定义常量的专用语法。

换句话说，Python 没有严格意义上的常量。它只有变量，主要是因为它的动态性质。因此，要在 Python 中拥有一个常量，您需要定义一个永远不会改变的*变量，并通过避免对变量本身进行赋值操作来坚持这种行为。*

**注意:**在这一节，你将关注于*定义*你自己的常数。然而，Python 中内置了一些常量。稍后你会了解到他们[。](#exploring-other-constants-in-python)

那么，Python 开发人员如何知道一个给定的变量代表一个常量呢？Python 社区已经决定使用一个强大的**命名约定**来区分变量和常量。继续阅读，了解更多！

### 用户定义的常数

要告诉其他程序员给定的值应该被视为常量，您必须使用一个被广泛接受的常量标识符或名称的命名约定。如 [PEP 8](https://realpython.com/python-pep8/) 的[常量](https://peps.python.org/pep-0008/#constants)部分所述，你应该用大写字母写名字，并用下划线分隔单词。

以下是用户定义 Python 常量的几个例子:

```py
PI = 3.14
MAX_SPEED = 300
DEFAULT_COLOR = "\033[1;34m"
WIDTH = 20
API_TOKEN = "593086396372"
BASE_URL = "https://api.example.com"
DEFAULT_TIMEOUT = 5
ALLOWED_BUILTINS = ("sum", "max", "min", "abs")
INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    ...
]
```

请注意，您已经像创建变量一样创建了这些常量。您已经使用了一个描述性的名称、赋值操作符(`=`)和常量的具体值。

通过只使用大写字母，你在传达这样一种信息，即当前的名字应该被视为一个常量——或者更准确地说，是一个永不改变的变量。因此，其他 Python 开发人员会知道这一点，并且希望不会对手头的变量执行任何赋值操作。

**注意:**同样，Python 不支持常量或不可重新分配的名称。使用大写字母只是一种约定，并不妨碍开发人员给你的常量赋新值。因此，任何从事代码工作的程序员都需要小心，永远不要编写改变常量值的代码。记住这条规则，因为你也需要遵守它。

因为 Python 常量只是变量，所以两者都遵循相似的命名规则，唯一的区别是常量只使用大写字母。按照这个想法，常量的名称可以:

*   长度不限
*   由大写字母(`A`–`Z`)组成
*   包括数字(`0`–`9`)，但不作为第一个字符
*   使用下划线字符(`_`)来分隔单词或作为它们的第一个字符

使用大写字母使你的常量从变量中脱颖而出。通过这种方式，其他开发人员将清楚地认识到他们的目的。

作为一般的命名建议，在定义常数时避免缩写名称。常量名称的目的是*阐明常量值的含义*，以便您以后可以重用它。这个目标需要描述性的名称。避免使用单字母名称、不常见的缩写和通用名称，如`NUMBER`或`MAGNITUDE`。

推荐的做法是在任何 [`import`](https://realpython.com/python-import/) 语句之后的任何`.py`文件的顶部定义常数。这样，阅读您的代码的人将立即知道常量的用途和预期的处理。

### 模块级数据常量

[模块级数据名](https://peps.python.org/pep-0008/#module-level-dunder-names)是以双下划线开始和结束的特殊名称。一些例子包括诸如`__all__`、`__author__`和`__version__`的名字。在 Python 项目中，这些名称通常被视为常量。

**注:**在 Python 中，一个 **dunder 名字**是一个有特殊含义的名字。它以双下划线开始和结束，单词 *dunder* 是**d**double**在** score 下的[组合词](https://en.wikipedia.org/wiki/Portmanteau)。

根据 Python 的编码风格指南， [PEP 8](https://peps.python.org/pep-0008/) ，模块级数据名称应该出现在模块的 [docstring](https://realpython.com/documenting-python-code/) 之后，任何`import`语句之前，除了`__future__` imports。

下面是一个示例模块，其中包括一组 dunder 名称:

```py
# greeting.py

"""This module defines some module-level dunder names."""

from __future__ import barry_as_FLUFL

__all__ = ["greet"]
__author__ = "Real Python"
__version__ = "0.1.0"

import sys

def greet(name="World"):
    print(f"Hello, {name}!")
    print(f"Greetings from version: {__version__}!")
    print(f"Yours, {__author__}!")
```

在这个例子中，`__all__`预先定义了当您在代码中使用`from module import *` import 构造时 Python 将导入的名称列表。在这种情况下，用通配符导入的人导入`greeting`将只是取回`greet()`函数。他们将无法访问`__author__`、`__version__`以及`__all__`上未列出的其他名称。

**注意:**`from module import *`构造允许您一次性导入给定模块中定义的所有名称。属性将导入的名字限制在底层列表中。

Python 社区强烈[不鼓励](https://peps.python.org/pep-0008/#imports)这种`import`构造，通常被称为**通配符导入**，因为它会使您当前的[名称空间](https://realpython.com/python-namespaces-scope/)中塞满您可能不会在代码中使用的名称。

相反，`__author__`和`__version__`只对代码的作者和用户有意义，而对代码的逻辑本身没有意义。这些名称应该被视为常量，因为在程序执行期间，不允许任何代码更改作者或版本。

注意,`greet()`函数确实访问了数据名称，但并没有改变它们。下面是`greet()`在实践中的工作方式:

>>>

```py
>>> from greeting import *

>>> greet()
Hello, World!
Greetings from version: 0.1.0!
Yours, Real Python!
```

一般来说，没有硬性规定阻止你定义自己的模块级数据名。然而，Python 文档强烈警告不要使用除了那些被社区普遍接受和使用的名字之外的名字。核心开发人员将来可能会在没有任何警告的情况下向该语言引入新的数据名称。

[*Remove ads*](/account/join/)

## 将常量付诸实施

到目前为止，您已经了解了常量及其在编程中的作用和重要性。您还了解到 Python 不支持严格常量。这就是为什么你可以把常数看成是永远不变的变量。

在接下来的几节中，您将编写一些例子来说明常量在日常编码工作中的价值。

### 替换幻数以提高可读性

在编程中，术语[幻数](https://en.wikipedia.org/wiki/Magic_number_(programming))指的是直接出现在你的代码中，没有任何解释的任何数字。它是一个突如其来的值，使你的代码变得神秘而难以理解。幻数也使得程序可读性更差，更难维护和更新。

例如，假设您有以下函数:

```py
def compute_net_salary(hours):
    return hours * 35 * (1 - (0.04 + 0.1))
```

你能预先告诉我这个计算中每个数字的含义吗？大概不会。这个函数中的不同数字是幻数，因为你不能从数字本身可靠地推断出它们的含义。

查看此函数的以下[重构版本](https://realpython.com/python-refactoring/):

```py
HOURLY_SALARY = 35 SOCIAL_SECURITY_TAX_RATE = 0.04 FEDERAL_TAX_RATE = 0.10 
def compute_net_salary(hours):
    return (
        hours
 * HOURLY_SALARY * (1 - (SOCIAL_SECURITY_TAX_RATE + FEDERAL_TAX_RATE))    )
```

有了这些小的更新，你的函数现在读起来很有魅力。您和其他任何阅读您的代码的开发人员肯定能知道这个函数是做什么的，因为您已经用适当命名的常数替换了原来的幻数。每个常数的名称都清楚地解释了其对应的含义。

每当你发现自己在使用一个神奇的数字时，花点时间用一个常数来代替它。这个常量的名称必须是描述性的，并且清楚地解释目标幻数的含义。这种做法会自动提高代码的可读性。

### 重用可维护性对象

常量的另一个日常使用案例是当一个给定值在代码的不同部分重复出现时。如果您在代码中每个需要的地方插入具体的值，那么如果您出于任何原因需要更改该值，您将会遇到麻烦。在这种情况下，您需要更改每个地方的值。

一次在多个地方更改目标值容易出错。即使您依赖于编辑器的*查找和替换*特性，您也可以留下一些值的未更改实例，这可能会导致以后出现意外的错误和奇怪的行为。

为了防止这些恼人的问题，您可以用一个正确命名的常数来替换该值。这将允许您设置一次该值，并根据需要在任意多个位置重复该值。如果你需要改变常量的值，那么你只需要在一个地方改变它:常量定义。

例如，假设您正在编写一个`Circle`类，您需要一些方法来计算圆的面积、周长等等。在几分钟的编码之后，您最终得到了下面的类:

```py
# circle.py

class Circle:
    def __init__(self, radius):
        self.radius = radius

    def area(self):
        return 3.14 * self.radius**2

    def perimeter(self):
        return 2 * 3.14 * self.radius

    def projected_volume(self):
        return 4/3 * 3.14 * self.radius**3

    def __repr__(self):
        return f"{self.__class__.__name__}(radius={self.radius})"
```

这个例子揭示了圆周率的近似值(`3.14`)是如何在你的`Circle`类的几个方法中被写成一个幻数的。为什么这种做法是一个问题？比如你需要提高圆周率的精度。然后，您将不得不在至少三个不同的地方手动更改该值，这既繁琐又容易出错，使得您的代码难以维护。

**注:**一般不需要自己定义 Pi。Python 附带了一些内置常量，包括 Pi。稍后你会看到如何利用它。

使用一个命名的常量来存储 Pi 的值是解决这些问题的一个很好的方法。下面是上述代码的增强版本:

```py
# circle.py

PI = 3.14 
class Circle:
    def __init__(self, radius):
        self.radius = radius

    def area(self):
 return PI * self.radius**2 
    def perimeter(self):
 return 2 * PI * self.radius 
    def projected_volume(self):
 return 4/3 * PI * self.radius**3 
    def __repr__(self):
        return f"{self.__class__.__name__}(radius={self.radius})"
```

这个版本的`Circle`用全局常数`PI`代替幻数。与原始代码相比，这段代码有几个优点。如果你需要增加圆周率的精度，那么你只需要更新文件开头的`PI`常量的值。这一更新将立即反映在代码的其余部分，而不需要您进行任何额外的操作。

**注意:**常量不应该在代码执行期间改变。但是，在开发过程中，您可以根据需要更改和调整您的常数。在您的`Circle`类中更新 Pi 的精度是一个很好的例子，说明了为什么您可能需要在代码开发期间更改常量的值。

另一个好处是，现在你的代码可读性更强，更容易理解。常数的名称不言自明，反映了公认的数学术语。

一次声明一个常量，然后多次重用它，就像您在上面的例子中所做的那样，这代表了一个显著的可维护性改进。如果您必须更新常量的值，那么您将在一个地方更新它，而不是在多个地方，这意味着更少的工作和错误风险。

[*Remove ads*](/account/join/)

### 提供默认参数值

使用命名常量为函数、方法和类提供默认参数值是 Python 中的另一种常见做法。在 Python 标准库中有很多这种实践的例子。

例如， [`zipfile`](https://realpython.com/python-zipfile/) 模块提供了创建、读取、写入、追加和列出 ZIP 文件的工具。这个模块最相关的类是 [`ZipFile`](https://docs.python.org/3/library/zipfile.html#zipfile.ZipFile) 。有了`ZipFile`，你可以高效快速地操作你的 ZIP 文件。

`ZipFile`的[类构造函数](https://realpython.com/python-class-constructor/)接受一个名为`compression`的参数，它允许你在一些可用的数据压缩方法中进行选择。这个参数是[可选的](https://realpython.com/python-optional-arguments/)，并且将 [`ZIP_STORED`](https://docs.python.org/3/library/zipfile.html#zipfile.ZIP_STORED) 作为其默认值，这意味着默认情况下`ZipFile`不会压缩输入数据。

在这个例子中，`ZIP_STORED`是在`zipfile`中定义的常数。该常数保存未压缩数据的数值。例如，您还会发现其他压缩方法，这些方法由命名的常数表示，如用于 [Deflate](https://en.wikipedia.org/wiki/Deflate) 压缩算法的 [`ZIP_DEFLATED`](https://docs.python.org/3/library/zipfile.html#zipfile.ZIP_DEFLATED) 。

`ZipFile`类构造函数中的`compression`参数是一个很好的例子，当您的参数只能接受有限数量的有效值时，可以使用常量来提供默认的参数值。

常量作为默认参数值很方便的另一个例子是当您有几个带有循环参数的函数时。假设您正在开发一个连接到本地 [SQLite](https://realpython.com/python-sqlite-sqlalchemy/) 数据库的应用程序。您的应用程序使用以下一组函数来管理数据库:

```py
import sqlite3
from sqlite3 import Error

def create_database(db_path):
    # Code to create the initial database goes here...

def create_connection(db_path):
    # Code to create a database connection goes here...

def backup_database(db_path):
    # Code to back up the database goes here...
```

这些函数对 SQLite 数据库执行不同的操作。注意，所有的函数都共享`db_path`参数。

在开发应用程序时，您决定为函数提供一个默认的数据库路径，以便可以快速测试它们。在这种情况下，您可以直接使用路径作为`db_path`参数的默认值。

但是，最好使用命名常量来提供默认的数据库路径:

```py
import sqlite3
from sqlite3 import Error

DEFAULT_DB_PATH = "/path/to/database.sqlite" 
def create_database(db_path=DEFAULT_DB_PATH):
    # Code to create the initial database goes here...

def create_connection(db_path=DEFAULT_DB_PATH):
    # Code to create a database connection goes here...

def backup_database(db_path=DEFAULT_DB_PATH):
    # Code to back up the database goes here...
```

这个小小的更新使您能够在开发过程中针对一个示例数据库快速测试您的应用程序。它还提高了代码的可维护性，因为您可以在应用程序的未来版本中出现的其他数据库相关函数中重用该常量。

最后，您会发现一些情况，您希望将具有特定行为的对象传递给类、方法或函数。这种实践通常被称为 [duck typing](https://realpython.com/python-type-checking/#duck-typing) ，是 Python 中的一个基本原则。现在假设您的代码将负责提供所需对象的标准实现。如果你的用户想要一个自定义对象，那么他们应该自己提供。

在这种情况下，您可以使用一个常数来定义默认对象，然后将该常数作为默认参数值传递给目标类、方法或函数。看看下面这个假想的`FileReader`类的例子:

```py
# file_handler.py

from readers import DEFAULT_READER

class FileHandler:
    def __init__(self, file, reader=DEFAULT_READER):
        self._file = file
        self._reader = reader

    def read(self):
        self._reader.read(self._file)

    # FileHandler implementation goes here...
```

这个类提供了一种操作不同类型文件的方法。`.read()`方法使用注入的`reader`对象根据其特定格式读取输入的`file`。

下面是一个 reader 类的玩具实现:

```py
# readers.py

class _DefaultReader:
    def read(self, file):
        with open(file, mode="r", encoding="utf-8") as file_obj:
            for line in file_obj:
                print(line)

DEFAULT_READER = _DefaultReader()
```

本例中的`.read()`方法获取一个文件的路径，打开它，并将其内容逐行打印到屏幕上。这个类将扮演默认读者的角色。最后一步是创建一个常量`DEFAULT_READER`，用来存储默认阅读器的实例。就是这样！您有一个处理输入文件的类，还有一个提供默认阅读器的助手类。

您的用户也可以编写自定义阅读器。例如，他们可以为 [CSV](https://realpython.com/python-csv/) 和 [JSON](https://realpython.com/python-json/) 文件编写代码阅读器。一旦他们编写了一个给定的阅读器，他们可以将它传递给`FileHandler`类构造函数，并使用产生的实例来处理使用阅读器的目标文件格式的文件。

[*Remove ads*](/account/join/)

## 在真实项目中处理您的常量

既然您已经知道了如何在 Python 中创建常量，那么是时候学习如何在实际项目中处理和组织它们了。为此，您可以使用几种方法或策略。例如，您可以将常数放入:

*   与使用它们的代码相同的**文件**
*   用于项目范围常量的专用模块
*   一个**配置文件**
*   一些**环境变量**

在接下来的几节中，您将编写一些实际的例子来演示上述适当管理常量的策略。

### 将常量与相关代码放在一起

组织和管理常量的第一个也可能是最自然的策略是将它们和使用它们的代码一起定义。使用这种方法，您将在包含相关代码的模块顶部定义常数。

例如，假设您正在创建一个自定义模块来执行计算，您需要使用数学常数，如圆周率、欧拉数等。在这种情况下，您可以这样做:

```py
# calculations.py

"""This module implements custom calculations."""

# Imports go here...
import numpy as np

# Constants go here...
PI = 3.141592653589793
EULER_NUMBER = 2.718281828459045
TAU = 6.283185307179586

# Your custom calculations start here...
def circular_land_area(radius):
    return PI * radius**2

def future_value(present_value, interest_rate, years):
    return present_value * EULER_NUMBER ** (interest_rate * years)

# ...
```

在这个例子中，您在使用它们的代码所在的同一个模块中定义您的常量。

**注意:**如果你想明确地表明一个常量应该只在它的包含模块中使用，那么你可以在它的名字前面加上一个下划线(`_`)。比如可以做`_PI = 3.141592653589793`这样的事情。这个前导下划线将这个名字标记为[非公共](https://peps.python.org/pep-0008/#method-names-and-instance-variables)，这意味着用户的代码不应该直接使用这个名字。

对于仅与给定项目中的单个模块相关的窄范围常量，将常量与使用它们的代码放在一起是一种快速而合适的策略。在这种情况下，您可能不会在包含模块本身之外使用常量。

### 为常量创建专用模块

组织和管理常量的另一个常见策略是创建一个专用模块来存放它们。这种策略适用于在给定项目的许多模块甚至包中使用的常量。

这种策略的中心思想是为常量创建一个直观且唯一的名称空间。要将此策略应用于您的计算示例，您可以创建包含以下文件的 Python 包:

```py
calc/
├── __init__.py
├── calculations.py
└── constants.py
```

`__init__.py`文件将把`calc/`目录变成一个 Python 包。然后您可以将以下内容添加到您的`constants.py`文件中:

```py
# constants.py

"""This module defines project-level constants."""

PI = 3.141592653589793
EULER_NUMBER = 2.718281828459045
TAU = 6.283185307179586
```

一旦您将这段代码添加到`constants.py`，那么您就可以在需要使用任何常量时导入模块:

```py
# calculations.py

"""This module implements custom calculations."""

# Imports go here...
import numpy as np

from . import constants 
# Your custom calculations start here...
def circular_land_area(radius):
 return constants.PI * radius**2 
def future_value(present_value, interest_rate, years):
 return present_value * constants.EULER_NUMBER ** (interest_rate * years) 
# ...
```

注意，您使用[相对导入](https://realpython.com/absolute-vs-relative-python-imports/#relative-imports)直接从`calc`包中导入`constants`模块。然后，使用完全限定名来访问计算中所需的任何常数。这种练习可以改善你的意图交流。现在完全清楚了，`PI`和`EULER_NUMBER`在您的项目中是常量，因为有了`constants`前缀。

要使用你的`calculations`模块，你可以这样做:

>>>

```py
>>> from calc import calculations
>>> calculations.circular_land_area(100)
31415.926535897932

>>> from calc.calculations import circular_land_area
>>> circular_land_area(100)
31415.926535897932
```

现在你的`calculations`模块存在于`calc`包中。这意味着如果你想使用`calculations`中的功能，那么你需要从`calc`中导入`calculations`。您也可以像在上面的第二个例子中一样，通过引用包和模块来直接导入函数。

[*Remove ads*](/account/join/)

### 在配置文件中存储常数

现在假设您想更进一步，将一个给定项目的常量外部化。您可能需要在项目的源代码中保留所有的常量。为此，您可以使用外部**配置文件**。

以下是如何将常数移动到配置文件中的示例:

```py
; constants.ini [CONSTANTS] PI=3.141592653589793 EULER_NUMBER=2.718281828459045 TAU=6.283185307179586
```

该文件使用 [INI 文件](https://en.wikipedia.org/wiki/INI_file)格式。您可以使用标准库中的`configparser`模块读取这种类型的文件。

现在回到`calculations.py`并更新它，如下所示:

```py
# calculations.py

"""This module implements custom calculations."""

# Imports go here...
from configparser import ConfigParser 
import numpy as np

constants = ConfigParser() constants.read("path/to/constants.ini") 
# Your custom calculations start here...
def circular_land_area(radius):
 return float(constants.get("CONSTANTS", "PI")) * radius**2 
def future_value(present_value, interest_rate, years):
    return (
 present_value * float(constants.get( "CONSTANTS", "EULER_NUMBER" ))) ** (interest_rate * years) 
# ...
```

在本例中，您的代码首先读取配置文件，并将结果`ConfigParser`对象存储在全局变量`constants`中。您也可以将这个变量命名为`CONSTANTS`，并将其作为常量全局使用。然后更新计算，从配置对象本身读取常数。

注意，`ConfigParser`对象将配置参数存储为字符串，因此需要使用内置的`float()`函数将值转换为数字。

例如，当你创建一个[图形用户界面(GUI)应用程序](https://realpython.com/python-pyqt-gui-calculator/)并需要设置一些参数来定义加载和显示 GUI 时应用程序窗口的形状和大小时，这种策略可能是有益的。

### 将常量作为环境变量处理

另一个处理常量的有用策略是，如果你在 Windows 上，将它们定义为**系统变量**，如果你在 macOS 或 Linux 上，将它们定义为**环境变量**。

这种方法通常用于在不同的环境中配置[部署](https://12factor.net/config)。您还可以将环境变量用于暗示安全风险的常量，并且不应该直接提交给源代码。这些常量类型的示例包括身份验证凭证、API 访问令牌等。

**注意:**在使用敏感信息的环境变量时，你应该[小心](https://blog.diogomonica.com/2017/03/27/why-you-shouldnt-use-env-variables-for-secret-data/)，因为它们可能会意外地暴露在日志或子进程中。所有的云提供商都提供某种更安全的[秘密管理](https://kubernetes.io/docs/concepts/configuration/secret/)。

要使用这种策略，首先必须将常量导出为操作系统中的环境或系统变量。至少有两种方法可以做到这一点:

1.  手动导出当前 [shell](https://en.wikipedia.org/wiki/Shell_(computing)) 会话中的常量
2.  将常量添加到 shell 的配置文件中

第一种技术非常快速和实用。您可以使用它对您的代码运行一些快速测试。例如，假设您需要导出一个 API 令牌作为系统或环境变量。在这种情况下，您只需运行以下命令:

*   [*视窗*](#windows-1)
**   [**Linux + macOS**](#linux-macos-1)*

```py
C:\> set API_TOKEN="593086396372"
```

```py
$ export API_TOKEN="593086396372"
```

这种技术的主要缺点是，您的常量只能从定义它们的命令行会话中访问。一个好得多的方法是让操作系统在您启动命令行窗口时加载这些常量。

如果你在 Windows 上，那么查看[你的 Python 编码环境在 Windows 上:设置指南](https://realpython.com/python-coding-setup-windows/)中的[配置环境变量](https://realpython.com/python-coding-setup-windows/#configuring-environment-variables)部分，学习如何创建系统变量。遵循本指南中的说明，添加一个值为`593086396372`的`API_TOKEN`系统变量。

如果您使用的是 Linux 或 macOS，那么您可以转到您的主文件夹并打开您的 shell 的配置文件。打开该文件后，在文件末尾添加下面一行:

*   [*Linux*](#linux-2)
**   [*macOS*](#macos-2)*

```py
# .bashrc

export API_TOKEN="593086396372"
```

```py
# .zshrc

export API_TOKEN="593086396372"
```

每当您启动终端或命令行窗口时，Linux 和 macOS 都会自动加载相应的 shell 配置文件。这样，您可以确保`API_TOKEN`变量在您的系统中始终可用。

一旦为 Python 常量定义了所需的环境变量，就需要将它们加载到代码中。为此，可以使用 Python 的 [`os`](https://docs.python.org/3/library/os.html#module-os) 模块中的 [`environ`](https://docs.python.org/3/library/os.html#os.environ) 字典。`environ`的键和值是分别代表环境变量及其值的字符串。

您的`API_TOKEN`常量现在出现在`environ`字典中。因此，您可以用两行代码从那里读取它:

>>>

```py
>>> import os

>>> os.environ["API_TOKEN"]
'593086396372'
```

使用环境变量存储常数，并使用`os.environ`字典将它们读入代码，这是配置常数的有效方法，这些常数依赖于应用程序部署的环境。这在使用云时特别有用，所以将这种技术放在您的 Python 工具包中。

[*Remove ads*](/account/join/)

## 探索 Python 中的其他常量

除了用户定义的常量之外，Python 还定义了几个可以被视为常量的内部名称。其中一些名称是严格的常量，这意味着一旦解释器运行，就不能更改它们。此例为 [`__debug__`](https://realpython.com/python-assert-statement/#understanding-the-__debug__-built-in-constant) 常数为例。

在接下来的几节中，您将了解一些内部 Python 名称，您可以考虑并应该在代码中将其视为常量。首先，您将回顾一些内置常量和常量值。

### 内置常数

根据 Python 文档，“少量常量存在于内置名称空间中”( [Source](https://docs.python.org/3/library/constants.html) )。文档中列出的前两个常量是`True`和`False`，它们是 Python [布尔值](https://realpython.com/python-boolean/)。这两个值也是`int`的实例。`True`的值为`1`，而`False`的值为`0`:

>>>

```py
>>> True
True
>>> False
False

>>> isinstance(True, int)
True
>>> isinstance(False, int)
True

>>> int(True)
1
>>> int(False)
0

>>> True = 42
 ...
SyntaxError: cannot assign to True

>>> True is True
True
>>> False is False
True
```

请注意，`True`和`False`名称是严格的常量。换句话说，它们不能被重新分配。如果你试图重新分配它们，那么你会得到一个 [`SyntaxError`](https://realpython.com/invalid-syntax-python/) 。这两个值在 Python 中也是单例对象，这意味着每个值只有一个实例。这就是为什么在上面最后的例子中，[标识运算符](https://realpython.com/python-is-identity-vs-equality/) ( `is`)返回`True`。

另一个重要且常见的常量值是 [`None`](https://realpython.com/null-in-python/) ，这是 Python 中的空值。当您想要表达[可空性](https://en.wikipedia.org/wiki/Nullable_type)的想法时，这个常量值就派上了用场。与`True`和`False`一样，`None`也是一个不能被重新分配的单例严格常量对象:

>>>

```py
>>> None is None
True

>>> None = 42
 ...
SyntaxError: cannot assign to None
```

`None`作为函数、方法和类构造函数中的默认参数值非常有用。它通常用于表示变量为空。在内部，Python 使用`None`作为没有[显式`return`语句](https://realpython.com/python-return-statement/#explicit-return-statements)的函数的隐式返回值。

省略号文字(`...`)是 Python 中的另一个常量值。这个特殊值与 [`Ellipsis`](https://realpython.com/python-ellipsis/) 相同，是 [`types.EllipsisType`](https://docs.python.org/3/library/types.html#types.EllipsisType) 类型的唯一实例:

>>>

```py
>>> Ellipsis
Ellipsis

>>> ...
Ellipsis

>>> ... is Ellipsis
True
```

您可以使用`Ellipsis`作为未写代码的占位符。你也可以用它来代替 [`pass`](https://realpython.com/python-pass/) 语句。在类型提示中，`...`文字传达了一个具有统一类型的[未知长度数据集合](https://mypy.readthedocs.io/en/stable/builtin_types.html#generic-types)的思想:

>>>

```py
>>> def do_something():
...     ...  # TODO: Implement this function later
...

>>> class CustomException(Exception): ...
...
>>> raise CustomException("some error message")
Traceback (most recent call last):
    ...
CustomException: some error message

>>> # A tuple of integer values
>>> numbers: tuple[int, ...]
```

在许多情况下，`Ellipsis`常量值可以派上用场，并帮助您使代码更具可读性，因为它在语义上等同于英文省略号标点符号(…)。

另一个有趣且可能有用的内置常量是`__debug__`，正如您在本节开始时已经了解到的。Python 的`__debug__`是一个布尔常量，默认为`True`。它是一个严格的常量，因为一旦解释器运行，就不能改变它的值:

>>>

```py
>>> __debug__
True

>>> __debug__ = False
 ...
SyntaxError: cannot assign to __debug__
```

`__debug__`常数与 [`assert`](https://realpython.com/python-assert-statement/) 语句密切相关。简而言之，如果`__debug__`是`True`，那么你所有的`assert`语句都会运行。如果`__debug__`是`False`，那么您的`assert`语句将被禁用，根本不会运行。这个特性可以稍微提高生产代码的性能。

**注意:**尽管`__debug__`也有一个 dunder 名称，但它是一个严格的常量，因为一旦解释器运行，你就不能改变它的值。相比之下，下一节中的内部数据名称应被视为常量，但不是严格的常量。您可以在代码执行期间更改它们的值。然而，这种做法可能很棘手，需要高深的知识。

要将`__debug__`的值更改为`False`，您必须使用 [`-O`](https://docs.python.org/3/using/cmdline.html#cmdoption-O) 或 [`-OO`](https://docs.python.org/3/using/cmdline.html#cmdoption-OO) 命令行选项在**优化模式**下运行 Python，这提供了两个级别的[字节码](https://docs.python.org/3/glossary.html#term-bytecode)优化。这两个级别都生成不包含断言的优化 Python 字节码。

[*Remove ads*](/account/join/)

### 内部数据名称

Python 也有一组广泛的内部数据名称，您可以将其视为常量。因为有几个这样的特殊名称，所以在本教程中，您将只学习 [`__name__`](https://docs.python.org/3/reference/import.html#name__) 和 [`__file__`](https://docs.python.org/3/reference/import.html#file__) 。

**注意:**要更深入地了解 Python 中的其他 dunder 名称以及它们对该语言的意义，请查看关于 Python 的[数据模型](https://docs.python.org/3/reference/datamodel.html#data-model)的官方文档。

`__name__`属性与您如何运行一段给定的代码密切相关。当导入一个模块时，Python 在内部将`__name__`设置为一个字符串，该字符串包含您正在导入的模块的名称。

启动您的[代码编辑器](https://realpython.com/python-ides-code-editors-guide/)并创建以下示例模块:

```py
# sample_name.py

print(f"The type of __name__ is: {type(__name__)}")
print(f"The value of __name__ is: {__name__}")
```

准备好该文件后，返回命令行窗口并运行以下命令:

```py
$ python -c "import sample_name"
The type of __name__ is: <class 'str'>
The value of __name__ is: sample_name
```

使用`-c`开关，您可以在命令行执行一小段 Python 代码。在这个例子中，您导入了`sample_name`模块，该模块[将一些消息打印到屏幕上。第一条消息告诉你`__name__`的类型是](https://realpython.com/python-print/) [`str`](https://docs.python.org/3/library/stdtypes.html#str) ，或者字符串。第二条消息显示`__name__`被设置为`sample_name`，这是您刚刚导入的模块的名称。

或者，如果您将`sample_name.py`和[作为脚本](https://realpython.com/run-python-scripts/)运行，那么 Python 会将 [`__name__`设置为`"__main__"`](https://realpython.com/if-name-main-python/) 字符串。要确认这一事实，请继续运行以下命令:

```py
$ python sample_name.py
The type of __name__ is: <class 'str'>
The value of __name__ is: __main__
```

注意现在`__name__`保存了`"__main__"`字符串。这种行为表明您已经将该文件作为可执行的 Python 程序直接运行。

`__file__`属性将包含 Python 当前导入或执行的文件的路径。当需要获取模块本身的路径时，可以在给定的模块内部使用`__file__`。

作为`__file__`如何工作的示例，继续创建以下模块:

```py
# sample_file.py

print(f"The type of __file__ is: {type(__file__)}")
print(f"The value of __file__ is: {__file__}")
```

如果您在 Python 代码中导入了`sample_file`模块，那么`__file__`将在您的文件系统中存储其包含模块的路径。通过运行以下命令来检查这一点:

```py
$ python -c "import sample_file"
The type of __file__ is: <class 'str'>
The value of __file__ is: /path/to/sample_file.py
```

同样，如果您将`sample_file.py`作为一个 Python 可执行程序运行，那么您将得到与之前相同的输出:

```py
$ python sample_file.py
The type of __file__ is: <class 'str'>
The value of __file__ is: /path/to/sample_file.py
```

简而言之，Python 将`__file__`设置为包含使用或访问该属性的模块的路径。

[*Remove ads*](/account/join/)

### 有用的字符串和数学常数

你会在标准库中找到许多有用的常数。其中一些与一些特定的模块、函数和类紧密相连。其他的更通用，您可以在各种场景中使用它们。你可以分别在 [`math`](https://realpython.com/python-math-module/) 和 [`string`](https://docs.python.org/3/library/string.html#module-string) 模块中找到的一些数学和字符串相关的常量就是这种情况。

`math`模块提供以下常量:

>>>

```py
>>> import math

>>> # Euler's number (e)
>>> math.e
2.718281828459045

>>> # Pi (π)
>>> math.pi
3.141592653589793

>>> # Infinite (∞)
>>> math.inf
inf

>>> # Not a number (NaN)
>>> math.nan
nan

>>> # Tau (τ)
>>> math.tau
6.283185307179586
```

每当你编写与数学相关的代码，甚至只是使用它们来执行特定计算的代码时，这些常量都会派上用场，就像你在[重用对象以实现可维护性](#reusing-objects-for-maintainability)一节中的`Circle`类一样。

这里有一个使用`math.pi`代替自定义`PI`常量的`Circle`的更新实现:

```py
# circle.py

import math 
class Circle:
    def __init__(self, radius):
        self.radius = radius

    def area(self):
 return math.pi * self.radius**2 
    def perimeter(self):
 return 2 * math.pi * self.radius 
    def projected_volume(self):
 return 4/3 * math.pi * self.radius**3 
    def __repr__(self):
        return f"{self.__class__.__name__}(radius={self.radius})"
```

这个更新版本的`Circle`比您的原始版本更易读，因为它提供了更多关于 Pi 常数来源的上下文，清楚地表明它是一个数学相关的常数。

`math.pi`常量还有一个优点，如果您使用的是旧版本的 Python，那么您将获得 32 位版本的 Pi。相比之下，如果您在现代版本的 Python 中使用`Circle`，那么您将得到 64 位版本的 Pi。因此，您的程序将自适应其具体的执行环境。

`string`模块还定义了几个有用的[字符串常量](https://docs.python.org/3/library/string.html#string-constants)。下表显示了每个常量的名称和值:

| 名字 | 价值 |
| --- | --- |
| `ascii_lowercase` | abcdefghijklmnopqrstuvwxyz |
| `ascii_uppercase` | ABCDEFGHIJKLMNOPQRSTUVWXYZ |
| `ascii_letters` | ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz |
| `digits` | 0123456789 |
| `hexdigits` | 0123456789abcdefABCDEF |
| `octdigits` | 01234567 |
| `punctuation` | ！"#$%&'()*+,-./:;<=>？@[\]^_`{&#124;}~ |
| `whitespace` | [空格](https://en.wikipedia.org/wiki/Space_(punctuation))字符、[横纵制表符](https://en.wikipedia.org/wiki/Tab_key#Tab_characters)、[换行](https://en.wikipedia.org/wiki/Newline)、[回车](https://en.wikipedia.org/wiki/Carriage_return)、[换页](https://en.wikipedia.org/wiki/Page_break#Form_feed)的组合 |
| `printable` | `digits`、`ascii_letters`、`punctuation`和`whitespace`的组合 |

这些与字符串相关的常量在很多情况下都会派上用场。当你进行大量的字符串处理、使用[正则表达式](https://realpython.com/regex-python/)、处理[自然语言](https://realpython.com/nltk-nlp-python/)等等时，你可以使用它们。

## 类型注释常量

从 Python [3.8](https://realpython.com/python38-new-features/) 开始， [`typing`](https://realpython.com/python-type-checking/) 模块包含了一个 [`Final`](https://docs.python.org/3/library/typing.html#typing.Final) 类，允许你对常量进行类型注释。如果你在定义你的常量时使用这个类，那么你将告诉静态类型检查器像 [mypy](https://mypy.readthedocs.io/en/latest/index.html) 你的常量不应该被重新分配。这样，类型检查器可以帮助您检测对常数的未授权赋值。

下面是一些使用`Final`定义常数的例子:

```py
from typing import Final

MAX_SPEED: Final[int] = 300
DEFAULT_COLOR: Final[str] = "\033[1;34m"
ALLOWED_BUILTINS: Final[tuple[str, ...]] = ("sum", "max", "min", "abs")

# Later in your code...
MAX_SPEED = 450  # Cannot assign to final name "MAX_SPEED" mypy(error)
```

`Final`类代表了一个特殊的类型构造，它指示类型检查器在代码中的某个地方重新分配名字时报告一个错误。注意，即使您得到了类型检查器的错误报告，Python 也确实改变了`MAX_SPEED`的值。因此，`Final`并不能防止运行时意外的常量重新分配。

## 在 Python 中定义严格常量

到目前为止，您已经学习了很多关于编程和 Python 常量的知识。您现在知道 Python 不支持严格常量。只是有变数而已。因此，Python 社区采用了使用大写字母来表示给定变量实际上是常数的命名约定。

所以，在 Python 中，你没有常量。相反，你有永不改变的变量。如果您与不同级别的许多程序员一起处理一个大型 Python 项目，这可能是一个问题。在这种情况下，最好有一种机制来保证**严格常数**——在程序启动后没有人可以更改的常数。

因为 Python 是一种非常灵活的编程语言，所以您可以找到几种方法来实现使常量不变的目标。在接下来的几节中，您将了解其中的一些方法。它们都意味着创建一个自定义类，并将其用作常数的命名空间。

为什么应该使用类作为常数的命名空间？在 Python 中，任何名字都可以被随意反弹。在模块级别，您没有适当的工具来防止这种情况发生。所以，你需要使用一个类，因为类比模块提供了更多的定制工具。

在接下来的几节中，您将了解使用类作为严格常量的命名空间的几种不同方式。

[*Remove ads*](/account/join/)

### `.__slots__`属性

Python 类允许你定义一个名为 [`.__slots__`](https://docs.python.org/3/reference/datamodel.html#object.__slots__) 的特殊类属性。该属性将保存一系列名称，这些名称将作为实例属性。

您将无法向具有`.__slots__`属性的类添加新的实例属性，因为`.__slots__`阻止创建实例 [`.__dict__`](https://docs.python.org/3/library/stdtypes.html#object.__dict__) 属性。此外，没有`.__dict__`属性意味着在内存消耗方面的优化。

使用`.__slots__`，您可以创建一个类，作为只读常量的名称空间:

>>>

```py
>>> class ConstantsNamespace:
...     __slots__ = ()
...     PI = 3.141592653589793
...     EULER_NUMBER = 2.718281828459045
...

>>> constants = ConstantsNamespace()

>>> constants.PI
3.141592653589793
>>> constants.EULER_NUMBER
2.718281828459045

>>> constants.PI = 3.14
Traceback (most recent call last):
    ...
AttributeError: 'ConstantsNamespace' object attribute 'PI' is read-only
```

在这个例子中，您定义了`ConstantsNamespace`。该类的`.__slots__`属性包含一个空的[元组](https://realpython.com/python-lists-tuples/)，这意味着该类的实例将没有属性。然后将常量定义为类属性。

下一步是实例化该类，以创建一个变量来保存包含所有常数的名称空间。请注意，您可以快速访问特殊名称空间中的任何常量，但不能给它赋值。如果你尝试去做，你会得到一个`AttributeError`。

使用这种技术，您可以保证团队中的其他人不能更改您的常量的值。您已经实现了严格常数的预期行为。

### `@property`装饰者

你也可以利用 [`@property`](https://realpython.com/python-property/) [装饰器](https://realpython.com/primer-on-python-decorators/)来创建一个类，作为你的常量的命名空间。为此，您只需将常量定义为属性，而无需为它们提供 setter 方法:

>>>

```py
>>> class ConstantsNamespace:
...     @property
...     def PI(self):
...         return 3.141592653589793
...     @property
...     def EULER_NUMBER(self):
...         return 2.718281828459045
...

>>> constants = ConstantsNamespace()

>>> constants.PI
3.141592653589793
>>> constants.EULER_NUMBER
2.718281828459045

>>> constants.PI = 3.14
Traceback (most recent call last):
    ...
AttributeError: can't set attribute 'PI'
```

因为您没有为`PI`和`EULER_NUMBER`属性提供 setter 方法，所以它们是[只读属性](https://realpython.com/python-property/#providing-read-only-attributes)。这意味着你只能*访问*它们的值。不可能给任何一个赋予新的值。如果你尝试去做，你会得到一个`AttributeError`。

### `namedtuple()`工厂功能

Python 的 [`collections`](https://docs.python.org/3/library/collections.html#module-collections) 模块提供了一个[工厂函数](https://en.wikipedia.org/wiki/Factory_(object-oriented_programming))叫做 [`namedtuple()`](https://docs.python.org/3/library/collections.html#collections.namedtuple) 。这个函数允许您创建**元组子类**，允许使用**命名字段**和**点符号**来访问它们的项目，就像在`tuple_obj.attribute`中一样。

像常规元组一样，命名元组实例是[不可变的](https://docs.python.org/3/glossary.html#term-immutable)，这意味着您不能在适当的位置修改现有的命名元组对象[。不可变听起来适合于创建一个作为严格常量的命名空间的类。](https://en.wikipedia.org/wiki/In-place_algorithm)

以下是如何做到这一点:

>>>

```py
>>> from collections import namedtuple

>>> ConstantsNamespace = namedtuple(
...     "ConstantsNamespace", ["PI", "EULER_NUMBER"]
... )
>>> constants = ConstantsNamespace(3.141592653589793, 2.718281828459045)

>>> constants.PI
3.141592653589793
>>> constants.EULER_NUMBER
2.718281828459045

>>> constants.PI = 3.14
Traceback (most recent call last):
    ...
AttributeError: can't set attribute
```

在这个例子中，您的常量在底层命名元组`ConstantsNamespace`中扮演字段的角色。一旦创建了命名元组实例`constants`，就可以通过使用点符号来访问常量，就像在`constants.PI`中一样。

因为元组是不可变的，所以没有办法修改任何字段的值。因此，您的`constants`命名的元组对象是一个完全成熟的严格常量名称空间。

### `@dataclass`装饰者

[数据类](https://realpython.com/python-data-classes/)顾名思义，主要包含数据的类。他们也可以有方法，但这不是他们的主要目标。要创建一个数据类，需要使用 [`dataclasses`](https://docs.python.org/3/library/dataclasses.html#module-dataclasses) 模块中的 [`@dataclass`](https://docs.python.org/3/library/dataclasses.html#dataclasses.dataclass) 装饰器。

如何使用这种类型的类来创建严格常量的命名空间？`@dataclass`装饰器接受一个`frozen`参数，允许您将数据类标记为不可变的。如果它是不可变的，那么一旦创建了给定数据类的实例，就没有办法修改它的实例属性。

下面是如何使用数据类创建包含常量的命名空间:

>>>

```py
>>> from dataclasses import dataclass

>>> @dataclass(frozen=True)
... class ConstantsNamespace:
...     PI = 3.141592653589793
...     EULER_NUMBER = 2.718281828459045
...

>>> constants = ConstantsNamespace()

>>> constants.PI
3.141592653589793
>>> constants.EULER_NUMBER
2.718281828459045

>>> constants.PI = 3.14
Traceback (most recent call last):
    ...
dataclasses.FrozenInstanceError: cannot assign to field 'PI'
```

在这个例子中，首先导入`@dataclass`装饰器。然后使用这个装饰器将`ConstantsNamespace`转换成一个数据类。为了使数据类不可变，您将`frozen`参数设置为`True`。最后，用常量作为类属性定义`ConstantsNamespace`。

您可以创建该类的一个实例，并将其用作您的常量命名空间。同样，您可以访问所有常量，但不能修改它们的值，因为数据类是冻结的。

### `.__setattr__()`特殊方法

Python 类让你定义一个叫做 [`.__setattr__()`](https://docs.python.org/3/reference/datamodel.html#object.__setattr__) 的特殊方法。该方法允许您自定义属性赋值过程，因为 Python 会在每次属性赋值时自动调用该方法。

实际上，您可以覆盖`.__setattr__()`来防止所有的属性重新分配，并使您的属性不可变。下面是如何重写此方法来创建一个类，作为常数的命名空间:

>>>

```py
>>> class ConstantsNamespace:
...     PI = 3.141592653589793
...     EULER_NUMBER = 2.718281828459045
...     def __setattr__(self, name, value):
...         raise AttributeError(f"can't reassign constant '{name}'")
...

>>> constants = ConstantsNamespace()

>>> constants.PI
3.141592653589793
>>> constants.EULER_NUMBER
2.718281828459045

>>> constants.PI = 3.14
Traceback (most recent call last):
    ...
AttributeError: can't reassign constant 'PI'
```

您的自定义实现`.__setattr__()`不在类的属性上执行任何赋值操作。当您试图设置任何属性时，它只会引发一个`AttributeError`。这种实现使得属性不可变。同样，您的`ConstantsNamespace`表现为常量的名称空间。

## 结论

现在你知道什么是**常量**，以及为什么和什么时候在你的代码中使用它们。你也知道 Python 没有严格的常量。Python 社区使用*大写字母*作为命名约定来传达变量应该作为常量使用。这种命名约定有助于防止其他开发人员更改应该是常量的变量。

常量在编程中无处不在，Python 开发人员也在使用它们。所以，学习在 Python 中定义和使用常量是你需要掌握的一项重要技能。

**在本教程中，您学习了如何:**

*   在代码中定义 **Python 常量**
*   识别并理解一些**内置常数**
*   用常量提高代码的**可读性**、**可重用性**和**可维护性**
*   使用不同的策略来组织和管理现实项目中的常量
*   应用各种技术使你的 Python 常量**严格恒定**

了解了什么是常量，为什么它们很重要，以及何时使用它们，您就可以立即开始改进代码的可读性、可维护性和可重用性了。来吧，试一试！

**示例代码:** [点击此处下载示例代码](https://realpython.com/bonus/python-constants-code/)，向您展示如何在 Python 中使用常量。***************